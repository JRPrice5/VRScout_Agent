#!/usr/bin/env python
"""
Unity VR GAIL Training Interface

- Receives states from Unity (direct or wrapped format)
- Samples actions from ACT policy (unsquashed Gaussian + Bernoulli)
- Sends denormalized actions back to Unity
- Accumulates rollouts and triggers a callback when a batch is ready

Messages:
- State (direct): {"qpos": [...], "image": <base64 or raw bytes>, "timestep": int}
- State (wrapped): {"type":"state", "data": {...}}
- Episode control: {"type":"episode_control", "done": true}

Delimiter: "END_OF_MESSAGE"
Port: default 5557 (separate from inference 5556)
"""
import os
import math
import sys
import io
import json
import time
import base64
import socket
import threading
from typing import Callable, Optional, Dict

import numpy as np
import torch
from PIL import Image

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_utils import ACTION_INDICES
from action_io import preprocess_qpos, postprocess_action_to_env
from rl.rollout import RolloutBuffer
from distributions import TanhDiagNormal


class UnityVRGAILInterface:
    def __init__(self,
                 policy,
                 stats: Dict,
                 host: str = '127.0.0.1',
                 port: int = 5557,
                 update_every: int = 1024,
                 query_frequency: int = 1,
                 on_batch: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None,
                 action_mode: str = 'sample',
                 mean_temperature: float = 0.25):
        self.policy = policy.eval().cuda()
        self.stats = stats
        self.host = host
        self.port = port
        self.update_every = update_every
        self.query_frequency = query_frequency
        self.on_batch = on_batch
        assert action_mode in ('sample', 'mean', 'ahat', 'mean_low_noise'), "action_mode must be 'sample', 'mean', 'ahat', or 'mean_low_noise'"
        self.action_mode = action_mode
        self.mean_temperature = float(mean_temperature)

        self.socket = None
        self.client_socket = None
        self.running = False
        self.buffer = RolloutBuffer()
        self.boolean_indices = ACTION_INDICES['boolean']

        self.internal_timestep = 0
        self._mark_prev_done = False
        # Cached multi-step chunk sampled from policy; consumed across timesteps
        # Structure: {'a_cont': [1,T,Cc], 'a_bin': [1,T,Cb] or None, 'logp': [1,T], 'value': [1,T], 'idx': int, 'len': int}
        self._chunk_cache = None

    # ---------- Networking ----------
    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(1.0)
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"[GAIL] Server on {self.host}:{self.port}")
            self.running = True
            while self.running:
                try:
                    client_sock, addr = self.socket.accept()
                    print(f"[GAIL] Client connected: {addr}")
                    self.client_socket = client_sock
                    self._handle_client()
                except socket.timeout:
                    continue
                except Exception as e:
                    # Keep server alive on unexpected client handling errors
                    print(f"[GAIL] Client handling error: {e}")
                    try:
                        if self.client_socket:
                            self.client_socket.close()
                    except Exception:
                        pass
                    self.client_socket = None
        finally:
            self.stop_server()

    def stop_server(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    # ---------- Client handling ----------
    def _handle_client(self):
        self.internal_timestep = 0
        self._chunk_cache = None
        recv_buf = b''
        try:
            while self.running:
                try:
                    chunk = self.client_socket.recv(4096)
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    # Remote closed or socket error: exit client loop gracefully
                    print(f"[GAIL] recv error (client likely disconnected): {e}")
                    break
                if not chunk:
                    break
                recv_buf += chunk
                while True:
                    eom = recv_buf.find(b"END_OF_MESSAGE")
                    if eom == -1:
                        break
                    packet = recv_buf[:eom]
                    recv_buf = recv_buf[eom + len(b"END_OF_MESSAGE"):]
                    try:
                        msg = json.loads(packet.decode('utf-8'))
                    except Exception as e:
                        print(f"[GAIL] JSON decode error: {e}")
                        continue
                    self._process_message(msg)
        finally:
            print("[GAIL] Client disconnected")
            # Ensure client socket is cleaned up so server can accept new clients
            try:
                if self.client_socket:
                    try:
                        self.client_socket.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None

    # ---------- Message processing ----------
    def _process_message(self, msg: Dict):
        # Accept both wrapped and direct format
        if isinstance(msg, dict) and msg.get('type') == 'episode_control':
            if msg.get('done', False):
                # Mark episode done for next transition; here we just note but do not need to send special action
                self._mark_done()
            return

        if isinstance(msg, dict) and msg.get('type') == 'state':
            data = msg.get('data', {})
        else:
            data = msg

        # Expect qpos + image
        if 'qpos' not in data or 'image' not in data:
            return

        # Perform policy inference and buffer storage without autograd
        with torch.inference_mode():
            qpos_np = np.array(data['qpos'], dtype=np.float32)
            qpos = preprocess_qpos(qpos_np, self.stats).cuda().unsqueeze(0)
            image = self._process_image_data(data)

            # Prepare or consume a cached multi-step chunk depending on query_frequency
            need_new_chunk = (
                self._chunk_cache is None or
                self._chunk_cache.get('idx', 0) >= self._chunk_cache.get('len', 0)
            )

            if need_new_chunk: # TODO  overlapping chunks for perfromance?
                # Query policy once to obtain a sequence; then emit step-by-step over timesteps
                out = self.policy.forward(qpos, image, out_mode='dist')
                cont_dist = out['cont_dist']
                bin_dist = out.get('bin_dist', None)
                value_seq = out['value']  # [1, T]
                a_hat = out.get('a_hat', None)  # [1, T, A]

                # Choose action generation mode and compute matching log-probs
                if self.action_mode == 'ahat' and a_hat is not None:
                    a_cont_seq = a_hat[:, :, ACTION_INDICES['continuous']]
                    a_bin_seq = a_hat[:, :, ACTION_INDICES['boolean']] if bin_dist is not None else None
                    # Ensure continuous actions are in (-1,1) for TanhDiagNormal
                    # Using tanh is safe; also clamp away from boundaries to avoid numerical issues
                    a_cont_seq = torch.tanh(a_cont_seq).clamp_(-0.999, 0.999)
                    # Compute logp under current dists at the executed action
                    logp_seq = cont_dist.log_prob(a_cont_seq)
                    if bin_dist is not None:
                        logp_seq = logp_seq + bin_dist.log_prob(a_bin_seq)
                elif self.action_mode == 'mean':
                    a_cont_seq = cont_dist.mean
                    a_bin_seq = bin_dist.mean if bin_dist is not None else None
                    # For deterministic booleans, use hard 0/1 for env but compute logp at the same hard value
                    if a_bin_seq is not None:
                        a_bin_seq = (a_bin_seq > 0.5).float()
                    logp_seq = cont_dist.log_prob(a_cont_seq)
                    if bin_dist is not None:
                        logp_seq = logp_seq + bin_dist.log_prob(a_bin_seq)
                elif self.action_mode == 'mean_low_noise':
                    # Sample with reduced std for continuous; sample booleans
                    # Use current cont_dist to compute logp on the sampled action
                    if hasattr(cont_dist, 'mu') and hasattr(cont_dist, 'log_std'):
                        mu = cont_dist.mu
                        log_std = cont_dist.log_std + math.log(max(self.mean_temperature, 1e-3))
                        # Rebuild a temp distribution matching cont_dist type
                        cont_temp = TanhDiagNormal(mu, log_std)
                        a_cont_seq = cont_temp.sample()
                    else:
                        a_cont_seq = cont_dist.sample()  # fallback
                    a_bin_seq = bin_dist.sample() if bin_dist is not None else None
                    logp_seq = cont_dist.log_prob(a_cont_seq)
                    if bin_dist is not None:
                        logp_seq = logp_seq + bin_dist.log_prob(a_bin_seq)
                else:  # 'sample'
                    a_cont_seq = cont_dist.sample()
                    logp_seq = cont_dist.log_prob(a_cont_seq)
                    a_bin_seq = None
                    if bin_dist is not None:
                        a_bin_seq = bin_dist.sample()
                        logp_seq = logp_seq + bin_dist.log_prob(a_bin_seq)

                # Limit the usable horizon to query_frequency to control emission span
                # Limit by available temporal horizon
                T = a_cont_seq.shape[1]
                use_T = int(min(self.query_frequency, T))
                if use_T <= 0:
                    use_T = 1

                self._chunk_cache = {
                    # Store actions (exactly what we execute and compute logp for)
                    'a_cont': a_cont_seq[:, :use_T],
                    'a_bin': a_bin_seq[:, :use_T] if a_bin_seq is not None else None,
                    'logp': logp_seq[:, :use_T],
                    'value': value_seq[:, :use_T],
                    'idx': 0,
                    'len': use_T,
                }

            # Consume one step from cache
            t_idx = self._chunk_cache['idx']
            a_cont = self._chunk_cache['a_cont'][:, t_idx]        # [1, Cc]
            a_bin = None if self._chunk_cache['a_bin'] is None else self._chunk_cache['a_bin'][:, t_idx]  # [1, Cb]
            logp = self._chunk_cache['logp'][:, t_idx]            # [1]
            value = self._chunk_cache['value'][:, t_idx]          # [1]
            self._chunk_cache['idx'] = t_idx + 1

            # If previous transition should be marked done, backpatch last dones to 1
            if self._mark_prev_done and len(self.buffer.storage['dones']) > 0:
                try:
                    self.buffer.storage['dones'][-1][...] = 1.0
                except Exception:
                    pass
                self._mark_prev_done = False

            # Store transition in rollout buffer (mask=1 valid step, dones default 0)
            # Ensure tensors include an explicit time dimension T=1
            mask = torch.ones_like(value).unsqueeze(1)   # [1,1]
            dones = torch.zeros_like(value).unsqueeze(1) # [1,1]
            a_cont_t = a_cont.unsqueeze(1)               # [1,1,Cc]
            a_bin_t = None if a_bin is None else a_bin.unsqueeze(1)  # [1,1,Cb]
            logp_t = logp.unsqueeze(1)                   # [1,1]
            value_t = value.unsqueeze(1)                 # [1,1]

            self.buffer.add(
                obs={'qpos': qpos, 'image': image},
                a_cont=a_cont_t,
                a_bin=a_bin_t,
                logp=logp_t,
                value=value_t,
                dones=dones,
                mask=mask,
            )

            # Send action back to Unity (denormalized + sigmoid booleans)
            # Merge cont/bin into full vector in normalized space
            # Here we assume action head outputs are already normalized space predictions
            # For env, we postprocess including boolean sigmoid + denorm
            if a_bin is not None:
                # Merge continuous and boolean sampled actions into full vector
                A = len(self.stats['action_mean'])
                a_full = torch.zeros((A,), dtype=a_cont.dtype, device=a_cont.device)
                a_full[ACTION_INDICES['continuous']] = a_cont.squeeze(0)
                a_full[ACTION_INDICES['boolean']] = a_bin.squeeze(0)
            else:
                a_full = a_cont.squeeze(0)

            action_env = postprocess_action_to_env(a_full, self.stats, self.boolean_indices)
            self._send_action(action_env)

            self.internal_timestep += 1
            # Decide whether to emit a batch (outside of inference_mode)
            emit_now = (len(self.buffer) >= self.update_every and self.on_batch is not None)

        # If a batch is ready, finalize and call the callback with autograd enabled
        if 'emit_now' in locals() and emit_now:
            batch = self._finalize_and_reset_buffer()
            # Ensure gradients are enabled during training steps inside on_batch
            with torch.enable_grad():
                self.on_batch(batch)

    def _mark_done(self):
        # Backpatch last transition's done flag; new episode starts with next state
        self._mark_prev_done = True
        self.internal_timestep = 0
        self._chunk_cache = None

    def _send_action(self, action_np: np.ndarray):
        try:
            payload = {"action": action_np.tolist(), "timestep": self.internal_timestep}
            blob = (json.dumps(payload) + "END_OF_MESSAGE").encode('utf-8')
            self.client_socket.sendall(blob)
        except Exception as e:
            print(f"[GAIL] send action error: {e}")

    # ---------- Image processing (mirrors inference) ----------
    def _process_image_data(self, state_data: Dict) -> torch.Tensor:
        image_format = state_data.get('imageFormat', 'raw')
        is_raw = image_format == 'raw'
        width = state_data.get('imageWidth', 224)
        height = state_data.get('imageHeight', 224)

        def decode_single(img_payload):
            if is_raw and isinstance(img_payload, list):
                arr = np.asarray(img_payload, dtype=np.uint8)
                img = Image.fromarray(arr.reshape((height, width, 3)))
            else:
                if isinstance(img_payload, str):
                    img_bytes = base64.b64decode(img_payload)
                else:
                    img_bytes = bytes(img_payload)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Ensure a writable, contiguous NumPy array for torch.from_numpy
            img_np = np.array(img, dtype=np.uint8, copy=True)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # [C,H,W]
            return img_t

        image_field = state_data['image']
        if isinstance(image_field, dict):  # multi-camera
            frames = [decode_single(image_field[k]) for k in sorted(image_field.keys())]
            img = torch.stack(frames, dim=0)  # [K,C,H,W]
        else:
            img = decode_single(image_field).unsqueeze(0)  # [1,C,H,W]

        img = img.unsqueeze(0)  # [1,K,C,H,W]
        img = img.float() / 255.0
        return img.cuda()

    # ---------- Buffer finalize ----------
    def _finalize_and_reset_buffer(self) -> Dict[str, torch.Tensor]:
        self.buffer.finalize()
        batch = self.buffer.tensors
        self.buffer = RolloutBuffer()
        return batch
