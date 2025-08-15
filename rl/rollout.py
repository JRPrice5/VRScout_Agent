from __future__ import annotations
from typing import Dict, Iterator, List, Tuple
import torch


class RolloutBuffer:
    """
    Stores variable-length sequence rollouts with masks and attention masks.
    Fields per add(): obs, prev_actions, z, a_cont, a_bin, logp, value, dones, mask, attn_mask
    Exposes finalize() to stack into tensors and iter_minibatches(batch_size)
    """
    def __init__(self):
        self.storage: Dict[str, List] = {
            'obs': [],  # list of dicts {'qpos': Tensor, 'image': Tensor}
            'prev_actions': [],
            'z': [],
            'a_cont': [],
            'a_bin': [],
            'logp': [],
            'value': [],
            'dones': [],
            'mask': [],
            'attn_mask': [],
        }
        self.finalized = False
        self.tensors: Dict[str, torch.Tensor] = {}

    def add(self, **kwargs):
        for k in self.storage.keys():
            if k in kwargs and kwargs[k] is not None:
                self.storage[k].append(kwargs[k])

    def finalize(self):
        assert not self.finalized
        for k, lst in self.storage.items():
            if len(lst) == 0:
                continue
            if k == 'obs':
                # lst is list of dicts
                qpos_list = [d['qpos'] for d in lst]
                image_list = [d['image'] for d in lst]
                self.tensors['obs'] = {
                    'qpos': torch.cat(qpos_list, dim=0),
                    'image': torch.cat(image_list, dim=0),
                }
            else:
                self.tensors[k] = torch.cat(lst, dim=0)
        self.finalized = True

    def __len__(self):
        if not self.finalized:
            return sum(x.size(0) for x in self.storage['mask']) if self.storage['mask'] else 0
        return self.tensors['mask'].size(0)

    def iter_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        assert self.finalized
        N = self.tensors['mask'].size(0)
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch: Dict[str, torch.Tensor] = {}
            for k, v in self.tensors.items():
                if k == 'obs':
                    batch['obs'] = {
                        'qpos': v['qpos'][idx],
                        'image': v['image'][idx],
                    }
                else:
                    batch[k] = v[idx]
            yield batch
