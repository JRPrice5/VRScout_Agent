# VRScout - an ACT Style VR Testing & Gameplay Agent

This repository provides the **VR-side policy stack** for the **VR Agent** system (this is **one of the three repos** that together make up the VR Agent).

At a high level, this repo contains:

- A Python **inference server** (`unity_vr_inference.py`) that loads an ACT-style policy and serves actions to Unity over TCP.
- Unity C# **IPC clients** (`unity/InferencePoseClient.cs`, etc.) that capture VR state (poses / images / optional audio), send it to Python, and apply returned actions.
- Training/evaluation utilities have been extended based on **ACT (Action Chunking with Transformers)**.

## Major Features

- **Unity ↔ Python realtime inference (TCP)**
  - Unity streams observations (poses + camera images, optional audio) to Python.
  - Python runs policy inference and streams back actions.

- **Temporal aggregation / ensembling**
  - Optional temporal aggregation of action chunks for smoother control (`--temporal_agg`).

- **Dynamic temporal aggregation (adaptive smoothing vs responsiveness)**
  - Optional **dynamic window sizing** (`--use_dynamic_window`) to adjust smoothing online.
  - Note: A model with a longer chunk size is recommended to be used with dynamic window sizing.

- **Runtime toggles from Unity UI**
  - Unity can toggle dynamic components live (window size / decay factor / update rate).
  - Python sends current toggle states back in the response for bidirectional sync.

- **Persistent adaptive normalization stats**
  - Dynamic-window normalization stats are persisted to `normalization_stats.pkl` under `--ckpt_dir` so tuning carries across runs.

- **Optional audio features (YAMNet)**
  - When enabled, the server extracts audio embeddings via TensorFlow Hub YAMNet and includes them in inference.

## Repo Entry Points

- `unity_vr_inference.py`
  - Main Python inference server used during VR runtime.
- `unity/InferencePoseClient.cs`
  - Main Unity client for pose + image capture, networking, UI toggles, and applying actions.
- `imitate_episodes.py`, `evaluate_episodes.py`
  - Training/evaluation scripts for ACT-style policies.
- `detr/`, `policy.py`
  - Model implementation and policy adapter.

## Setup

Create a Python 3.10 environment:

```bash
conda create -n py310 python=3.10
conda activate py310
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you hit install/runtime issues, the most common cause is **incompatible `torch` / `triton` / CUDA** combinations.
Make sure you pick a `torch` build that matches your CUDA version, and ensure the corresponding `triton` version is compatible with that `torch` build.

## Running the VR Inference Server

Start the server (defaults to `127.0.0.1:5556`):

```bash
python unity_vr_inference.py \
  --ckpt_dir <path-to-checkpoint-dir> \
  --policy_class ACT \
  --chunk_size 8 \
  --temporal_agg
```

Dynamic temporal aggregation example:

```bash
python unity_vr_inference.py \
  --ckpt_dir <path-to-checkpoint-dir> \
  --temporal_agg \
  --use_dynamic_window
```

Optional audio:

```bash
python unity_vr_inference.py \
  --ckpt_dir <path-to-checkpoint-dir> \
  --use_audio \
  --input_sample_rate 48000
```

## Unity Integration

 The legacy Unity-side scripts under `unity/` are **deprecated**.
 
 Please refer to the **Unity Component Repo** for the current Unity integration.

## Notes

- The inference server expects a checkpoint in `--ckpt_dir` named `policy_best.ckpt`.
- If you run into mismatch issues, ensure `--chunk_size` matches the model’s training config.

***

Based on Action Chunking with Transformers and extended for VR runtime inference and Unity integration.
