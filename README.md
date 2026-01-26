# VRScout - an ACT Style VR Testing & Gameplay Agent

This repository provides the **VR-side policy stack** for the **VR Agent** system (this is **one of the three repos** that together make up the VR Agent).

You can find the data collection and Agent_Orchestration_Unity_Project here:
https://github.com/wrainw/VRScout_Agent_Orchestration_Unity_Project

And you can find the virtual devices driver code here:
https://github.com/wrainw/VRScout_openvr_virtualdriver

You can also find the WASAPI audio capture wrapper code here:
https://github.com/wrainw/WASAPI_AudioCaptureWrapper

At a high level, this repo contains:

- A Python **inference server** (`unity_vr_inference.py`) that loads an ACT-style policy and serves actions to Unity over TCP.
- It enables a Unity C# **IPC clients** (`unity/InferencePoseClient.cs`, etc.) that capture VR state (poses / images / optional audio), to apply returned actions to OpenVR Applications (via the virtual driver).
- A data preprocessing pipeline (`batch_convert_new_vr_datasets.py`) that prepares the collected VR record data for training.
- And Training utilities (`imitate_episodes.py`) extended based on **ACT (Action Chunking with Transformers)** for vr settings and performance.

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
 
 Please refer to the **Agent_Orchestration_Unity_Project** for the current Unity integration.

## Notes

- The inference server expects a checkpoint in `--ckpt_dir` named `policy_best.ckpt`.
- If you run into mismatch issues, ensure `--chunk_size` matches the model’s training config.

***

## Attribution

This project builds upon and is inspired by the following prior works and systems:

  ACT: Action Chunking with Transformers

    This implementation is ACT-based, drawing on the ideas and methodology introduced in:

    Tony Z. Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, 2023.
    Licensed under the MIT License.

    ACT provides the foundational approach for temporally chunked action prediction used in this policy implementation.

  DETR: DEtection TRansformer

    This project incorporates architectural concepts and components derived from:

    DETR (DEtection TRansformer),
    Copyright © Facebook, Inc. and its affiliates.
    Licensed under the Apache License, Version 2.0.

    DETR-inspired transformer structures are used for perception and sequence modeling.

  VRScout: VR Testing and Gameplay Agent

    This work is developed as part of the VRScout Agent system, an autonomous VR interaction and testing agent designed for embodied policy learning and evaluation in virtual reality environments.

## License Notes

This repository is licensed under the Apache License, Version 2.0.

Third-party components retain their original licenses:

ACT — MIT License

DETR — Apache License 2.0

See LICENSE, NOTICE, and THIRD_PARTY_NOTICES.txt for details.

## Citations

If you use this code, please consider citing:

```bibtex
@article{carion2020endtoend,
  title={End-to-End Object Detection with Transformers},
  author={Carion, N. and Massa, F. and Synnaeve, G. and Usunier, N. and Kirillov, A. and Zagoruyko, S.},
  journal={arXiv preprint arXiv:2005.12872},
  year={2020},
  url={https://arxiv.org/abs/2005.12872}
}

@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, T. Z.},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023},
  url={https://arxiv.org/abs/2304.13705}
}

@article{wu2025vrscout,
  title={VRScout: Towards Real-Time, Autonomous Testing of Virtual Reality Games},
  author={Wu, Y. and Sun, Y. and Wunsche, B. and Wang, J. and Wen, E.},
  journal={arXiv preprint arXiv:2511.00002},
  year={2025},
  url={https://arxiv.org/abs/2511.00002}
}
```
