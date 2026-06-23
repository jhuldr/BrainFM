<div align="center">
  <h1>BrainFM<br><sub>A Modality-agnostic Multi-task Foundation Model for Human Brain Imaging</sub></h1>
</div>

<p align="center">
  [<a href="https://arxiv.org/abs/2509.00549">arXiv</a>]
</p>

<p align="center">
<b align="center">Contact: Peirong Liu (peirong@jhu.edu)</b>
</p>

<p align="center">
Department of Electrical and Computer Engineering,<br/>
Data Science and AI Institute,<br/>
Johns Hopkins University
</p>


BrainFM is a unified foundation model for multi-task brain MRI analysis, including anatomy synthesis, segmentation, registration, bias-field estimation, cortical distance maps, and more. This repository provides inference demos, a synthetic data generator, and training code.

## Installation

Tested with Python 3.11.4, PyTorch 2.0.1, and CUDA 12.2.

```bash
conda create -n brainfm python=3.11
conda activate brainfm

git clone https://github.com/jhuldr/BrainFM
cd BrainFM
pip install -r requirements.txt
```

Optional: install the package in editable mode.

```bash
pip install -e .
```

## Pretrained weights

Download pretrained weights from [OneDrive](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pliu53_jh_edu/IgBQQLRuGowgS4aiSlHh86hCAd8erNM2383oNJDaKN1zA8A?e=1MPTQu) and place them under `ckp/`. The inference demos expect:

```
ckp/brainfm_pretrained.pth
```

## Quick start: inference

Run all commands from the repository root.

1. Place test volumes in `test_data_folder/` (or update the path in the script).
2. Run the demo:

```bash
python scripts/demo_test.py
```

Results are written to `outs/test_results/`. The script supports whole-volume and tiled inference on large volumes.

### Feature extraction only

```bash
python scripts/demo_get_feature.py
```

Update `img_path` in the script to point at your input volume. This returns a 64-channel feature map from the pretrained encoder.

## Synthetic data generator

Visualize or debug the training data pipeline:

```bash
python scripts/demo_generator.py
```

Generator configs live in `cfgs/generator/`. `default.yaml` defines shared settings; task-specific configs in `train/` and `test/` override those defaults.

Key generator settings:


| Setting              | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `dataset_names`      | Datasets to sample from (paths in `Generator/constants.py`) |
| `dataset_option`     | `default` (`BaseGen`) or `brain_id` (`BrainIDGen`)          |
| `mix_synth_prob`     | Probability of blending synthetic with real images          |
| `task`               | Toggle individual training tasks on/off                     |
| `augmentation_steps` | Augmentation pipeline per input type                        |


Dataset paths, augmentation functions, and per-modality processing steps are defined in `Generator/constants.py`. See `BrainIDGen` in `Generator/datasets.py` for a customized generator example.

## Training

Training is launched from the repo root with a generator config and a trainer config:

```bash
python scripts/train.py \
  cfgs/generator/train/brain_id.yaml \
  cfgs/trainer/train/joint.yaml
```

SLURM helpers are provided in `scripts/train.sh` and `scripts/test.sh`. Update cluster-specific settings there before submitting jobs.

Trainer configs are under `cfgs/trainer/`. Defaults are in `default_train.yaml` and `default_val.yaml`; experiment configs in `train/` override them.

## Project structure

```
BrainFM/
├── assets/              # Figures for documentation
├── cfgs/                # Generator and trainer YAML configs
├── ckp/                 # Pretrained checkpoints (not tracked)
├── files/               # Atlas and local test volumes (e.g. gca.mgz)
├── Generator/           # Data loading, augmentation, and synthesis
├── scripts/             # Demos, training, and evaluation entry points
├── ShapeID/             # Shape synthesis and PDE-based deformation
├── Trainer/             # Model, losses, training engine
└── utils/               # Config, logging, interpolation, and I/O helpers
```

## Citation

```bibtex
@article{Liu_2025_BrainFM,
    author    = {Liu, Peirong and Puonti, Oula and Hu, Xiaoling and Gopinath, Karthik and Sorby-Adams, Annabel and Alexander, Daniel C. and Iglesias, Juan E.},
    title     = {A Modality-agnostic Multi-task Foundation Model for Human Brain Imaging},
    booktitle = {arXiv preprint arXiv:2509.00549},
    year      = {2025},
}
```

## Copyright

"A Modality-agnostic Multi-task Foundation Model for Human Brain Imaging" is a publication of The Johns Hopkins University and copyright © 2026 The Johns Hopkins University. All rights reserved.
