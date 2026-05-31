# EmCom-Diffusion: Probing Visual Reflection in Emergent Languages via Image Generation

**[ICONIP 2026]** Haruumi Omoto and Tadahiro Taniguchi  
Graduate School of Informatics, Kyoto University

> **Visual reflection** is the extent to which emergent messages preserve information about their source images that can be recovered without appeal to the speaker–listener pair that produced them. EmCom-Diffusion measures this property by fine-tuning a text-to-image diffusion model on (image, emergent-message) pairs and quantifying the perceptual similarity between the generated and original image.

---

## Overview

EmCom-Diffusion consists of three steps:

1. **(Phase 1)** Train a Referential Game on MS-COCO to obtain emergent-language messages per image.
2. **(Phase 2)** Fine-tune Stable Diffusion v1.5 on (image, emergent-message) pairs.
3. **(Phase 3–7)** Evaluate the generated images and compare against existing metrics (CBM, Translation, TopSim, R@1).

## Requirements

```bash
pip install -r requirements.txt
```

PyTorch must be installed separately following [pytorch.org](https://pytorch.org). DINOv2 is loaded via `torch.hub` from `facebookresearch/dinov2` at runtime.

**Environment variables** (set before running any script):

```bash
export COCO_ROOT=/path/to/coco2017      # root of MS-COCO 2017 dataset
export PROJ_DIR=/path/to/emcom-diffusion # repo root (for Phase 5–7 scripts)
```

---

## Repository Structure

```
emcom-diffusion/
├── ec_game/                    # Phase 1: Referential Game
│   ├── models/
│   │   ├── emcom.py            # Speaker (cross-attn) + Receiver (BERT)
│   │   ├── vit.py              # DINOv2 wrapper
│   │   ├── med.py              # BERT implementation
│   │   └── blip_utils.py       # create_vit helper
│   ├── configs/
│   │   ├── config_referential.yaml
│   │   └── bert_config.json
│   ├── train.py                # Referential Game training
│   └── extract_corpus.py       # Extract EC corpus from checkpoint
│
├── diffusion/                  # Phase 2: SD fine-tuning & inference
│   ├── train.py                # Fine-tune SD v1.5 with EC tokens
│   ├── train_nl.py             # Fine-tune SD v1.5 with NL captions (SD-NL baseline)
│   ├── inference.py            # Generate images (EC / Random / Fixed / NL)
│   ├── ec_text_encoder.py      # Custom CLIP text encoder for EC tokens
│   ├── data.py                 # Dataset classes
│   └── utils.py                # Shared utilities
│
├── experiments/                # Phase 3–7: Evaluation
│   ├── evaluate.py             # Table 2: CLIP-text, FID (Phase 3)
│   ├── exp_b1_multi_encoder.py # Table 2: CLIP-img, DINOv2, SigLIP (Phase 3)
│   ├── exp_a1_generate_multiseed.py  # Table 3 preprocessing (Phase 3)
│   ├── exp_a1_eval_diversity.py      # Table 3: multi-seed diversity (Phase 3)
│   ├── exp_a2_distribution_metrics.py # Vendi, Recall (Phase 3)
│   ├── cache_dino_gen.py       # Cache DINOv2 embeddings of generated images (Phase 5)
│   ├── run_A_triplet.py        # Table 4 point estimates (Phase 6)
│   ├── run_B_step2_triplet.py  # Table 5 point estimates (Phase 6)
│   ├── run_distractor_sensitivity.py # Table 6 point estimates (Phase 6)
│   ├── eval_r1_val5k.py        # Appendix D R@1 (Phase 6)
│   └── bootstrap_paper.py      # Tables 4–6 final CI (Phase 7)
│
└── translation/                # Phase 4: EC→NL translation baseline
    ├── vocab.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── translate.py
```

---

## Step-by-Step Reproduction

### Phase 1 — Referential Game

Prepare an image-list JSON: a list of `{"image": "/path/to/img.jpg"}` entries for all MS-COCO 2017 training images.

```bash
# Train (4-GPU example)
torchrun --nproc_per_node=4 -m ec_game.train \
    --config ec_game/configs/config_referential.yaml \
    --img_list data/coco_image_list.json \
    --output_dir outputs/ec_game

# Extract EC corpus from the epoch-29 checkpoint
python -m ec_game.extract_corpus \
    --ckpt    outputs/ec_game/referential_YYYYMMDD/checkpoint_29.pth \
    --ec_json data/ec_captions.json \
    --output  outputs/ec_corpus.json
```

`ec_captions.json` maps each image to its split (`train` / `val`) and path.

### Phase 2 — SD Fine-tuning & Inference

```bash
# Fine-tune SD v1.5 with EC tokens (setting B: embedding + U-Net LoRA)
accelerate launch -m diffusion.train --config configs/setting_b.yaml

# Fine-tune SD v1.5 with NL captions (SD-NL baseline)
accelerate launch diffusion/train_nl.py --config configs/setting_nl.yaml

# Generate images for all conditions (EC / Random / Fixed)
python -m diffusion.inference \
    --ckpts  checkpoints/pilot_B/step_010000 \
    --labels B \
    --n_samples 5000 --input_type ec \
    --ec_json outputs/ec_corpus.json \
    --out_img_dir outputs/generated_images
```

### Phase 3 — Main Evaluation (Table 2 & 3)

```bash
# CLIP-text and FID
python experiments/evaluate.py \
    --ec_json  outputs/ec_corpus.json \
    --gen_dirs outputs/generated_images/B \
    --rand_dirs outputs/generated_images/B_random \
    --fixed_dirs outputs/generated_images/B_fixed \
    --nl_ft_dir  outputs/generated_images/NL_ft \
    --labels B --n_samples 5000 \
    --out_csv outputs/pilot_metrics.csv

# CLIP-img (ViT-B/32), DINOv2, SigLIP  →  Table 2 similarity columns
python experiments/exp_b1_multi_encoder.py \
    --ec_json  outputs/ec_corpus.json \
    --gen_dir  outputs/generated_images \
    --out_csv  outputs/b1_multi_encoder.csv \
    --n_samples 5000

# Multi-seed generation and diversity analysis  →  Table 3
python experiments/exp_a1_generate_multiseed.py \
    --ec_json  outputs/ec_corpus.json \
    --ec_ckpt  checkpoints/pilot_B/step_010000 \
    --out_dir  outputs/multiseed \
    --n_samples 500 --n_seeds 8

python experiments/exp_a1_eval_diversity.py \
    --multiseed_dir outputs/multiseed \
    --out_csv outputs/a1_diversity.csv
```

### Phase 4 — Translation Baseline

```bash
python -m translation.train \
    --ec_corpus outputs/ec_corpus.json \
    --coco_root $COCO_ROOT \
    --vocab     outputs/translation/vocab.json \
    --ckpt_dir  outputs/translation/checkpoints \
    --epochs 10

python -m translation.translate \
    --ec_corpus  outputs/ec_corpus.json \
    --ckpt       outputs/translation/checkpoints/best.pt \
    --vocab      outputs/translation/vocab.json \
    --output     outputs/translation/translations_val5k.json \
    --clip_cache outputs/cache_pred_clip_text_vitl14.npy
```

### Phase 5 — Cache DINOv2 Embeddings of Generated Images

```bash
python experiments/cache_dino_gen.py \
    --gen_dir outputs/generated_images/B \
    --ec_json outputs/ec_corpus.json \
    --output  outputs/cache_dino_vitb14_gen.npy
```

### Phase 6 — Comparison Experiments (Point Estimates)

```bash
# Table 4: Translation vs EmCom-Diffusion
python experiments/run_A_triplet.py   # reads caches from $PROJ_DIR

# Table 5: TopSim / CBM vs EmCom-Diffusion
python experiments/run_B_step2_triplet.py

# Table 6: R@1 distractor sensitivity
python experiments/run_distractor_sensitivity.py \
    --ckpt       outputs/ec_game/referential_YYYYMMDD/checkpoint_29.pth \
    --corpus     outputs/ec_corpus.json \
    --cache_dir  outputs/experiments_v2 \
    --output_dir outputs/experiments_v3
```

### Phase 7 — Bootstrap CI (Tables 4–6 Final Values)

```bash
python experiments/bootstrap_paper.py
# Output CSVs written to $PROJ_DIR/experiments_v3/outputs/
```

---

## Model Architecture

### Speaker (Referential Game)

| Component | Details |
|---|---|
| Visual encoder | Frozen DINOv2 ViT-B/14 |
| Token generation | K=8 learnable queries → cross-attention to patch tokens → linear → Gumbel-softmax (simultaneous) |
| Vocabulary | V=256 |
| Temperature annealing | τ: 1.0 → 0.5 over first 30,000 steps (linear) |

### Receiver (Referential Game)

| Component | Details |
|---|---|
| Message encoding | Soft one-hot → differentiable word embedding lookup → 6-layer BERT (hidden=768, 6 heads) |
| Projection | Linear(768 → 256) + L2 normalisation |

### Training

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 0.05 |
| Batch size | 128 (128-way game) |
| Warmup | 1,000 steps |
| Epochs | 30 |
| GPUs | 4 × NVIDIA RTX 6000 Ada |

---

## Citation

```bibtex
@inproceedings{omoto2026emcomdiffusion,
  title     = {{EmCom-Diffusion}: Probing Visual Reflection in Emergent Languages via Image Generation},
  author    = {Omoto, Haruumi and Taniguchi, Tadahiro},
  booktitle = {Proceedings of the International Conference on Neural Information Processing (ICONIP)},
  year      = {2026}
}
```

---

## License

This repository is released under the MIT License.  
The BERT implementation (`ec_game/models/med.py`) is derived from [BLIP](https://github.com/salesforce/BLIP) (BSD-3-Clause).
