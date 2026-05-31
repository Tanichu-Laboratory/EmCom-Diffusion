# EmCom-Diffusion: Probing Visual Reflection in Emergent Languages via Image Generation

Haruumi Omoto and Tadahiro Taniguchi  
Graduate School of Informatics, Kyoto University

> **Visual reflection** is the extent to which emergent messages preserve information about their source images that can be recovered without appeal to the speaker–listener pair that produced them. EmCom-Diffusion measures this property by fine-tuning a text-to-image diffusion model on (image, emergent-message) pairs and quantifying the perceptual similarity between the generated and original image.

---

## Quick Start: Evaluate Your Own EC Corpus

If you already have an EC corpus from any Referential Game setup, you can run EmCom-Diffusion directly without retraining the game model.

**Corpus format** — a JSON file with the following structure:

```json
{
  "vocab_size": 256,
  "num_tokens": 8,
  "pad_token_id": 256,
  "train": [
    {"image_id": 391895, "image_path": "/path/to/train2017/000000391895.jpg", "ec_tokens": [13, 42, 7, 0, 255, 3, 99, 18]},
    ...
  ],
  "val": [
    {"image_id": 139, "image_path": "/path/to/val2017/000000000139.jpg", "ec_tokens": [26, 1, 88, ...]},
    ...
  ]
}
```

**Step 1 — Fine-tune Stable Diffusion v1.5 on your corpus:**

```bash
export COCO_ROOT=/path/to/coco2017

accelerate launch -m diffusion.train \
    --config configs/setting_b.yaml   # EC embedding + U-Net LoRA
```

Edit `configs/setting_b.yaml` to set `ec_json` to your corpus path.

**Step 2 — Generate images:**

```bash
python -m diffusion.inference \
    --ckpts    checkpoints/your_model/step_010000 \
    --labels   EL \
    --n_samples 5000 \
    --input_type ec \
    --ec_json  your_corpus.json \
    --out_img_dir outputs/generated_images
```

Also generate baselines for validation:

```bash
# Random-token baseline
python -m diffusion.inference ... --input_type random --labels Random

# Fixed-token baseline
python -m diffusion.inference ... --input_type fixed  --labels Fixed
```

**Step 3 — Compute EmCom-Diffusion scores:**

```bash
# CLIP-img (ViT-B/32), DINOv2 ViT-B/14, SigLIP ViT-B/16
python experiments/exp_b1_multi_encoder.py \
    --ec_json  your_corpus.json \
    --gen_dir  outputs/generated_images \
    --out_csv  outputs/emcomdiffusion_scores.csv \
    --n_samples 5000

# CLIP-text and FID
python experiments/evaluate.py \
    --ec_json    your_corpus.json \
    --gen_dirs   outputs/generated_images/EL \
    --rand_dirs  outputs/generated_images/Random \
    --fixed_dirs outputs/generated_images/Fixed \
    --labels EL --n_samples 5000 \
    --out_csv outputs/eval_metrics.csv
```

The expected ordering is **Random/Fixed < EL** across all similarity metrics if the emergent language encodes visual content.

---

## Requirements

```bash
pip install -r requirements.txt
```

PyTorch must be installed separately following [pytorch.org](https://pytorch.org).  
DINOv2 is loaded at runtime via `torch.hub` from `facebookresearch/dinov2`.

---

## Repository Structure

```
emcom-diffusion/
├── ec_game/                    # Referential Game training
│   ├── models/
│   │   ├── emcom.py            # Speaker (cross-attn over DINOv2 patches) + Receiver (BERT)
│   │   ├── vit.py              # DINOv2 wrapper
│   │   ├── med.py              # BERT implementation
│   │   └── blip_utils.py       # create_vit helper
│   ├── configs/
│   │   ├── config_referential.yaml
│   │   └── bert_config.json
│   ├── train.py                # Training script
│   └── extract_corpus.py       # Extract EC corpus from checkpoint
│
├── diffusion/                  # SD fine-tuning & image generation
│   ├── train.py                # Fine-tune SD v1.5 with EC tokens (Settings A/B/C)
│   ├── train_nl.py             # Fine-tune SD v1.5 with NL captions (SD-NL baseline)
│   ├── inference.py            # Generate images (EC / Random / Fixed / NL)
│   ├── ec_text_encoder.py      # Custom CLIP text encoder for EC tokens
│   ├── data.py                 # Dataset classes
│   └── utils.py
│
├── experiments/                # Evaluation
│   ├── exp_b1_multi_encoder.py # EmCom-Diffusion score: CLIP-img, DINOv2, SigLIP
│   ├── evaluate.py             # CLIP-text, FID
│   ├── exp_a1_generate_multiseed.py  # Multi-seed generation (Table 3)
│   ├── exp_a1_eval_diversity.py      # Multi-seed diversity analysis (Table 3)
│   ├── exp_a2_distribution_metrics.py # Vendi, Recall
│   ├── cache_dino_gen.py       # Cache DINOv2 embeddings of generated images
│   ├── run_A_triplet.py        # Table 4: Translation vs EmCom-Diffusion
│   ├── run_B_step2_triplet.py  # Table 5: TopSim / CBM vs EmCom-Diffusion
│   ├── run_distractor_sensitivity.py # Table 6: R@1 distractor sensitivity
│   ├── eval_r1_val5k.py        # Corpus-level R@1
│   └── bootstrap_paper.py      # Bootstrap 95% CI for Tables 4–6
│
└── translation/                # EC→NL translation baseline (Yao et al. 2022)
    ├── vocab.py / dataset.py / model.py
    ├── train.py
    └── translate.py
```

---

## Referential Game

This section is only needed if you want to train your own EC model.  
If you already have an EC corpus, skip to [Validate EmCom-Diffusion](#validate-emcom-diffusion).

**Prepare an image-list JSON** — a list of `{"image": "/path/to/img.jpg"}` entries for all training images:

```bash
# Train (4-GPU)
export COCO_ROOT=/path/to/coco2017

torchrun --nproc_per_node=4 -m ec_game.train \
    --config   ec_game/configs/config_referential.yaml \
    --img_list data/coco_image_list.json \
    --output_dir outputs/ec_game

# Extract EC corpus from the trained checkpoint (e.g. epoch 29)
python -m ec_game.extract_corpus \
    --ckpt    outputs/ec_game/referential_YYYYMMDD/checkpoint_29.pth \
    --ec_json data/ec_captions.json \
    --output  outputs/ec_corpus.json
```

`ec_captions.json` provides the train/val image paths (same format as the corpus output, without `ec_tokens`).

**Model architecture:**

| | Speaker | Receiver |
|---|---|---|
| Visual encoder | Frozen DINOv2 ViT-B/14 | Same frozen DINOv2 |
| Core module | K=8 learnable queries → cross-attention to patch tokens | 6-layer BERT (hidden=768, 6 heads) |
| Output | Gumbel-softmax → K tokens from V=256 vocab | L2-normalised 256-dim projection |

Training: AdamW, lr=3×10⁻⁴, weight decay=0.05, batch=128 (128-way game), warmup=1,000 steps, 30 epochs, τ: 1.0→0.5 (linear over 30k steps).

---

## Validate EmCom-Diffusion

Reproduces Table 2 (validity) and Table 3 (multi-seed analysis).

### Fine-tune Stable Diffusion

```bash
# Setting B: EC embedding + U-Net LoRA  (used for main results)
accelerate launch -m diffusion.train --config configs/setting_b.yaml

# SD-NL baseline: NL captions + U-Net LoRA
accelerate launch diffusion/train_nl.py --config configs/setting_nl.yaml
```

### Generate images (5,000 val images × 4 conditions)

```bash
for TYPE in ec random fixed; do
  python -m diffusion.inference \
    --ckpts  checkpoints/pilot_B/step_010000 \
    --labels B --n_samples 5000 \
    --input_type $TYPE \
    --ec_json outputs/ec_corpus.json \
    --out_img_dir outputs/generated_images
done

# SD-NL baseline
python -m diffusion.inference \
    --ckpts  checkpoints/pilot_B/step_010000 \
    --labels B --n_samples 5000 --nl_only \
    --nl_ft_ckpt checkpoints/pilot_NL/step_010000 \
    --ec_json outputs/ec_corpus.json \
    --out_img_dir outputs/generated_images
```

Generation uses DDIM (50 steps, guidance=7.5), evaluated on a fixed 5,000-image subset with seed 42.

### Compute metrics (Table 2)

```bash
# CLIP-img (ViT-B/32), DINOv2, SigLIP
python experiments/exp_b1_multi_encoder.py \
    --ec_json  outputs/ec_corpus.json \
    --gen_dir  outputs/generated_images \
    --out_csv  outputs/b1_multi_encoder.csv \
    --n_samples 5000

# CLIP-text, FID
python experiments/evaluate.py \
    --ec_json    outputs/ec_corpus.json \
    --gen_dirs   outputs/generated_images/B \
    --rand_dirs  outputs/generated_images/B_random \
    --fixed_dirs outputs/generated_images/B_fixed \
    --nl_ft_dir  outputs/generated_images/NL_ft \
    --labels B --n_samples 5000 \
    --out_csv outputs/pilot_metrics.csv

# Vendi, Recall
python experiments/exp_a2_distribution_metrics.py \
    --ec_json  outputs/ec_corpus.json \
    --gen_dir  outputs/generated_images \
    --out_csv  outputs/distribution_metrics.csv
```

### Multi-seed analysis (Table 3)

```bash
python experiments/exp_a1_generate_multiseed.py \
    --ec_json   outputs/ec_corpus.json \
    --ec_ckpt   checkpoints/pilot_B/step_010000 \
    --out_dir   outputs/multiseed \
    --n_samples 500 --n_seeds 8

python experiments/exp_a1_eval_diversity.py \
    --multiseed_dir outputs/multiseed \
    --out_csv       outputs/a1_diversity.csv
```

---

## Compare with Other Methods

Reproduces Tables 4–6 (comparison against Translation, TopSim, CBM, R@1).

### Translation baseline (Table 4)

```bash
# Train EC→NL translator
python -m translation.train \
    --ec_corpus outputs/ec_corpus.json \
    --vocab     outputs/translation/vocab.json \
    --ckpt_dir  outputs/translation/checkpoints \
    --epochs 10

# Translate val set and cache CLIP ViT-L/14 text embeddings
python -m translation.translate \
    --ec_corpus  outputs/ec_corpus.json \
    --ckpt       outputs/translation/checkpoints/best.pt \
    --vocab      outputs/translation/vocab.json \
    --output     outputs/translation/translations_val5k.json \
    --clip_cache outputs/cache_pred_clip_text_vitl14.npy
```

### Cache embeddings for comparison experiments

```bash
# DINOv2 embeddings of generated images (EmCom-Diffusion score)
python experiments/cache_dino_gen.py \
    --gen_dir outputs/generated_images/B \
    --ec_json outputs/ec_corpus.json \
    --output  outputs/cache_dino_vitb14_gen.npy
```

The following GT caches must also exist in `outputs/experiments_v2/`  
(generated by earlier evaluation steps):
- `cache_gt_clip_vitl14.npy` — GT image embeddings (CLIP ViT-L/14)
- `cache_dino_vitb14_gt.npy` — GT image embeddings (DINOv2 ViT-B/14)
- `cache_gt_clip_captions_vitl14.npy` — GT caption embeddings (CLIP ViT-L/14)

### Run comparison experiments

```bash
export PROJ_DIR=/path/to/emcom-diffusion

# Table 4: Translation vs EmCom-Diffusion
python experiments/run_A_triplet.py

# Table 5: TopSim / CBM vs EmCom-Diffusion
python experiments/run_B_step2_triplet.py

# Table 6: R@1 distractor sensitivity
python experiments/run_distractor_sensitivity.py \
    --ckpt       outputs/ec_game/referential_YYYYMMDD/checkpoint_29.pth \
    --corpus     outputs/ec_corpus.json \
    --cache_dir  outputs/experiments_v2 \
    --output_dir outputs/experiments_v3
```

### Bootstrap 95% CI (final values for Tables 4–6)

```bash
python experiments/bootstrap_paper.py
# Results written to $PROJ_DIR/experiments_v3/outputs/
```

---

## License

This repository is released under the MIT License.  
The BERT implementation (`ec_game/models/med.py`) is derived from [BLIP](https://github.com/salesforce/BLIP) (BSD-3-Clause).
