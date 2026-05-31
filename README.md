# EmCom-Diffusion: Probing Visual Reflection in Emergent Languages via Image Generation

Haruumi Omoto and Tadahiro Taniguchi  
Graduate School of Informatics, Kyoto University

> **Visual reflection** is the extent to which emergent messages preserve information about their source images that can be recovered without appeal to the speaker–listener pair that produced them. EmCom-Diffusion measures this property by fine-tuning a text-to-image diffusion model on (image, emergent-message) pairs and quantifying the perceptual similarity between the generated and original image.

---

## Quick Start: Evaluate Your Own EC Corpus

If you already have an EC corpus from any Referential Game setup, you can apply EmCom-Diffusion without retraining the game model.

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

accelerate launch -m diffusion.train --config configs/diffusion_ec_lora.yaml
```

Edit `configs/diffusion_ec_lora.yaml` to point `ec_json` to your corpus file.

**Step 2 — Generate images for each condition:**

```bash
EC_JSON=your_corpus.json
CKPT=checkpoints/emcom_model/step_010000

# Emergent-language condition
python -m diffusion.inference \
    --ckpts $CKPT --labels emcom \
    --n_samples 5000 --input_type ec \
    --ec_json $EC_JSON --out_img_dir outputs/generated_images

# Random-token baseline (no image-specific information)
python -m diffusion.inference \
    --ckpts $CKPT --labels emcom_random \
    --n_samples 5000 --input_type random \
    --ec_json $EC_JSON --out_img_dir outputs/generated_images

# Fixed-token baseline (same sequence for every image)
python -m diffusion.inference \
    --ckpts $CKPT --labels emcom_fixed \
    --n_samples 5000 --input_type fixed \
    --ec_json $EC_JSON --out_img_dir outputs/generated_images
```

**Step 3 — Compute EmCom-Diffusion scores:**

```bash
# Per-image similarity: CLIP-img (ViT-B/32), DINOv2 ViT-B/14, SigLIP ViT-B/16
python experiments/eval_similarity.py \
    --ec_json your_corpus.json \
    --gen_dir outputs/generated_images \
    --conditions emcom emcom_random emcom_fixed \
    --out_csv outputs/emcomdiffusion_scores.csv \
    --n_samples 5000

# CLIP-text similarity and FID
python experiments/evaluate.py \
    --ec_json    your_corpus.json \
    --gen_dirs   outputs/generated_images/emcom \
    --rand_dirs  outputs/generated_images/emcom_random \
    --fixed_dirs outputs/generated_images/emcom_fixed \
    --labels emcom --n_samples 5000 \
    --out_csv outputs/eval_metrics.csv
```

The expected ordering is **emcom_random / emcom_fixed < emcom** across all similarity metrics if the emergent language encodes visual content.

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
├── ec_game/                        # Referential Game
│   ├── models/
│   │   ├── emcom.py                # Speaker (cross-attn over DINOv2 patches) + Receiver (BERT)
│   │   ├── vit.py                  # DINOv2 wrapper
│   │   ├── med.py                  # BERT implementation
│   │   └── blip_utils.py           # create_vit helper
│   ├── configs/
│   │   ├── config_referential.yaml
│   │   └── bert_config.json
│   ├── train.py                    # Training script
│   └── extract_corpus.py           # Extract EC corpus from checkpoint
│
├── diffusion/                      # SD fine-tuning & image generation
│   ├── train.py                    # Fine-tune SD v1.5 with EC tokens
│   ├── train_nl.py                 # Fine-tune SD v1.5 with NL captions (SD-NL baseline)
│   ├── inference.py                # Generate images (EC / Random / Fixed / NL)
│   ├── ec_text_encoder.py          # Custom CLIP text encoder for EC tokens
│   ├── data.py
│   └── utils.py
│
├── experiments/                    # Evaluation scripts
│   ├── eval_similarity.py          # EmCom-Diffusion score: CLIP-img, DINOv2, SigLIP
│   ├── evaluate.py                 # CLIP-text similarity, FID
│   ├── generate_multiseed.py       # Generate N seeds per token sequence (Table 3)
│   ├── eval_multiseed_diversity.py # Intra/inter-token distance, distance to GT (Table 3)
│   ├── eval_distribution.py        # Vendi score, Recall
│   ├── cache_embeddings.py         # Cache DINOv2 embeddings of generated images
│   ├── compare_translation.py      # Table 4: Translation vs EmCom-Diffusion
│   ├── compare_topsim_cbm.py       # Table 5: TopSim / CBM vs EmCom-Diffusion
│   ├── run_distractor_sensitivity.py # Table 6: R@1 distractor sensitivity
│   ├── eval_r1.py                  # Corpus-level R@1
│   └── compute_ci.py               # Bootstrap 95% CI for Tables 4–6
│
└── translation/                    # EC→NL translation baseline (Yao et al. 2022)
    ├── vocab.py / dataset.py / model.py
    ├── train.py
    └── translate.py
```

---

## Referential Game

This section is only needed if you want to train your own EC model from scratch.  
If you already have an EC corpus, skip directly to [Validate EmCom-Diffusion](#validate-emcom-diffusion).

**Prepare an image-list JSON** — a list of `{"image": "/path/to/img.jpg"}` entries for training images.

```bash
export COCO_ROOT=/path/to/coco2017

# Train (4-GPU)
torchrun --nproc_per_node=4 -m ec_game.train \
    --config   ec_game/configs/config_referential.yaml \
    --img_list data/coco_image_list.json \
    --output_dir outputs/ec_game

# Extract EC corpus from the final checkpoint
python -m ec_game.extract_corpus \
    --ckpt    outputs/ec_game/referential_YYYYMMDD/checkpoint_29.pth \
    --ec_json data/ec_captions.json \
    --output  outputs/ec_corpus.json
```

`ec_captions.json` maps each image to its train/val split and file path (same format as the corpus output, without the `ec_tokens` field).

**Model architecture:**

| | Speaker | Receiver |
|---|---|---|
| Visual encoder | Frozen DINOv2 ViT-B/14 | Same frozen DINOv2 |
| Core module | K=8 learnable queries → cross-attention to DINOv2 patch tokens | 6-layer BERT (hidden=768, 6 heads) |
| Output | Straight-through Gumbel-softmax → K tokens, V=256 vocab | L2-normalised 256-dim projection |

Training: AdamW, lr=3×10⁻⁴, weight decay=0.05, batch=128 (128-way game), warmup=1,000 steps, 30 epochs, temperature τ: 1.0→0.5 linear over 30k steps.

---

## Validate EmCom-Diffusion

Reproduces Table 2 (validity) and Table 3 (multi-seed analysis).

### Fine-tune Stable Diffusion

SD v1.5 is fine-tuned in three settings depending on how many parameters are updated:

| Config | What is trained | Notes |
|---|---|---|
| `diffusion_ec_embed.yaml` | EC token embedding only | U-Net frozen |
| `diffusion_ec_lora.yaml` | EC token embedding + U-Net LoRA | **Recommended** |
| `diffusion_ec_full.yaml` | EC token embedding + CLIP top-4 layers + U-Net LoRA | Heaviest |

```bash
# EC-conditioned model
accelerate launch -m diffusion.train --config configs/diffusion_ec_lora.yaml

# SD-NL baseline: same budget, NL captions instead of EC tokens
accelerate launch diffusion/train_nl.py --config configs/diffusion_nl_lora.yaml
```

### Generate images (5,000 val images × conditions)

```bash
EC_JSON=outputs/ec_corpus.json
EC_CKPT=checkpoints/emcom_model/step_010000
NL_CKPT=checkpoints/sd_nl_model/step_010000

for TYPE in ec random fixed; do
  python -m diffusion.inference \
    --ckpts $EC_CKPT --labels emcom_${TYPE} \
    --n_samples 5000 --input_type $TYPE \
    --ec_json $EC_JSON --out_img_dir outputs/generated_images
done

# SD-NL baseline
python -m diffusion.inference \
    --ckpts $EC_CKPT --labels sd_nl \
    --n_samples 5000 --nl_only \
    --nl_ft_ckpt $NL_CKPT \
    --ec_json $EC_JSON --out_img_dir outputs/generated_images
```

Generation: DDIM 50 steps, classifier-free guidance 7.5, seed 42.

### Compute metrics (Table 2)

```bash
EC_JSON=outputs/ec_corpus.json
GEN=outputs/generated_images

# CLIP-img (ViT-B/32), DINOv2, SigLIP
python experiments/eval_similarity.py \
    --ec_json $EC_JSON \
    --gen_dir $GEN \
    --conditions emcom_ec emcom_random emcom_fixed sd_nl \
    --out_csv outputs/similarity_scores.csv \
    --n_samples 5000

# CLIP-text, FID
python experiments/evaluate.py \
    --ec_json    $EC_JSON \
    --gen_dirs   $GEN/emcom_ec \
    --rand_dirs  $GEN/emcom_random \
    --fixed_dirs $GEN/emcom_fixed \
    --nl_ft_dir  $GEN/sd_nl \
    --labels emcom --n_samples 5000 \
    --out_csv outputs/clip_fid_scores.csv

# Vendi score, Recall
python experiments/eval_distribution.py \
    --ec_json  $EC_JSON \
    --gen_dir  $GEN \
    --out_csv  outputs/distribution_scores.csv
```

### Multi-seed analysis (Table 3)

Verifies that EmCom-Diffusion measures input-specific signal rather than sampling noise.

```bash
# Generate 8 images per token sequence for 500 samples
python experiments/generate_multiseed.py \
    --ec_json   outputs/ec_corpus.json \
    --ec_ckpt   checkpoints/emcom_model/step_010000 \
    --out_dir   outputs/multiseed \
    --n_samples 500 --n_seeds 8

# Compute intra-token distance, inter-token distance, distance to GT
python experiments/eval_multiseed_diversity.py \
    --multiseed_dir outputs/multiseed \
    --out_csv       outputs/multiseed_diversity.csv
```

---

## Compare with Other Methods

Reproduces Tables 4–6 (comparison against Translation, TopSim, CBM, R@1).

All scripts in this section read intermediate embedding caches generated by the steps above.  
Set `PROJ_DIR` to the repository root before running:

```bash
export PROJ_DIR=/path/to/emcom-diffusion
```

### Translation baseline (Table 4)

```bash
# Train EC→NL seq2seq translator (Transformer, 3 enc / 6 dec layers, d=256)
python -m translation.train \
    --ec_corpus outputs/ec_corpus.json \
    --vocab     outputs/translation/vocab.json \
    --ckpt_dir  outputs/translation/checkpoints \
    --epochs 10

# Translate val set; cache CLIP ViT-L/14 text embeddings of translations
python -m translation.translate \
    --ec_corpus  outputs/ec_corpus.json \
    --ckpt       outputs/translation/checkpoints/best.pt \
    --vocab      outputs/translation/vocab.json \
    --output     outputs/translation/translations_val5k.json \
    --clip_cache outputs/clip_translation_embeddings.npy
```

### Cache embeddings for triplet experiments

```bash
# DINOv2 ViT-B/14 embeddings of EC-generated images (EmCom-Diffusion score)
python experiments/cache_embeddings.py \
    --gen_dir outputs/generated_images/emcom_ec \
    --ec_json outputs/ec_corpus.json \
    --output  outputs/emcom_dino_embeddings.npy
```

The following GT embedding caches are also required (generated automatically by `eval_similarity.py` and `evaluate.py` above):

| Cache file | Content | Encoder |
|---|---|---|
| `outputs/gt_clip_embeddings.npy` | GT val images | CLIP ViT-L/14 |
| `outputs/gt_dino_embeddings.npy` | GT val images | DINOv2 ViT-B/14 |
| `outputs/gt_caption_clip_embeddings.npy` | GT captions (5-caption average) | CLIP ViT-L/14 |

### Run comparison experiments

```bash
# Table 4: Translation vs EmCom-Diffusion (τ=0.7, varying caption-similarity ε)
python experiments/compare_translation.py

# Table 5: TopSim / CBM vs EmCom-Diffusion (matched edit distance)
python experiments/compare_topsim_cbm.py

# Table 6: R@1 under random vs hard-negative distractors
python experiments/run_distractor_sensitivity.py \
    --ckpt       outputs/ec_game/referential_YYYYMMDD/checkpoint_29.pth \
    --corpus     outputs/ec_corpus.json \
    --cache_dir  outputs \
    --output_dir outputs/distractor_results
```

### Bootstrap 95% CI (Tables 4–6 final values)

```bash
python experiments/compute_ci.py
# CSV results written to $PROJ_DIR/experiments_v3/outputs/
```

---

## License

This repository is released under the MIT License.  
The BERT implementation (`ec_game/models/med.py`) is derived from [BLIP](https://github.com/salesforce/BLIP) (BSD-3-Clause).
