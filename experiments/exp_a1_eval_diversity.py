#!/usr/bin/env python3
"""
Experiment A1 (Step 2): Evaluate multi-seed image diversity.

Reads the output of exp_a1_generate_multiseed.py and computes:
  - Intra-sample diversity:  mean pairwise CLIP/DINO distance among N images of the same token
  - Inter-sample diversity:  mean CLIP/DINO distance between images of different tokens
  - Distance to GT:          cosine distance from the N-image centroid to the reference image

Conditions evaluated: ec, random, fixed, nl

Usage:
    cd .
    python experiments_v2/exp_a1_eval_diversity.py \
        --multiseed_dir outputs/experiments_v2/multiseed \
        --out_csv outputs/experiments_v2/a1_diversity.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).parent.parent


# ─── Encoder loading ──────────────────────────────────────────────────────────

def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


def load_dino(device):
    from torchvision import transforms
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14",
        pretrained=True, verbose=False,
    ).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


# ─── Embedding helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def embed_clip(model, proc, images: list[Image.Image],
               device, batch_size: int = 64) -> torch.Tensor:
    embs = []
    for i in range(0, len(images), batch_size):
        inp = proc(images=images[i:i+batch_size], return_tensors="pt").to(device)
        embs.append(F.normalize(model.get_image_features(**inp), dim=-1).cpu())
    return torch.cat(embs)  # [N, 512]


@torch.no_grad()
def embed_dino(model, preprocess, images: list[Image.Image],
               device, batch_size: int = 64) -> torch.Tensor:
    embs = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(img) for img in images[i:i+batch_size]]).to(device)
        cls = model(batch)
        embs.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(embs)  # [N, 768]


# ─── Diversity metrics ────────────────────────────────────────────────────────

def intra_diversity(embs: torch.Tensor) -> float:
    """Mean pairwise cosine distance among N embeddings (1 - cos_sim)."""
    N = embs.shape[0]
    if N < 2:
        return 0.0
    sims = embs @ embs.T  # [N, N]
    mask = ~torch.eye(N, dtype=torch.bool)
    dists = 1.0 - sims[mask]
    return float(dists.mean().item())


def inter_diversity(all_embs: list[torch.Tensor]) -> float:
    """
    Mean cosine distance between centroids of different samples.
    all_embs: list of [N_seeds, D] tensors, one per sample.
    """
    centroids = torch.stack([e.mean(0) for e in all_embs])
    centroids = F.normalize(centroids, dim=-1)  # [S, D]
    sims = centroids @ centroids.T  # [S, S]
    S = len(all_embs)
    mask = ~torch.eye(S, dtype=torch.bool)
    dists = 1.0 - sims[mask]
    return float(dists.mean().item())


def dist_to_gt(sample_embs: torch.Tensor, gt_emb: torch.Tensor) -> float:
    """Cosine distance from the centroid of sample_embs to gt_emb."""
    centroid = F.normalize(sample_embs.mean(0, keepdim=True), dim=-1)  # [1, D]
    gt = F.normalize(gt_emb.unsqueeze(0), dim=-1)                      # [1, D]
    sim = float((centroid * gt).sum().item())
    return 1.0 - sim


# ─── Per-sample image loading ─────────────────────────────────────────────────

def load_sample_images(cond_dir: Path, sample_idx: int, n_seeds: int) -> list[Image.Image]:
    imgs = []
    for seed in range(n_seeds):
        p = cond_dir / f"{sample_idx:04d}_seed{seed:02d}.png"
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
    return imgs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed_dir", required=True,
                        help="Output directory of exp_a1_generate_multiseed.py")
    parser.add_argument("--conditions", nargs="+",
                        default=["ec", "random", "fixed", "nl"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_csv",    required=True, help="Output CSV path")
    args = parser.parse_args()

    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    multiseed_dir = Path(args.multiseed_dir)
    with open(multiseed_dir / "metadata.json") as f:
        meta = json.load(f)
    samples = meta["samples"]
    n_seeds = meta["n_seeds"]
    print(f"Loaded metadata: {len(samples)} samples, {n_seeds} seeds each")

    print("Loading CLIP ...")
    clip_model, clip_proc = load_clip(device)
    print("Loading DINOv2 ...")
    dino_model, dino_proc = load_dino(device)

    rows = []
    per_sample_rows = []  # for C2 correlation

    for cond in args.conditions:
        cond_dir = multiseed_dir / cond
        if not cond_dir.exists():
            print(f"  SKIP {cond}: not found at {cond_dir}")
            continue

        print(f"\n=== Condition: {cond} ===")

        # Collect per-sample embeddings
        all_clip_embs = []   # list of [n_seeds, 512]
        all_dino_embs = []   # list of [n_seeds, 768]
        gt_clip_embs = []    # [S, 512] reference image
        gt_dino_embs = []    # [S, 768]
        valid_indices = []

        # Batch all images for efficient embedding
        all_gen_images_flat = []   # [S*n_seeds]
        all_gt_images = []         # [S]
        sample_valid = []

        for sm in tqdm(samples, desc=f"{cond} loading"):
            idx = sm["sample_idx"]
            gen_imgs = load_sample_images(cond_dir, idx, n_seeds)
            if len(gen_imgs) == 0:
                continue
            gt_img = Image.open(sm["gt_path"]).convert("RGB")
            all_gen_images_flat.extend(gen_imgs)
            all_gt_images.append(gt_img)
            sample_valid.append((idx, len(gen_imgs)))

        if len(sample_valid) == 0:
            print(f"  No images found for {cond}")
            continue

        print(f"  Embedding {len(all_gen_images_flat)} generated images ...")
        gen_clip_flat = embed_clip(clip_model, clip_proc, all_gen_images_flat,
                                   device, args.batch_size)  # [S*n_seeds, 512]
        gen_dino_flat = embed_dino(dino_model, dino_proc, all_gen_images_flat,
                                   device, args.batch_size)  # [S*n_seeds, 768]

        print(f"  Embedding {len(all_gt_images)} reference images ...")
        gt_clip_all = embed_clip(clip_model, clip_proc, all_gt_images,
                                 device, args.batch_size)    # [S, 512]
        gt_dino_all = embed_dino(dino_model, dino_proc, all_gt_images,
                                 device, args.batch_size)    # [S, 768]

        # Reshape flat embeddings back to per-sample
        offset = 0
        for si, (s_idx, s_count) in enumerate(sample_valid):
            clip_s = gen_clip_flat[offset:offset+s_count]  # [s_count, 512]
            dino_s = gen_dino_flat[offset:offset+s_count]  # [s_count, 768]
            offset += s_count
            all_clip_embs.append(clip_s)
            all_dino_embs.append(dino_s)
            gt_clip_embs.append(gt_clip_all[si])
            gt_dino_embs.append(gt_dino_all[si])
            valid_indices.append(s_idx)

        # ── Aggregate metrics ─────────────────────────────────────────────────
        intra_clip_list = [intra_diversity(e) for e in all_clip_embs]
        intra_dino_list = [intra_diversity(e) for e in all_dino_embs]
        dist_gt_clip_list = [dist_to_gt(all_clip_embs[i], gt_clip_embs[i])
                             for i in range(len(all_clip_embs))]
        dist_gt_dino_list = [dist_to_gt(all_dino_embs[i], gt_dino_embs[i])
                             for i in range(len(all_dino_embs))]

        inter_clip = inter_diversity(all_clip_embs)
        inter_dino = inter_diversity(all_dino_embs)

        row = {
            "condition": cond,
            "n_samples": len(valid_indices),
            "n_seeds": n_seeds,
            "intra_clip_mean": round(float(np.mean(intra_clip_list)), 4),
            "intra_clip_std":  round(float(np.std(intra_clip_list)), 4),
            "intra_dino_mean": round(float(np.mean(intra_dino_list)), 4),
            "intra_dino_std":  round(float(np.std(intra_dino_list)), 4),
            "inter_clip":      round(inter_clip, 4),
            "inter_dino":      round(inter_dino, 4),
            "dist_gt_clip_mean": round(float(np.mean(dist_gt_clip_list)), 4),
            "dist_gt_clip_std":  round(float(np.std(dist_gt_clip_list)), 4),
            "dist_gt_dino_mean": round(float(np.mean(dist_gt_dino_list)), 4),
            "dist_gt_dino_std":  round(float(np.std(dist_gt_dino_list)), 4),
        }
        rows.append(row)
        print(f"  intra_clip={row['intra_clip_mean']:.4f}  "
              f"inter_clip={row['inter_clip']:.4f}  "
              f"dist_gt_clip={row['dist_gt_clip_mean']:.4f}  "
              f"intra_dino={row['intra_dino_mean']:.4f}  "
              f"dist_gt_dino={row['dist_gt_dino_mean']:.4f}")

        # Per-sample rows for C2 correlation input
        for i, s_idx in enumerate(valid_indices):
            # image-generation similarity = 1 - dist_to_gt (cosine similarity)
            per_sample_rows.append({
                "sample_idx": s_idx,
                "condition": cond,
                "clip_sim_gt": round(1.0 - dist_gt_clip_list[i], 4),
                "dino_sim_gt": round(1.0 - dist_gt_dino_list[i], 4),
                "intra_clip":  round(intra_clip_list[i], 4),
                "intra_dino":  round(intra_dino_list[i], 4),
            })

    # ── Write aggregate CSV ───────────────────────────────────────────────────
    fieldnames = ["condition", "n_samples", "n_seeds",
                  "intra_clip_mean", "intra_clip_std",
                  "intra_dino_mean", "intra_dino_std",
                  "inter_clip", "inter_dino",
                  "dist_gt_clip_mean", "dist_gt_clip_std",
                  "dist_gt_dino_mean", "dist_gt_dino_std"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved aggregate: {args.out_csv}")

    # ── Write per-sample CSV (for C2) ─────────────────────────────────────────
    per_sample_csv = str(out_dir / "a1_per_sample_scores.csv")
    per_sample_fields = ["sample_idx", "condition",
                         "clip_sim_gt", "dino_sim_gt",
                         "intra_clip", "intra_dino"]
    with open(per_sample_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_sample_fields)
        w.writeheader()
        w.writerows(per_sample_rows)
    print(f"Saved per-sample scores: {per_sample_csv}")


if __name__ == "__main__":
    main()
