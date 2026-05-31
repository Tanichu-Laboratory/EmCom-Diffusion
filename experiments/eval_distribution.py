#!/usr/bin/env python3
"""
Experiment A2: Distribution-level distance metrics.

Computes KID, Precision/Recall, Vendi Score, and Density/Coverage
for each condition (EC, Random, Fixed, NL, NL_ft) against reference images.

Uses existing generated images in outputs/generated_images/.

Usage:
    cd .
    python experiments_v2/exp_a2_distribution_metrics.py \
        --n_samples 1000 \
        --out_csv outputs/experiments_v2/a2_distribution_metrics.csv
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).parent.parent
COCO_ROOT = Path(os.environ.get("COCO_ROOT", "data/coco2017"))


# ─── Image loading ────────────────────────────────────────────────────────────

def load_pil_images(img_dir: Path, indices: list[int]) -> list[Image.Image]:
    paths = sorted(img_dir.glob("*.png"))
    imgs = []
    for i in indices:
        if i < len(paths):
            imgs.append(Image.open(paths[i]).convert("RGB"))
    return imgs


def load_ref_images(val_entries: list[dict]) -> list[Image.Image]:
    return [Image.open(e["image_path"]).convert("RGB") for e in val_entries]


# ─── Inception features (for KID / Precision-Recall) ─────────────────────────

def _to_uint8_tensor(pil_images: list[Image.Image], size: int = 299) -> torch.Tensor:
    out = []
    for img in pil_images:
        arr = np.array(img.convert("RGB").resize((size, size)))
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(out)  # [N, 3, size, size] uint8


def compute_kid(gen_images: list[Image.Image],
                ref_images: list[Image.Image],
                device: torch.device,
                batch_size: int = 32) -> dict:
    """Compute KID (Kernel Inception Distance). Lower is better."""
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        return {"kid_mean": float("nan"), "kid_std": float("nan")}

    kid = KernelInceptionDistance(feature=2048, subset_size=min(50, len(gen_images))).to(device)
    bs = batch_size

    for i in range(0, len(ref_images), bs):
        kid.update(_to_uint8_tensor(ref_images[i:i+bs]).to(device), real=True)
    for i in range(0, len(gen_images), bs):
        kid.update(_to_uint8_tensor(gen_images[i:i+bs]).to(device), real=False)

    mean, std = kid.compute()
    return {"kid_mean": float(mean.item()), "kid_std": float(std.item())}


# ─── CLIP embeddings (reused across conditions) ───────────────────────────────

def load_clip(device: torch.device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


@torch.no_grad()
def embed_images_clip(model, proc, images: list[Image.Image],
                      device: torch.device, batch_size: int = 64) -> torch.Tensor:
    embs = []
    for i in range(0, len(images), batch_size):
        inp = proc(images=images[i:i+batch_size], return_tensors="pt").to(device)
        embs.append(F.normalize(model.get_image_features(**inp), dim=-1).cpu())
    return torch.cat(embs)  # [N, 512]


# ─── Vendi Score ──────────────────────────────────────────────────────────────

def vendi_score(embeddings: torch.Tensor) -> float:
    """
    Vendi Score = exp(H(K/n)) where K is the Gram matrix and H is entropy of eigenvalues.
    Friedman & Dieng 2022.
    """
    n = embeddings.shape[0]
    # Gram matrix via cosine similarity (embeddings already normalized)
    K = (embeddings @ embeddings.T).numpy()  # [n, n]
    K = K / n
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 0]
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-10))
    return float(np.exp(entropy))


# ─── Precision / Recall / Density / Coverage ─────────────────────────────────

def _knn_distances(X: torch.Tensor, k: int) -> torch.Tensor:
    """Return the k-th nearest neighbor distance for each point in X (excluding self)."""
    dists = torch.cdist(X, X)  # [N, N]
    dists.fill_diagonal_(float("inf"))
    knn_dists, _ = dists.topk(k, dim=-1, largest=False)  # [N, k]
    return knn_dists[:, -1]  # k-th NN distance [N]


def precision_recall_density_coverage(
    real_embs: torch.Tensor,
    fake_embs: torch.Tensor,
    k: int = 3,
) -> dict:
    """
    Improved Precision/Recall (Kynkäänniemi et al., 2019) and
    Density/Coverage (Naeem et al., 2020).

    All embeddings assumed to be L2-normalized.
    """
    real_knn = _knn_distances(real_embs, k)  # [N_real]
    fake_knn = _knn_distances(fake_embs, k)  # [N_fake]

    n_real = real_embs.shape[0]
    n_fake = fake_embs.shape[0]

    # pairwise cross-distances
    cross = torch.cdist(fake_embs, real_embs)  # [N_fake, N_real]
    cross_rf = torch.cdist(real_embs, fake_embs)  # [N_real, N_fake]

    # Precision: fraction of fake samples within a real ball
    in_real_ball = (cross <= real_knn.unsqueeze(0))  # [N_fake, N_real]
    precision = float(in_real_ball.any(dim=-1).float().mean().item())

    # Recall: fraction of real samples within a fake ball
    in_fake_ball = (cross_rf <= fake_knn.unsqueeze(0))  # [N_real, N_fake]
    recall = float(in_fake_ball.any(dim=-1).float().mean().item())

    # Density: average number of fake samples within real balls
    density = float((in_real_ball.float().sum(dim=0) / (k * n_fake)).mean().item())

    # Coverage: fraction of real balls that contain at least one fake sample
    coverage = float(in_real_ball.float().sum(dim=0).bool().float().mean().item())

    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k_nn", type=int, default=3,
                        help="k for kNN in Precision/Recall/Density/Coverage")
    parser.add_argument("--ec_json",    required=True, help="EC corpus JSON")
    parser.add_argument("--gen_dir",    required=True, help="Root dir of generated images")
    parser.add_argument("--out_csv",    required=True, help="Output CSV path")
    parser.add_argument("--conditions", nargs="+",
                        default=["A", "A_random", "A_fixed", "NL", "NL_ft"],
                        help="Subdirectory names under --gen_dir")
    args = parser.parse_args()

    EC_JSON = args.ec_json
    GEN_DIR = Path(args.gen_dir)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    with open(EC_JSON) as f:
        ec = json.load(f)
    val_all = ec["val"]
    n = min(args.n_samples, len(val_all))
    indices = rng.sample(range(len(val_all)), n)
    val_entries = [val_all[i] for i in indices]

    print(f"Loading reference images (n={n}) ...")
    ref_images = load_ref_images(val_entries)

    print("Loading CLIP ...")
    clip_model, clip_proc = load_clip(device)

    print("Embedding reference images ...")
    ref_embs = embed_images_clip(clip_model, clip_proc, ref_images, device,
                                 args.batch_size)  # [N, 512]

    rows = []
    for cond in args.conditions:
        img_dir = GEN_DIR / cond
        if not img_dir.exists():
            print(f"  SKIP {cond}: directory not found at {img_dir}")
            continue

        print(f"\n[{cond}] loading {n} images ...")
        gen_images = load_pil_images(img_dir, indices)
        if len(gen_images) < n:
            print(f"  WARNING: only {len(gen_images)} images found")

        gen_embs = embed_images_clip(clip_model, clip_proc, gen_images, device,
                                     args.batch_size)  # [M, 512]

        print(f"  Computing Vendi Score ...")
        vs = vendi_score(gen_embs)

        print(f"  Computing Precision/Recall/Density/Coverage (k={args.k_nn}) ...")
        prdc = precision_recall_density_coverage(ref_embs[:len(gen_embs)], gen_embs, k=args.k_nn)

        print(f"  Computing KID ...")
        kid = compute_kid(gen_images, ref_images[:len(gen_images)], device, args.batch_size)

        row = {
            "condition": cond,
            "n": len(gen_images),
            "vendi_score": round(vs, 4),
            "precision": round(prdc["precision"], 4),
            "recall": round(prdc["recall"], 4),
            "density": round(prdc["density"], 4),
            "coverage": round(prdc["coverage"], 4),
            "kid_mean": round(kid["kid_mean"], 6),
            "kid_std": round(kid["kid_std"], 6),
        }
        rows.append(row)
        print(f"  {cond}: vendi={vs:.3f}  prec={prdc['precision']:.4f}  "
              f"rec={prdc['recall']:.4f}  dens={prdc['density']:.4f}  "
              f"cov={prdc['coverage']:.4f}  kid={kid['kid_mean']:.4f}±{kid['kid_std']:.4f}")

    fieldnames = ["condition", "n", "vendi_score",
                  "precision", "recall", "density", "coverage",
                  "kid_mean", "kid_std"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
