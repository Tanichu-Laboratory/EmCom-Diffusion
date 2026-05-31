#!/usr/bin/env python3
"""
Experiment B1: Multi-encoder consistency.

Evaluates EC vs baselines across multiple visual encoders to show that the
improvement is not an artifact of CLIP-specific biases.

Encoders:
  - CLIP ViT-B/32       (openai/clip-vit-base-patch32)      — language-supervised
  - DINOv2 ViT-B/14     (facebookresearch/dinov2)            — self-supervised
  - SigLIP ViT-B/16     (open_clip: ViT-B-16-SigLIP, pretrained="webli") — language-supervised, diff. dataset
  - MAE ViT-B/16        (facebook/vit-mae-base)              — self-supervised, vision-only

Metric: cosine similarity between generated image embedding and reference image embedding.

Usage:
    cd .
    python experiments_v2/exp_b1_multi_encoder.py \
        --n_samples 1000 \
        --conditions A A_random A_fixed NL NL_ft
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / '..' / 'diffusion'))


# ─── Encoder factories ────────────────────────────────────────────────────────

def make_clip_encoder(device: torch.device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def embed(images: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            inp = proc(images=images[i:i+batch_size], return_tensors="pt").to(device)
            embs.append(F.normalize(model.get_image_features(**inp), dim=-1).cpu())
        return torch.cat(embs)

    return embed


def make_dino_encoder(device: torch.device):
    from torchvision import transforms
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14",
        pretrained=True, verbose=False,
    ).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def embed(images: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(img) for img in images[i:i+batch_size]]).to(device)
            embs.append(F.normalize(model(batch), dim=-1).cpu())
        return torch.cat(embs)

    return embed


def make_siglip_encoder(device: torch.device):
    """SigLIP ViT-B/16 — via open_clip (transformers <4.39 lacks SigLIP support)."""
    try:
        import open_clip
        from torchvision import transforms

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli"
        )
        model = model.to(device).eval()

        @torch.no_grad()
        def embed(images: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
            embs = []
            for i in range(0, len(images), batch_size):
                batch = torch.stack([preprocess(img) for img in images[i:i+batch_size]]).to(device)
                feats = model.encode_image(batch)
                embs.append(F.normalize(feats, dim=-1).cpu())
            return torch.cat(embs)

        return embed
    except Exception as e:
        print(f"  WARNING: SigLIP unavailable ({e}); skipping")
        return None


def make_mae_encoder(device: torch.device):
    """MAE ViT-B/16 — masked autoencoder, purely vision self-supervised."""
    try:
        from transformers import AutoFeatureExtractor, ViTMAEModel
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device).eval()
        proc = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")

        @torch.no_grad()
        def embed(images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
            embs = []
            for i in range(0, len(images), batch_size):
                inp = proc(images=images[i:i+batch_size], return_tensors="pt").to(device)
                # MAE outputs sequence; use CLS token (index 0)
                out = model(**inp)
                cls = out.last_hidden_state[:, 0, :]  # [B, 768]
                embs.append(F.normalize(cls, dim=-1).cpu())
            return torch.cat(embs)

        return embed
    except Exception as e:
        print(f"  WARNING: MAE unavailable ({e}); skipping")
        return None


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_mean_ci(scores: np.ndarray, n: int = 1000, ci: float = 0.95,
                      seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(scores, size=len(scores), replace=True).mean()
             for _ in range(n)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(scores.mean()), float(lo), float(hi)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--conditions", nargs="+",
                        default=["A", "A_random", "A_fixed", "NL", "NL_ft"])
    parser.add_argument("--encoders", nargs="+",
                        default=["clip", "dino", "siglip", "mae"],
                        choices=["clip", "dino", "siglip", "mae"])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--ec_json",    required=True, help="EC corpus JSON")
    parser.add_argument("--gen_dir",    required=True,
                        help="Root directory of generated images (subfolders: A, A_random, ...)")
    parser.add_argument("--val_ann",    default=None, help="Path to captions_val2017.json")
    parser.add_argument("--out_csv",    required=True, help="Output CSV path")
    args = parser.parse_args()

    EC_JSON = args.ec_json
    GEN_DIR = Path(args.gen_dir)
    VAL_ANN = Path(args.val_ann) if args.val_ann else None

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    with open(EC_JSON) as f:
        ec = json.load(f)
    val_all = ec["val"]
    n = min(args.n_samples, len(val_all))

    # Generated images were produced with random.seed(42) + random.sample(val_all, 5000),
    # giving a fixed ordering stored as 0000_imgid*.png, 0001_imgid*.png, ...
    # Reference images must be loaded in the SAME order so pairs match.
    import random as stdlib_random
    stdlib_random.seed(args.seed)
    gen_order = stdlib_random.sample(val_all, len(val_all))
    val_entries = gen_order[:n]  # first n entries match generated images 0000..{n-1}

    ref_images = [Image.open(e["image_path"]).convert("RGB") for e in val_entries]

    # ── Build encoders ─────────────────────────────────────────────────────────
    encoder_factories = {
        "clip": make_clip_encoder,
        "dino": make_dino_encoder,
        "siglip": make_siglip_encoder,
        "mae": make_mae_encoder,
    }
    encoders: dict[str, Callable | None] = {}
    ref_embs: dict[str, torch.Tensor] = {}

    for name in args.encoders:
        print(f"Loading {name} encoder ...")
        embed_fn = encoder_factories[name](device)
        if embed_fn is None:
            continue
        encoders[name] = embed_fn
        print(f"  Embedding {n} reference images ...")
        ref_embs[name] = embed_fn(ref_images, batch_size=args.batch_size)
        print(f"  ref_embs[{name}]: {ref_embs[name].shape}")

    if not encoders:
        print("No encoders available. Exiting.")
        return

    # ── Per-condition evaluation ───────────────────────────────────────────────
    rows = []

    for cond in args.conditions:
        img_dir = GEN_DIR / cond
        if not img_dir.exists():
            print(f"  SKIP {cond}: not found")
            continue

        gen_paths = sorted(img_dir.glob("*.png"))
        gen_images = [Image.open(gen_paths[i]).convert("RGB")
                      for i in range(n) if i < len(gen_paths)]
        actual_n = len(gen_images)
        print(f"\n[{cond}] n={actual_n}")

        row = {"condition": cond, "n": actual_n}

        for enc_name, embed_fn in encoders.items():
            print(f"  [{enc_name}] embedding generated images ...")
            gen_embs = embed_fn(gen_images, batch_size=args.batch_size)  # [actual_n, D]
            ref_e = ref_embs[enc_name][:actual_n]

            per_sample_sims = (gen_embs * ref_e).sum(-1).numpy()  # [actual_n]
            mean, lo, hi = bootstrap_mean_ci(per_sample_sims, args.n_bootstrap)

            row[f"{enc_name}_mean"] = round(mean, 4)
            row[f"{enc_name}_ci_lo"] = round(lo, 4)
            row[f"{enc_name}_ci_hi"] = round(hi, 4)
            print(f"    {enc_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

        rows.append(row)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    fieldnames = ["condition", "n"]
    for enc_name in encoders:
        fieldnames += [f"{enc_name}_mean", f"{enc_name}_ci_lo", f"{enc_name}_ci_hi"]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {args.out_csv}")

    # ── Print rank consistency table ──────────────────────────────────────────
    print("\n=== Rank consistency across encoders ===")
    enc_names = list(encoders.keys())
    for enc_name in enc_names:
        key = f"{enc_name}_mean"
        ranked = sorted(rows, key=lambda r: r.get(key, 0), reverse=True)
        print(f"  {enc_name}: " + " > ".join(r["condition"] for r in ranked
                                              if key in r))


if __name__ == "__main__":
    main()
