"""
Cache DINOv2 ViT-B/14 embeddings for generated images.

Reads generated images in the order determined by the EC corpus val split,
encodes them with DINOv2, and saves a (N, 768) float32 numpy array.

Usage:
    python experiments/cache_dino_gen.py \
        --gen_dir  outputs/generated_images/A \
        --ec_json  outputs/ec_corpus.json \
        --output   outputs/cache_dino_vitb14_gen.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",  required=True,
                        help="Directory containing generated images (*.png)")
    parser.add_argument("--ec_json",  required=True,
                        help="EC corpus JSON; used to get the val image ordering")
    parser.add_argument("--output",   required=True, help="Output .npy path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    save_path = Path(args.output)
    if save_path.exists():
        print(f"Already exists: {save_path}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading DINOv2 ViT-B/14 ...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    dino = dino.to(device).eval()

    with open(args.ec_json) as f:
        corpus = json.load(f)
    val_entries = corpus["val"]
    N = len(val_entries)
    print(f"Val entries: {N}")

    gen_dir = Path(args.gen_dir)

    feats = []
    batch_imgs = []
    BATCH = args.batch_size

    for i, entry in enumerate(val_entries):
        iid = entry["image_id"]
        # generated images are named {idx:04d}_imgid{iid}.png
        candidates = list(gen_dir.glob(f"{i:04d}_imgid{iid}.png"))
        if not candidates:
            candidates = list(gen_dir.glob(f"*_imgid{iid}.png"))
        if candidates:
            img = Image.open(candidates[0]).convert("RGB")
            batch_imgs.append(TRANSFORM(img))
        else:
            print(f"  WARNING: not found idx={i} iid={iid}, using zeros")
            batch_imgs.append(torch.zeros(3, 224, 224))

        if len(batch_imgs) == BATCH or i == N - 1:
            batch_t = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                f = dino(batch_t)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            batch_imgs = []
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{N}")

    feats_all = np.concatenate(feats, axis=0).astype(np.float32)
    print(f"Shape: {feats_all.shape}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, feats_all)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
