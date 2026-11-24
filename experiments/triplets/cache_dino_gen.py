"""
生成画像（ep29, 5000枚）の DINOv2 ViT-B/14 埋め込みを計算してキャッシュする
出力: experiments_v3/outputs/cache_dino_vitb14_gen.npy  shape=(5000, 768)
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms

BASE     = Path("/workspace/project/omoto/emlan-image-generate")
IMGS_DIR = BASE / "outputs/generated_images/ep29"
DATA_DIR = BASE / "outputs/experiments_v2"
OUT      = BASE / "experiments_v3/outputs"
OUT.mkdir(parents=True, exist_ok=True)
SAVE_PATH = OUT / "cache_dino_vitb14_gen.npy"

if SAVE_PATH.exists():
    print(f"Already exists: {SAVE_PATH}")
    exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# DINOv2 ViT-B/14 ロード
print("Loading DINOv2 ViT-B/14...")
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
dino = dino.to(device).eval()

transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 画像 ID 順に並べる
df = pd.read_csv(DATA_DIR / "t11_translation_per_sample_scores.csv")
image_ids = df["image_id"].astype(int).tolist()

def find_gen_path(idx, iid):
    candidates = list(IMGS_DIR.glob(f"{idx:04d}_imgid{iid}.png"))
    if candidates:
        return candidates[0]
    candidates = list(IMGS_DIR.glob(f"*_imgid{iid}.png"))
    return candidates[0] if candidates else None

BATCH = 64
feats = []

print(f"Processing {len(image_ids)} images (batch={BATCH})...")
batch_imgs = []
batch_indices = []

for i, (idx, iid) in enumerate(zip(df["idx"].tolist(), image_ids)):
    path = find_gen_path(idx, iid)
    if path is None:
        print(f"  WARNING: not found idx={idx} iid={iid}, using zeros")
        batch_imgs.append(torch.zeros(3, 224, 224))
    else:
        img = Image.open(path).convert("RGB")
        batch_imgs.append(transform(img))
    batch_indices.append(i)

    if len(batch_imgs) == BATCH or i == len(image_ids) - 1:
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            f = dino(batch_tensor)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        batch_imgs = []
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(image_ids)}")

feats_all = np.concatenate(feats, axis=0).astype(np.float32)
print(f"Shape: {feats_all.shape}")
np.save(SAVE_PATH, feats_all)
print(f"Saved: {SAVE_PATH}")
