#!/usr/bin/env python3
"""
Compute referential-game R@1 on COCO val 5,000 images (Appendix D).

Protocol:
  - Forward all val images through Speaker + Listener
  - Build 5000×5000 similarity matrix: S[i,j] = txt_feat[i] · img_feat[j]
  - R@1 = fraction of images where S[i,i] is the maximum in row i
  - Also reports 128-way R@1 (matching training setup)

Usage:
    python experiments/eval_r1_val5k.py \
        --ckpt   outputs/ec_game/referential_.../checkpoint_29.pth \
        --corpus outputs/ec_corpus.json \
        --output outputs/r1_val5k.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ec_game.models.emcom import build_model

IMG_SIZE  = 224
TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


@torch.no_grad()
def encode_all(model, val_entries, device, batch_size=128):
    img_feats, txt_feats = [], []
    model.eval()

    for start in range(0, len(val_entries), batch_size):
        batch = val_entries[start: start + batch_size]
        imgs  = torch.stack([
            TRANSFORM(Image.open(e['image_path']).convert('RGB'))
            for e in batch
        ]).to(device)

        embeds  = model.visual_encoder(imgs)
        img_f   = F.normalize(model.vision_proj(embeds[:, 0, :]), dim=-1)

        tokens  = model.get_tokens(imgs)            # [B, K]
        V       = model.vocab_size
        soft    = F.one_hot(tokens, V).float()
        txt_f, _ = model._encode_message(soft)

        img_feats.append(img_f.cpu())
        txt_feats.append(txt_f.cpu())

    return torch.cat(img_feats), torch.cat(txt_feats)


def r_at_1_full(img_feats, txt_feats):
    sim = txt_feats @ img_feats.T    # [N, N]
    rank = (sim > sim[torch.arange(len(sim)), torch.arange(len(sim))].unsqueeze(1)).sum(1)
    return (rank == 0).float().mean().item()


def r_at_1_nway(img_feats, txt_feats, n_way=128, n_trials=1000, seed=42):
    rng = torch.Generator(); rng.manual_seed(seed)
    N   = len(img_feats)
    correct = 0
    for _ in range(n_trials):
        idx  = torch.randperm(N, generator=rng)[:n_way]
        pool_img = img_feats[idx]
        pool_txt = txt_feats[idx]
        sim  = pool_txt @ pool_img.T   # [n_way, n_way]
        correct += (sim.argmax(1) == torch.arange(n_way)).sum().item()
    return correct / (n_trials * n_way)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True, help='Checkpoint .pth')
    parser.add_argument('--corpus',     required=True, help='EC corpus JSON')
    parser.add_argument('--output',     required=True, help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt   = torch.load(args.ckpt, map_location='cpu')
    config = ckpt['config']
    model  = build_model(config).to(device).eval()
    model.load_state_dict(ckpt['model'])
    print(f"Loaded: V={config['vocab_size']} K={config['num_slots']}")

    with open(args.corpus) as f:
        corpus = json.load(f)
    val_entries = corpus['val']
    print(f"Val entries: {len(val_entries)}")

    t0 = time.time()
    img_feats, txt_feats = encode_all(model, val_entries, device, args.batch_size)
    print(f"Encoded in {time.time()-t0:.1f}s")

    r1_full = r_at_1_full(img_feats, txt_feats)
    r1_128  = r_at_1_nway(img_feats, txt_feats, n_way=128)
    print(f"R@1 (5000-way): {r1_full:.4f}")
    print(f"R@1  (128-way): {r1_128:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("condition,r1_5000way,r1_128way\n")
        name = Path(args.ckpt).parent.name
        f.write(f"{name},{r1_full:.4f},{r1_128:.4f}\n")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
