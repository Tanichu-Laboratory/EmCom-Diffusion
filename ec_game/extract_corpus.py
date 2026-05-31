"""
Extract EC corpus from a trained SimpleEmCom (referential game) checkpoint.

Usage:
    python -m ec_game.extract_corpus \
        --ckpt  outputs/ec_game/referential_YYYYMMDD/checkpoint_29.pth \
        --ec_json data/ec_captions.json \
        --output  outputs/ec_corpus.json

Output format (compatible with diffusion/train.py):
    {
      "vocab_size": 256,
      "num_tokens": 8,
      "pad_token_id": 256,
      "ckpt_path": "...",
      "train": [{"image_id": ..., "image_path": ..., "ec_tokens": [...]}, ...],
      "val":   [...]
    }
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ec_game.models.emcom import build_model


# ──────────────────────────────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    def __init__(self, entries, tf):
        self.entries = entries
        self.tf      = tf

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e   = self.entries[i]
        img = Image.open(e['image_path']).convert('RGB')
        return self.tf(img), e['image_id']


def _extract_split(model, entries, device, batch_size, num_workers):
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])
    ds     = _ImageDataset(entries, tf)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, shuffle=False)

    id2path = {e['image_id']: e['image_path'] for e in entries}
    results = []

    model.eval()
    with torch.no_grad():
        for imgs, ids in loader:
            imgs   = imgs.to(device)
            tokens = model.get_tokens(imgs)          # [B, K]
            for j in range(len(ids)):
                image_id = ids[j].item() if isinstance(ids[j], torch.Tensor) else ids[j]
                results.append({
                    'image_id':   image_id,
                    'image_path': id2path[image_id],
                    'ec_tokens':  tokens[j].tolist(),
                })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        required=True,  help='Path to checkpoint .pth')
    parser.add_argument('--ec_json',     required=True,  help='ec_captions.json (train/val split)')
    parser.add_argument('--output',      required=True,  help='Output corpus JSON path')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    config = ckpt['config']
    print(f"[load] game={config['game']}  V={config['vocab_size']}  K={config['num_slots']}")

    model = build_model(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Load image split from ec_captions.json
    with open(args.ec_json) as f:
        ec_data = json.load(f)

    output = {
        'vocab_size':   config['vocab_size'],
        'num_tokens':   config['num_slots'],
        'pad_token_id': config['vocab_size'],
        'ckpt_path':    str(Path(args.ckpt).resolve()),
        'train':        [],
        'val':          [],
    }

    for split in ('train', 'val'):
        entries = ec_data.get(split, [])
        if not entries:
            continue
        print(f"[extract] {split}: {len(entries)} images …")
        output[split] = _extract_split(
            model, entries, device, args.batch_size, args.num_workers
        )
        print(f"  done: {len(output[split])} entries")

        # Quick diversity check
        all_tokens = [t for e in output[split] for t in e['ec_tokens']]
        V = config['vocab_size']
        counts = [0] * V
        for t in all_tokens:
            counts[t] += 1
        total = len(all_tokens)
        probs  = [c / total for c in counts if c > 0]
        ent    = -sum(p * math.log(p) for p in probs)
        used   = sum(1 for c in counts if c > 0)
        print(f"  vocab used: {used}/{V} ({100*used/V:.1f}%)  "
              f"entropy: {ent:.3f}/{math.log(V):.3f}  "
              f"ratio: {ent/math.log(V):.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f)
    print(f"[done] corpus saved → {out_path}  "
          f"(train={len(output['train'])}, val={len(output['val'])})")


if __name__ == '__main__':
    main()
