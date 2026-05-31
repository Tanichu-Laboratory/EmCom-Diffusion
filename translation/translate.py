#!/usr/bin/env python3
"""
Translate EC tokens → NL captions and cache CLIP text embeddings.

Usage:
    python -m translation.translate \
        --ec_corpus   outputs/ec_corpus.json \
        --ckpt        outputs/translation/checkpoints/best.pt \
        --vocab       outputs/translation/vocab.json \
        --coco_root   data/coco2017 \
        --output      outputs/translation/translations_val5k.json \
        --clip_cache  outputs/cache_pred_clip_text_vitl14.npy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from translation.vocab import Vocab
from translation.dataset import load_coco_captions
from translation.model import ECToNLTranslator


class ValECDataset(Dataset):
    def __init__(self, val_entries):
        self.entries = val_entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return {
            "image_id":   e["image_id"],
            "image_path": e["image_path"],
            "ec_tokens":  torch.tensor(e["ec_tokens"], dtype=torch.long),
        }


def collate_val(batch):
    return {
        "image_ids":   [x["image_id"]   for x in batch],
        "image_paths": [x["image_path"] for x in batch],
        "ec_tokens":   torch.stack([x["ec_tokens"] for x in batch]),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ec_corpus",  required=True)
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--vocab",      required=True)
    p.add_argument("--coco_root",  default=os.environ.get("COCO_ROOT", "data/coco2017"))
    p.add_argument("--output",     required=True, help="JSON with predicted captions")
    p.add_argument("--clip_cache", default=None,
                   help="If set, cache CLIP ViT-L/14 text embeddings of translations here (.npy)")
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--max_nl_len",  type=int,   default=30)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p",       type=float, default=0.9)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[translate] device={device}")

    val_ann = Path(args.coco_root) / "annotations" / "captions_val2017.json"

    vocab = Vocab.load(args.vocab)
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    ta    = ckpt["args"]
    print(f"[translate] epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.4f}")

    model = ECToNLTranslator(
        ec_vocab_size=256,
        nl_vocab_size=len(vocab),
        d_model=ta["d_model"],
        nhead=ta["nhead"],
        n_enc_layers=ta["n_enc"],
        n_dec_layers=ta["n_dec"],
        d_ff=ta["d_ff"],
        ec_pad_id=256,
        nl_pad_id=vocab.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    with open(args.ec_corpus) as f:
        corpus = json.load(f)
    val_entries = corpus["val"]
    id2caps     = load_coco_captions(str(val_ann))

    loader = DataLoader(ValECDataset(val_entries),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_val)

    results, n_done = [], 0
    for batch in loader:
        ec = batch["ec_tokens"].to(device)
        token_seqs = model.sample_decode(
            ec, vocab.bos_id, vocab.eos_id,
            max_len=args.max_nl_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for i, (iid, ipath) in enumerate(zip(batch["image_ids"], batch["image_paths"])):
            caption = vocab.decode(token_seqs[i])
            results.append({
                "image_id":           iid,
                "image_path":         ipath,
                "ec_tokens":          val_entries[n_done]["ec_tokens"],
                "predicted_caption":  caption,
                "reference_captions": id2caps.get(iid, []),
            })
            n_done += 1
        if n_done % 500 == 0 or n_done == len(val_entries):
            print(f"  {n_done}/{len(val_entries)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"val": results}, f, ensure_ascii=False)
    print(f"[done] saved {len(results)} entries → {out_path}")

    # Optionally cache CLIP text embeddings
    if args.clip_cache:
        print("Computing CLIP ViT-L/14 text embeddings of translations ...")
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_model = clip_model.to(device).eval()
        tokenizer  = open_clip.get_tokenizer("ViT-L-14")

        embs = []
        for start in range(0, len(results), 256):
            batch_caps = [r["predicted_caption"] for r in results[start:start+256]]
            toks = tokenizer(batch_caps).to(device)
            with torch.no_grad():
                feat = clip_model.encode_text(toks)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().numpy())
        all_embs = np.concatenate(embs, axis=0).astype(np.float32)
        cache_path = Path(args.clip_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, all_embs)
        print(f"CLIP cache saved → {cache_path} shape={all_embs.shape}")


if __name__ == "__main__":
    main()
