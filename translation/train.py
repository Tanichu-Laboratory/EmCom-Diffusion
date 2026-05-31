#!/usr/bin/env python3
"""
Train EC→NL seq2seq translator (Yao et al. 2022, corpus-transfer baseline).

Usage:
    python -m translation.train \
        --ec_corpus outputs/ec_corpus.json \
        --coco_root data/coco2017 \
        --vocab     outputs/translation/vocab.json \
        --ckpt_dir  outputs/translation/checkpoints \
        --epochs 10
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from translation.vocab import build_vocab, Vocab
from translation.dataset import ECNLDataset, collate_fn
from translation.model import ECToNLTranslator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ec_corpus",      required=True, help="EC corpus JSON")
    p.add_argument("--coco_root",      default=os.environ.get("COCO_ROOT", "data/coco2017"))
    p.add_argument("--vocab",          required=True, help="Path to save/load vocab JSON")
    p.add_argument("--ckpt_dir",       required=True, help="Directory to save checkpoints")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--d_model",        type=int,   default=256)
    p.add_argument("--nhead",          type=int,   default=8)
    p.add_argument("--n_enc",          type=int,   default=3)
    p.add_argument("--n_dec",          type=int,   default=6)
    p.add_argument("--d_ff",           type=int,   default=1024)
    p.add_argument("--max_nl_len",     type=int,   default=30)
    p.add_argument("--min_vocab_freq", type=int,   default=3)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    coco_root = Path(args.coco_root)
    train_ann = str(coco_root / "annotations" / "captions_train2017.json")
    val_ann   = str(coco_root / "annotations" / "captions_val2017.json")
    vocab_path = Path(args.vocab)
    ckpt_dir   = Path(args.ckpt_dir)

    vocab = build_vocab(train_ann, min_freq=args.min_vocab_freq)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(str(vocab_path))
    print(f"[train] vocab size={len(vocab)}, saved to {vocab_path}")

    train_ds = ECNLDataset(args.ec_corpus, train_ann, vocab,
                           split="train", max_nl_len=args.max_nl_len, random_caption=True)
    val_ds   = ECNLDataset(args.ec_corpus, val_ann,   vocab,
                           split="val",   max_nl_len=args.max_nl_len, random_caption=False)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    _collate = partial(collate_fn, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=_collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=_collate, pin_memory=True)

    model = ECToNLTranslator(
        ec_vocab_size=256,
        nl_vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        n_enc_layers=args.n_enc,
        n_dec_layers=args.n_dec,
        d_ff=args.d_ff,
        ec_pad_id=256,
        nl_pad_id=vocab.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ec       = batch["ec_tokens"].to(device)
            nl_in    = batch["nl_input"].to(device)
            nl_tgt   = batch["nl_target"].to(device)
            logits   = model(ec, nl_in)
            B, T, V  = logits.shape
            loss     = criterion(logits.view(B * T, V), nl_tgt.view(B * T))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ec     = batch["ec_tokens"].to(device)
                nl_in  = batch["nl_input"].to(device)
                nl_tgt = batch["nl_target"].to(device)
                logits = model(ec, nl_in)
                B, T, V = logits.shape
                val_loss += criterion(logits.view(B * T, V), nl_tgt.view(B * T)).item()
        val_loss /= len(val_loader)
        print(f"[epoch {epoch:3d}] train={train_loss:.4f}  val={val_loss:.4f}")

        ckpt = {
            "epoch": epoch, "val_loss": val_loss,
            "model_state": model.state_dict(),
            "args": {"d_model": args.d_model, "nhead": args.nhead,
                     "n_enc": args.n_enc, "n_dec": args.n_dec, "d_ff": args.d_ff},
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:02d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> best saved (val={best_val_loss:.4f})")

    print(f"[done] best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
