"""
PyTorch Dataset for (COCO image, emergent-language token sequence) pairs.

ec_captions.json format (produced by generate_ec_captions.py):
  {
    "vocab_size":    <int>,          # number of distinct token IDs (0..vocab_size-1)
    "num_tokens":    <int>,          # K tokens per image (fixed)
    "pad_token_id":  <int>,          # ID used for padding (= vocab_size)
    "train": [{"image_id": int, "image_path": str, "ec_tokens": [int, ...]}, ...],
    "val":   [{"image_id": int, "image_path": str, "ec_tokens": [int, ...]}, ...]
  }
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ECCOCODataset(Dataset):
    """
    Returns (image [3,512,512], ec_tokens [T], attention_mask [T]) per sample.

    Args:
        entries:       list of {"image_id", "image_path", "ec_tokens"}
        pad_token_id:  ID used for padding; sequences are padded to max_seq_len
        max_seq_len:   fixed sequence length (default: K from the data)
        nl_captions:   optional dict {image_id: [str, ...]} for reference
        image_size:    target resolution for SD 1.5 (default 512)
    """

    def __init__(
        self,
        entries: List[Dict],
        pad_token_id: int,
        max_seq_len: Optional[int] = None,
        nl_captions: Optional[Dict[int, List[str]]] = None,
        image_size: int = 512,
    ):
        self.entries = entries
        self.pad_token_id = pad_token_id
        # derive max_seq_len from data if not given
        if max_seq_len is None:
            max_seq_len = max((len(e["ec_tokens"]) for e in entries), default=0)
        self.max_seq_len = max_seq_len
        self.nl_captions = nl_captions or {}

        # SD 1.5 normalisation: [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        img = Image.open(entry["image_path"]).convert("RGB")
        image = self.transform(img)

        tokens = entry["ec_tokens"]
        T = self.max_seq_len
        # pad / truncate
        if len(tokens) >= T:
            tokens = tokens[:T]
            mask = [1] * T
        else:
            pad_len = T - len(tokens)
            mask = [1] * len(tokens) + [0] * pad_len
            tokens = tokens + [self.pad_token_id] * pad_len

        ec_tokens     = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(mask,   dtype=torch.long)

        out = {
            "image": image,
            "ec_tokens": ec_tokens,
            "attention_mask": attention_mask,
            "image_id": entry["image_id"],
        }
        if self.nl_captions:
            caps = self.nl_captions.get(entry["image_id"], [""])
            out["nl_caption"] = caps[0]
        return out


def load_nl_captions(ann_path: str) -> Dict[int, List[str]]:
    """Load COCO captions JSON → {image_id: [caption_str, ...]}."""
    with open(ann_path) as f:
        ann = json.load(f)
    captions: Dict[int, List[str]] = {}
    for item in ann["annotations"]:
        captions.setdefault(item["image_id"], []).append(item["caption"])
    return captions


def build_datasets(
    ec_json_path: str,
    train_ann_path: str,
    val_ann_path: str,
    image_size: int = 512,
) -> Tuple[ECCOCODataset, ECCOCODataset]:
    """Return (train_dataset, val_dataset) from paths."""
    with open(ec_json_path) as f:
        ec = json.load(f)

    pad_token_id = ec["pad_token_id"]
    max_seq_len  = ec["num_tokens"]

    train_nl = load_nl_captions(train_ann_path)
    val_nl   = load_nl_captions(val_ann_path)

    train_ds = ECCOCODataset(
        entries=ec["train"],
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
        nl_captions=train_nl,
        image_size=image_size,
    )
    val_ds = ECCOCODataset(
        entries=ec["val"],
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
        nl_captions=val_nl,
        image_size=image_size,
    )
    return train_ds, val_ds


if __name__ == "__main__":
    import sys

    ec_json   = sys.argv[1] if len(sys.argv) > 1 else "outputs/ec_corpus.json"
    coco_root = os.environ.get("COCO_ROOT", "data/coco2017")
    train_ann = os.path.join(coco_root, "annotations", "captions_train2017.json")
    val_ann   = os.path.join(coco_root, "annotations", "captions_val2017.json")

    train_ds, val_ds = build_datasets(ec_json, train_ann, val_ann)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    loader = DataLoader(train_ds, batch_size=4, num_workers=0, shuffle=True)
    batch = next(iter(loader))

    print("image shape:        ", batch["image"].shape)
    print("image range:        ", batch["image"].min().item(), batch["image"].max().item())
    print("ec_tokens shape:    ", batch["ec_tokens"].shape)
    print("ec_tokens dtype:    ", batch["ec_tokens"].dtype)
    print("attention_mask:     ", batch["attention_mask"])
    print("ec_tokens sample:   ", batch["ec_tokens"][0].tolist())
    print("nl_caption sample:  ", batch.get("nl_caption", ["N/A"])[0])
    print("OK")
