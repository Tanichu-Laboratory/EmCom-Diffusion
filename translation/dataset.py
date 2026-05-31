"""
Dataset for EC→NL seq2seq training.

Each sample: EC token sequence (length K=8) paired with one COCO caption.
During training we randomly sample one caption per image per access.
For eval we use the first caption.
"""

import json
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from yao_translation.vocab import Vocab


def load_coco_captions(ann_path: str) -> dict[int, list[str]]:
    with open(ann_path) as f:
        data = json.load(f)
    id2caps: dict[int, list[str]] = defaultdict(list)
    for a in data["annotations"]:
        id2caps[a["image_id"]].append(a["caption"])
    return dict(id2caps)


class ECNLDataset(Dataset):
    """
    Pairs EC token sequences with COCO NL captions (word-level vocab).

    split: "train" | "val"
    random_caption: True → randomly sample one caption per access (training)
                    False → always use first caption (eval)
    """

    def __init__(
        self,
        ec_corpus_path: str,
        coco_ann_path: str,
        vocab: Vocab,
        split: str = "train",
        max_nl_len: int = 30,
        random_caption: bool = True,
    ):
        super().__init__()
        with open(ec_corpus_path) as f:
            corpus = json.load(f)

        self.entries = corpus[split]
        self.id2caps = load_coco_captions(coco_ann_path)
        self.vocab = vocab
        self.max_nl_len = max_nl_len
        self.random_caption = random_caption

        self.entries = [e for e in self.entries if e["image_id"] in self.id2caps]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        ec_tokens = torch.tensor(e["ec_tokens"], dtype=torch.long)

        caps = self.id2caps[e["image_id"]]
        cap = random.choice(caps) if self.random_caption else caps[0]

        ids = self.vocab.encode(cap)[: self.max_nl_len - 2]
        ids = [self.vocab.bos_id] + ids + [self.vocab.eos_id]

        nl_input  = torch.tensor(ids[:-1], dtype=torch.long)   # decoder input
        nl_target = torch.tensor(ids[1:],  dtype=torch.long)   # shifted target

        return {
            "image_id":  e["image_id"],
            "ec_tokens": ec_tokens,
            "nl_input":  nl_input,
            "nl_target": nl_target,
        }


def collate_fn(batch: list[dict], pad_id: int) -> dict:
    max_nl = max(x["nl_input"].size(0) for x in batch)
    ec_tokens = torch.stack([x["ec_tokens"] for x in batch])
    image_ids = [x["image_id"] for x in batch]

    nl_input  = torch.full((len(batch), max_nl), pad_id, dtype=torch.long)
    nl_target = torch.full((len(batch), max_nl), -100,   dtype=torch.long)

    for i, x in enumerate(batch):
        T = x["nl_input"].size(0)
        nl_input[i, :T]  = x["nl_input"]
        nl_target[i, :T] = x["nl_target"]

    return {
        "image_ids": image_ids,
        "ec_tokens": ec_tokens,
        "nl_input":  nl_input,
        "nl_target": nl_target,
    }
