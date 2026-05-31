"""
Word-level vocabulary built from COCO train captions.
Special tokens: <pad>=0, <bos>=1, <eos>=2, <unk>=3
"""

import json
import re
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocab:
    def __init__(self, w2i: dict[str, int]):
        self.w2i = w2i
        self.i2w = {v: k for k, v in w2i.items()}
        self.pad_id = PAD
        self.bos_id = BOS
        self.eos_id = EOS
        self.unk_id = UNK

    def __len__(self):
        return len(self.w2i)

    def encode(self, text: str) -> list[int]:
        return [self.w2i.get(w, UNK) for w in tokenize(text)]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        words = []
        for i in ids:
            if skip_special and i in (PAD, BOS, EOS):
                continue
            words.append(self.i2w.get(i, "<unk>"))
        return " ".join(words)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.w2i, f)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            return cls(json.load(f))


def build_vocab(coco_ann_path: str, min_freq: int = 3) -> Vocab:
    with open(coco_ann_path) as f:
        data = json.load(f)
    counter: Counter = Counter()
    for a in data["annotations"]:
        counter.update(tokenize(a["caption"]))

    w2i: dict[str, int] = {s: i for i, s in enumerate(SPECIALS)}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if word not in w2i:
            w2i[word] = len(w2i)

    print(f"[vocab] size={len(w2i)}  (min_freq={min_freq})")
    return Vocab(w2i)
