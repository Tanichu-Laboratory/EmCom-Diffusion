"""Export the exact evaluation image IDs (reviewer R1-6).

R1 asked that the release include "the exact 5,000-image evaluation IDs".
Two orderings matter and they are NOT the same list:

  corpus order     - the order of corpus["val"], used to index every cached
                     embedding matrix (cache_gt_clip_vitl14.npy,
                     cache_dino_vitb14_gen.npy, B_edit_mat.npy, ...) and
                     therefore the index space of all Table 4/5/6 triplets.
  evaluation order - random.seed(42); random.sample(val, 5000), reproducing
                     diffusion/inference.py:275-282. Generated images are
                     written as {position:04d}_imgid{image_id}.png in THIS
                     order.

Both are emitted so a third party can line up any artefact with any other.

Usage:
    python experiments/export_eval_ids.py \
        --ec_json outputs/ec_corpus_blip_ref_ep29.json \
        --out data/eval_image_ids.json
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec_json", required=True, help="EC corpus JSON")
    ap.add_argument("--seed", type=int, default=42,
                    help="must match diffusion/inference.py --seed")
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.ec_json) as f:
        corpus = json.load(f)
    val = corpus["val"]
    corpus_order = [e["image_id"] for e in val]

    # reproduce diffusion/inference.py:275-282 exactly
    random.seed(args.seed)
    sampled = random.sample(val, min(args.n_samples, len(val)))
    eval_order = [e["image_id"] for e in sampled]

    out = {
        "source_corpus": Path(args.ec_json).name,
        "vocab_size": corpus.get("vocab_size"),
        "num_tokens": corpus.get("num_tokens"),
        "n_val": len(val),
        "seed": args.seed,
        "note": ("corpus_order indexes every cached embedding matrix and all "
                 "Table 4/5/6 triplet indices. evaluation_order is the order "
                 "generated images are written in "
                 "({position:04d}_imgid{image_id}.png). The two coincide only "
                 "if n_samples == len(val) AND the sampler is the identity, "
                 "which it is not -- keep them distinct."),
        "corpus_order": corpus_order,
        "evaluation_order": eval_order,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    same = sum(1 for a, b in zip(corpus_order, eval_order) if a == b)
    print(f"n_val={len(val)}  n_eval={len(eval_order)}  "
          f"unique={len(set(eval_order))}")
    print(f"positions where the two orders agree: {same}/{len(eval_order)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
