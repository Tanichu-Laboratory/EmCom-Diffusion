"""
Quantitative evaluation: CLIPScore (text & image), DINO Score (image), and FID.

Evaluates variants: 3 settings × 3 input types (EC / Random / Fixed) + optional NL baseline.
Reports bootstrap 95% CI for robustness.
Output: outputs/pilot_metrics.csv

DINO Score uses DINOv2-base (self-supervised, no language supervision) to measure
visual similarity without human-language bias.

Usage:
  python experiments/evaluate.py \
    --gen_dirs outputs/generated_images/A outputs/generated_images/B outputs/generated_images/C \
    --rand_dirs outputs/generated_images/A_random ... \
    --fixed_dirs outputs/generated_images/A_fixed ... \
    --nl_dir outputs/generated_images/NL \
    --labels A B C \
    --n_samples 500 --n_bootstrap 1000
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", str(_HERE.parent)))
COCO_ROOT = Path(os.environ.get("COCO_ROOT", "data/coco2017"))
VAL_ANN = COCO_ROOT / "annotations" / "captions_val2017.json"


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


def load_dino(device):
    """Load DINOv2 ViT-B/14 via torch.hub (self-supervised, no language)."""
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14",
        pretrained=True, verbose=False,
    ).to(device).eval()
    # Standard ImageNet normalisation used by DINOv2
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


@torch.no_grad()
def embed_images(clip_model, proc, pil_images, device, batch_size=32):
    embs = []
    for i in range(0, len(pil_images), batch_size):
        inp = proc(images=pil_images[i:i+batch_size], return_tensors="pt").to(device)
        embs.append(F.normalize(clip_model.get_image_features(**inp), dim=-1).cpu())
    return torch.cat(embs)  # [N, 512]


@torch.no_grad()
def embed_images_dino(dino_model, preprocess, pil_images, device, batch_size=32):
    """Extract DINOv2 CLS token features (no language supervision)."""
    embs = []
    for i in range(0, len(pil_images), batch_size):
        batch = torch.stack([preprocess(img) for img in pil_images[i:i+batch_size]]).to(device)
        cls = dino_model(batch)                    # [B, 768] CLS token
        embs.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(embs)  # [N, 768]


@torch.no_grad()
def embed_texts(clip_model, proc, captions, device, batch_size=32):
    embs = []
    for i in range(0, len(captions), batch_size):
        inp = proc(text=captions[i:i+batch_size], return_tensors="pt",
                   padding=True, truncation=True).to(device)
        embs.append(F.normalize(clip_model.get_text_features(**inp), dim=-1).cpu())
    return torch.cat(embs)  # [N, 512]


def bootstrap_mean_ci(scores, n_bootstrap=1000, ci=0.95, seed=0):
    """Return (mean, lower, upper) via bootstrap."""
    rng = np.random.default_rng(seed)
    scores = np.array(scores)
    boot_means = [rng.choice(scores, size=len(scores), replace=True).mean()
                  for _ in range(n_bootstrap)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(scores.mean()), float(lo), float(hi)


def compute_fid(gen_images, ref_images, device, batch_size=32):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return float("nan")

    fid = FrechetInceptionDistance(feature=2048).to(device)

    def to_uint8(imgs):
        out = []
        for img in imgs:
            arr = np.array(img.convert("RGB").resize((299, 299)))
            out.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(out).to(device)

    for i in range(0, len(ref_images), batch_size):
        fid.update(to_uint8(ref_images[i:i+batch_size]), real=True)
    for i in range(0, len(gen_images), batch_size):
        fid.update(to_uint8(gen_images[i:i+batch_size]), real=False)
    return float(fid.compute().item())


def load_pil_from_dir(img_dir, n):
    paths = sorted(Path(img_dir).glob("*.png"))[:n]
    return [Image.open(p).convert("RGB") for p in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dirs",   nargs="+", required=True)
    parser.add_argument("--labels",     nargs="+", default=None)
    parser.add_argument("--n_samples",  type=int, default=500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--ec_json",
                        default=str(PROJECT_DIR / "outputs" / "ec_captions.json"))
    parser.add_argument("--out_csv",
                        default=str(PROJECT_DIR / "outputs" / "pilot_metrics.csv"))
    parser.add_argument("--rand_dirs",  nargs="+", default=None)
    parser.add_argument("--fixed_dirs", nargs="+", default=None)
    parser.add_argument("--nl_dir",     default=None,
                        help="Dir of images generated by base SD + NL captions (unfinetuned baseline)")
    parser.add_argument("--nl_ft_dir", default=None,
                        help="Dir of images generated by COCO-finetuned SD + NL captions (fair baseline)")
    args = parser.parse_args()

    labels = args.labels or [chr(65 + i) for i in range(len(args.gen_dirs))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.ec_json) as f:
        ec = json.load(f)
    random.seed(args.seed)
    val_all     = ec["val"]
    val_entries = random.sample(val_all, min(args.n_samples, len(val_all)))

    from data import load_nl_captions
    nl_caps    = load_nl_captions(str(VAL_ANN))
    captions   = [nl_caps.get(e["image_id"], [""])[0] for e in val_entries]
    ref_images = [Image.open(e["image_path"]).convert("RGB") for e in val_entries]

    print("Loading CLIP ...")
    clip_model, clip_proc = load_clip(device)

    print("Loading DINOv2 ...")
    dino_model, dino_proc = load_dino(device)

    print("Embedding reference images and NL captions ...")
    ref_clip_emb = embed_images(clip_model, clip_proc, ref_images, device)  # [N, 512]
    ref_dino_emb = embed_images_dino(dino_model, dino_proc, ref_images, device)  # [N, 768]
    txt_emb      = embed_texts(clip_model, clip_proc, captions, device)     # [N, 512]

    def eval_variant(label, input_type, img_dir):
        print(f"  [{label}-{input_type}] from {img_dir}")
        gen_images = load_pil_from_dir(img_dir, args.n_samples)
        n = len(gen_images)
        if n < args.n_samples:
            print(f"    WARNING: only {n} images found (expected {args.n_samples})")

        gen_clip_emb = embed_images(clip_model, clip_proc, gen_images, device)       # [n, 512]
        gen_dino_emb = embed_images_dino(dino_model, dino_proc, gen_images, device)  # [n, 768]

        # per-sample scores
        cs_text_scores  = (gen_clip_emb * txt_emb[:n]).sum(-1).numpy()
        cs_image_scores = (gen_clip_emb * ref_clip_emb[:n]).sum(-1).numpy()
        dino_scores     = (gen_dino_emb * ref_dino_emb[:n]).sum(-1).numpy()

        cs_text_mean,  cs_text_lo,  cs_text_hi  = bootstrap_mean_ci(
            cs_text_scores,  args.n_bootstrap)
        cs_image_mean, cs_image_lo, cs_image_hi = bootstrap_mean_ci(
            cs_image_scores, args.n_bootstrap)
        dino_mean,     dino_lo,     dino_hi     = bootstrap_mean_ci(
            dino_scores,     args.n_bootstrap)

        fid = compute_fid(gen_images, ref_images[:n], device)

        return {
            "setting":            label,
            "input_type":         input_type,
            "n":                  n,
            "clip_score_text":    round(cs_text_mean,  4),
            "clip_text_ci_lo":    round(cs_text_lo,    4),
            "clip_text_ci_hi":    round(cs_text_hi,    4),
            "clip_score_image":   round(cs_image_mean, 4),
            "clip_image_ci_lo":   round(cs_image_lo,   4),
            "clip_image_ci_hi":   round(cs_image_hi,   4),
            "dino_score":         round(dino_mean,     4),
            "dino_ci_lo":         round(dino_lo,       4),
            "dino_ci_hi":         round(dino_hi,       4),
            "fid":                round(fid, 2),
        }

    rows = []

    for i, (label, gen_dir) in enumerate(zip(labels, args.gen_dirs)):
        rows.append(eval_variant(label, "EC", gen_dir))
        if args.rand_dirs:
            rows.append(eval_variant(label, "Random", args.rand_dirs[i]))
        if args.fixed_dirs:
            rows.append(eval_variant(label, "Fixed", args.fixed_dirs[i]))

    if args.nl_dir:
        rows.append(eval_variant("SD-NL-base", "NL", args.nl_dir))
    if args.nl_ft_dir:
        rows.append(eval_variant("SD-NL-ft", "NL", args.nl_ft_dir))

    fieldnames = ["setting", "input_type", "n",
                  "clip_score_text", "clip_text_ci_lo", "clip_text_ci_hi",
                  "clip_score_image", "clip_image_ci_lo", "clip_image_ci_hi",
                  "dino_score", "dino_ci_lo", "dino_ci_hi",
                  "fid"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.out_csv}")
    for r in rows:
        print(f"  {r['setting']}-{r['input_type']} (n={r['n']}): "
              f"clip_text={r['clip_score_text']:.4f} [{r['clip_text_ci_lo']:.4f}, {r['clip_text_ci_hi']:.4f}]  "
              f"clip_img={r['clip_score_image']:.4f} [{r['clip_image_ci_lo']:.4f}, {r['clip_image_ci_hi']:.4f}]  "
              f"dino={r['dino_score']:.4f} [{r['dino_ci_lo']:.4f}, {r['dino_ci_hi']:.4f}]  "
              f"fid={r['fid']:.2f}")


if __name__ == "__main__":
    main()
