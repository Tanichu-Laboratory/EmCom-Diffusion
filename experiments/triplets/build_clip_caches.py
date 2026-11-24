"""
Exp 2a-v3: Caption-identical / Image-different pair analysis

Structural argument: Translation-based evaluation is bounded by caption-distinguishability.

For pairs (x_a, x_b) where:
  caption_sim(a, b) > τ₁   [captions can't distinguish them]
  image_sim(a, b)   < τ₂   [but images are visually different]

Test:
  Translation     → |ROUGE-L(trans_a, ref_a) - ROUGE-L(trans_a, ref_b)| (should be ≈ 0)
  EmCom-Diffusion → CLIP(gen_a, x_a) - CLIP(gen_a, x_b) (should be > 0)

Discrimination accuracy:
  EmCom: P(CLIP(gen_a, x_a) > CLIP(gen_a, x_b)) for qualifying pairs
  Trans: P(ROUGE-L(trans_a, ref_a) > ROUGE-L(trans_a, ref_b)) for qualifying pairs

Outputs:
  t16_caption_id_image_diff_pairs.csv   — qualifying pairs with all scores
  caption_id_image_diff_summary.txt     — key numbers
  fig_exp2a_v3_discrimination.png/pdf   — figure
"""

import json, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from rouge_score import rouge_scorer
from scipy import stats

BASE  = Path("/workspace/project/omoto/emlan-image-generate")
OUT   = BASE / "outputs/experiments_v2"
IMGS  = BASE / "outputs/generated_images/ep29"
VAL   = Path("/workspace/project/datas/dataset/coco2017/val2017")
ANN_CAP = Path("/workspace/project/datas/dataset/coco2017/annotations/captions_val2017.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load CLIP ──────────────────────────────────────────────────────────────────
model, preprocess = clip.load("ViT-L/14", device=DEVICE)
model.eval()

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT / "t11_translation_per_sample_scores.csv")
with open(ANN_CAP) as f:
    coco_cap = json.load(f)
img_to_refs = defaultdict(list)
for ann in coco_cap["annotations"]:
    img_to_refs[ann["image_id"]].append(ann["caption"])

df["all_refs"] = df["image_id"].map(lambda i: img_to_refs.get(i, []))
N = len(df)
print(f"Eval set size: N={N}")

# ── Embed GT images ────────────────────────────────────────────────────────────
BATCH = 64
cache_gt  = OUT / "cache_gt_clip_vitl14.npy"
cache_gen = OUT / "cache_gen_clip_vitl14.npy"
cache_cap = OUT / "cache_cap_clip_vitl14.npy"

def gt_img_path(image_id):
    return VAL / f"{int(image_id):012d}.jpg"

def gen_img_path(idx, image_id):
    p = IMGS / f"{int(idx):04d}_imgid{int(image_id)}.png"
    if not p.exists():
        ms = list(IMGS.glob(f"*_imgid{int(image_id)}.png"))
        if ms:
            return ms[0]
    return p

@torch.no_grad()
def embed_images(paths, desc="images"):
    feats = []
    for i in tqdm(range(0, len(paths), BATCH), desc=desc):
        batch_paths = paths[i:i+BATCH]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        imgs = torch.stack(imgs).to(DEVICE)
        f = model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float().numpy())
    return np.vstack(feats)

@torch.no_grad()
def embed_texts(texts_list, desc="texts"):
    """texts_list: list of lists of strings. Average multiple captions per image."""
    feats = []
    for i in tqdm(range(0, len(texts_list), BATCH), desc=desc):
        batch = texts_list[i:i+BATCH]
        img_feats = []
        for caps in batch:
            if not caps:
                img_feats.append(np.zeros(768))
                continue
            tokens = clip.tokenize(caps[:5], truncate=True).to(DEVICE)
            f = model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            f = f / f.norm()
            img_feats.append(f.cpu().float().numpy())
        feats.extend(img_feats)
    return np.array(feats)

# GT image embeddings
if cache_gt.exists():
    print("Loading cached GT embeddings...")
    gt_feats = np.load(cache_gt)
else:
    print("Computing GT image embeddings...")
    gt_paths = [gt_img_path(iid) for iid in df["image_id"]]
    gt_feats = embed_images(gt_paths, "GT images")
    np.save(cache_gt, gt_feats)
print(f"GT feats: {gt_feats.shape}")

# Generated image embeddings
if cache_gen.exists():
    print("Loading cached gen embeddings...")
    gen_feats = np.load(cache_gen)
else:
    print("Computing generated image embeddings...")
    gen_paths = [gen_img_path(row["idx"], row["image_id"]) for _, row in df.iterrows()]
    gen_feats = embed_images(gen_paths, "Gen images")
    np.save(cache_gen, gen_feats)
print(f"Gen feats: {gen_feats.shape}")

# Caption embeddings (average of all 5 references)
if cache_cap.exists():
    print("Loading cached caption embeddings...")
    cap_feats = np.load(cache_cap)
else:
    print("Computing caption embeddings...")
    all_caps = [img_to_refs.get(iid, [""]) for iid in df["image_id"]]
    cap_feats = embed_texts(all_caps, "Captions")
    np.save(cache_cap, cap_feats)
print(f"Cap feats: {cap_feats.shape}")

# ── Pairwise similarity matrices (chunked to avoid OOM) ───────────────────────
print("Computing pairwise similarity matrices...")

# GT-GT image similarity: N×N
# Caption-Caption similarity: N×N
# Gen-GT cross similarity: N×N

# Use float32 dot products (embeddings are L2-normalized)
CHUNK = 500

gt_feats_t  = torch.from_numpy(gt_feats).to(DEVICE)   # (N, D)
cap_feats_t = torch.from_numpy(cap_feats).to(DEVICE)
gen_feats_t = torch.from_numpy(gen_feats).to(DEVICE)

# We only need to store the qualifying pairs, not the full matrix
# Strategy: scan chunks, find qualifying pairs, compute gen-gt for those pairs

# First determine thresholds from the full distribution
# Sample a subset of pairs to calibrate
print("Calibrating thresholds from 10k random pairs...")
rng = np.random.default_rng(42)
sample_idx = rng.choice(N, size=min(1000, N), replace=False)
gt_sub  = gt_feats_t[sample_idx]
cap_sub = cap_feats_t[sample_idx]

with torch.no_grad():
    gt_sim_sub  = (gt_sub @ gt_feats_t[sample_idx].T).cpu().numpy()
    cap_sim_sub = (cap_sub @ cap_feats_t[sample_idx].T).cpu().numpy()

# Off-diagonal only
mask = ~np.eye(len(sample_idx), dtype=bool)
gt_vals  = gt_sim_sub[mask]
cap_vals = cap_sim_sub[mask]

print(f"\nGT image pairwise sim:  mean={gt_vals.mean():.3f}  p25={np.percentile(gt_vals,25):.3f}  p75={np.percentile(gt_vals,75):.3f}  p90={np.percentile(gt_vals,90):.3f}")
print(f"Caption pairwise sim:   mean={cap_vals.mean():.3f}  p25={np.percentile(cap_vals,25):.3f}  p75={np.percentile(cap_vals,75):.3f}  p90={np.percentile(cap_vals,90):.3f}")

# Set thresholds:
# τ₁: caption similarity > p75 of caption sim  → "captions are similar"
# τ₂: image similarity   < p25 of image sim    → "images are different"
TAU_CAP   = float(np.percentile(cap_vals, 80))   # top 20% caption similarity
TAU_IMG   = float(np.percentile(gt_vals,  20))   # bottom 20% image similarity
print(f"\nThresholds: τ_cap={TAU_CAP:.3f} (caption sim > this), τ_img={TAU_IMG:.3f} (image sim < this)")

# ── Scan all pairs and collect qualifying ones ─────────────────────────────────
print("\nScanning all pairs for qualifying (caption-similar, image-different) pairs...")

pairs = []   # (a, b, gt_sim, cap_sim, clip_gen_a_gt_a, clip_gen_a_gt_b)
MAX_PAIRS = 50000  # cap for memory

with torch.no_grad():
    for i in tqdm(range(0, N, CHUNK)):
        if len(pairs) >= MAX_PAIRS:
            break
        i_end = min(i + CHUNK, N)
        ni = i_end - i

        gt_i  = gt_feats_t[i:i_end]    # (ni, D)
        cap_i = cap_feats_t[i:i_end]
        gen_i = gen_feats_t[i:i_end]

        for j in range(i + 1, N, CHUNK):
            if len(pairs) >= MAX_PAIRS:
                break
            j_end = min(j + CHUNK, N)

            gt_j  = gt_feats_t[j:j_end]
            cap_j = cap_feats_t[j:j_end]
            gen_j = gen_feats_t[j:j_end]

            cap_sim = (cap_i @ cap_j.T).cpu().numpy()  # (ni, nj)
            img_sim = (gt_i  @ gt_j.T).cpu().numpy()

            # Find qualifying cells
            qual_i, qual_j = np.where((cap_sim > TAU_CAP) & (img_sim < TAU_IMG))

            for qi, qj in zip(qual_i, qual_j):
                a = i + qi
                b = j + qj
                # CLIP(gen_a, x_a) already known as clip_sim in df, but recompute for cross
                clip_gen_a_gt_a = float(gen_feats_t[a] @ gt_feats_t[a])
                clip_gen_a_gt_b = float(gen_feats_t[a] @ gt_feats_t[b])
                clip_gen_b_gt_b = float(gen_feats_t[b] @ gt_feats_t[b])
                clip_gen_b_gt_a = float(gen_feats_t[b] @ gt_feats_t[a])
                pairs.append({
                    "a": int(a), "b": int(b),
                    "image_id_a": int(df["image_id"].iloc[a]),
                    "image_id_b": int(df["image_id"].iloc[b]),
                    "cap_sim": float(cap_sim[qi, qj]),
                    "img_sim": float(img_sim[qi, qj]),
                    "clip_gen_a_gt_a": clip_gen_a_gt_a,
                    "clip_gen_a_gt_b": clip_gen_a_gt_b,
                    "clip_gen_b_gt_b": clip_gen_b_gt_b,
                    "clip_gen_b_gt_a": clip_gen_b_gt_a,
                    "rouge_a": float(df["rouge_l"].iloc[a]),
                    "rouge_b": float(df["rouge_l"].iloc[b]),
                    "pred_a": str(df["predicted_caption"].iloc[a]),
                    "pred_b": str(df["predicted_caption"].iloc[b]),
                    "ref_a0": str(df["reference_caption_0"].iloc[a]),
                    "ref_b0": str(df["reference_caption_0"].iloc[b]),
                })
                if len(pairs) >= MAX_PAIRS:
                    break

print(f"\nQualifying pairs found: {len(pairs)}")

if len(pairs) == 0:
    print("No pairs found — try adjusting thresholds")
    sys.exit(0)

pairs_df = pd.DataFrame(pairs)

# ── ROUGE-L cross-evaluation ───────────────────────────────────────────────────
print("Computing ROUGE-L cross-evaluation for qualifying pairs...")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rouge_aa, rouge_ab, rouge_ba, rouge_bb = [], [], [], []

all_refs_a = [img_to_refs.get(int(df["image_id"].iloc[r["a"]]), [""])[0] for _, r in pairs_df.iterrows()]
all_refs_b = [img_to_refs.get(int(df["image_id"].iloc[r["b"]]), [""])[0] for _, r in pairs_df.iterrows()]

for i, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="ROUGE-L cross"):
    pred_a = str(df["predicted_caption"].iloc[int(row["a"])])
    pred_b = str(df["predicted_caption"].iloc[int(row["b"])])
    ref_a  = all_refs_a[i]
    ref_b  = all_refs_b[i]

    rouge_aa.append(scorer.score(ref_a, pred_a)["rougeL"].fmeasure)
    rouge_ab.append(scorer.score(ref_b, pred_a)["rougeL"].fmeasure)
    rouge_ba.append(scorer.score(ref_a, pred_b)["rougeL"].fmeasure)
    rouge_bb.append(scorer.score(ref_b, pred_b)["rougeL"].fmeasure)

pairs_df["rouge_aa"] = rouge_aa
pairs_df["rouge_ab"] = rouge_ab
pairs_df["rouge_ba"] = rouge_ba
pairs_df["rouge_bb"] = rouge_bb

# ── Discrimination scores ──────────────────────────────────────────────────────
# EmCom-Diffusion: correct if gen_a → x_a better than gen_a → x_b
pairs_df["emcom_correct_a"] = pairs_df["clip_gen_a_gt_a"] > pairs_df["clip_gen_a_gt_b"]
pairs_df["emcom_correct_b"] = pairs_df["clip_gen_b_gt_b"] > pairs_df["clip_gen_b_gt_a"]
pairs_df["emcom_correct"]   = pairs_df["emcom_correct_a"] & pairs_df["emcom_correct_b"]
pairs_df["emcom_gap_a"]     = pairs_df["clip_gen_a_gt_a"] - pairs_df["clip_gen_a_gt_b"]
pairs_df["emcom_gap_b"]     = pairs_df["clip_gen_b_gt_b"] - pairs_df["clip_gen_b_gt_a"]
pairs_df["emcom_gap"]       = (pairs_df["emcom_gap_a"] + pairs_df["emcom_gap_b"]) / 2

# Translation: correct if trans_a → ref_a higher than trans_a → ref_b
pairs_df["rouge_correct_a"] = pairs_df["rouge_aa"] > pairs_df["rouge_ab"]
pairs_df["rouge_correct_b"] = pairs_df["rouge_bb"] > pairs_df["rouge_ba"]
pairs_df["rouge_correct"]   = pairs_df["rouge_correct_a"] & pairs_df["rouge_correct_b"]
pairs_df["rouge_gap_a"]     = pairs_df["rouge_aa"] - pairs_df["rouge_ab"]
pairs_df["rouge_gap_b"]     = pairs_df["rouge_bb"] - pairs_df["rouge_ba"]
pairs_df["rouge_gap"]       = (pairs_df["rouge_gap_a"] + pairs_df["rouge_gap_b"]) / 2

# ── Stats ──────────────────────────────────────────────────────────────────────
n_pairs = len(pairs_df)
emcom_acc = pairs_df["emcom_correct"].mean()
rouge_acc = pairs_df["rouge_correct"].mean()
emcom_gap_mean = pairs_df["emcom_gap"].mean()
rouge_gap_mean = pairs_df["rouge_gap"].mean()

_, p_emcom = stats.wilcoxon(pairs_df["emcom_gap_a"].values, alternative="greater")
_, p_rouge = stats.wilcoxon(pairs_df["rouge_gap_a"].values, alternative="greater")

lines = [
    "=" * 70,
    "Caption-identical / Image-different Pair Analysis",
    f"N_pairs={n_pairs}, τ_cap={TAU_CAP:.3f}, τ_img={TAU_IMG:.3f}",
    "=" * 70,
    "",
    "=== EmCom-Diffusion (CLIP-image discrimination) ===",
    f"Discrimination accuracy (both a and b correct): {emcom_acc:.3f}  ({emcom_acc*100:.1f}%)",
    f"Mean gap CLIP(gen_a,x_a) - CLIP(gen_a,x_b): {emcom_gap_mean:.4f}",
    f"Wilcoxon p (gap > 0): {p_emcom:.2e}",
    "",
    "=== Translation (ROUGE-L discrimination) ===",
    f"Discrimination accuracy (both a and b correct): {rouge_acc:.3f}  ({rouge_acc*100:.1f}%)",
    f"Mean gap ROUGE-L(trans_a,ref_a) - ROUGE-L(trans_a,ref_b): {rouge_gap_mean:.4f}",
    f"Wilcoxon p (gap > 0): {p_rouge:.2e}",
    "",
    "=== Ratio ===",
    f"EmCom gap / ROUGE gap: {emcom_gap_mean / (abs(rouge_gap_mean) + 1e-6):.1f}x",
    f"EmCom accuracy / ROUGE accuracy: {emcom_acc / (rouge_acc + 1e-6):.2f}x",
]
summary = "\n".join(lines)
print(summary)

out_txt = OUT / "caption_id_image_diff_summary.txt"
out_txt.write_text(summary)

pairs_df.to_csv(OUT / "t16_caption_id_image_diff_pairs.csv", index=False)
print(f"Saved: t16 ({len(pairs_df)} pairs)")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

col_emcom = "#EE854A"
col_rouge = "#4878D0"

# Panel A: Gap distributions
ax = axes[0]
gap_emcom = pairs_df["emcom_gap_a"].values
gap_rouge = pairs_df["rouge_gap_a"].values
bins = np.linspace(-0.25, 0.35, 40)
ax.hist(gap_rouge, bins=bins, alpha=0.6, color=col_rouge, label="ROUGE-L")
ax.hist(gap_emcom, bins=bins, alpha=0.6, color=col_emcom, label="CLIP-image\n(EmCom-Diff)")
ax.axvline(0, color="k", lw=1.2, ls="--")
ax.axvline(gap_emcom.mean(), color=col_emcom, lw=2.0, ls="-",
           label=f"EmCom mean={gap_emcom.mean():.3f}")
ax.axvline(gap_rouge.mean(), color=col_rouge, lw=2.0, ls="-",
           label=f"ROUGE mean={gap_rouge.mean():.3f}")
ax.set_xlabel("Discrimination gap\n(correct image score − wrong image score)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("(a) Score gap distribution\n(caption-similar, image-different pairs)", fontsize=10)
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# Panel B: Discrimination accuracy bar
ax = axes[1]
accs = [rouge_acc, emcom_acc]
cols = [col_rouge, col_emcom]
labels = ["ROUGE-L\n(Translation)", "CLIP-image\n(EmCom-Diff)"]
bars = ax.bar(labels, accs, color=cols, alpha=0.85, width=0.5)
ax.axhline(0.5, color="k", lw=1.2, ls="--", alpha=0.6)
ax.text(1.55, 0.5 + 0.01, "chance\n(50%)", fontsize=8, color="k", va="bottom")
for bar, acc, p in zip(bars, [p_rouge, p_emcom], [p_rouge, p_emcom]):
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
            f"{acc:.2f}\n{star}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Pair discrimination accuracy\n(both images correctly identified)", fontsize=10)
ax.set_title(f"(b) Discrimination accuracy\n(n={n_pairs} pairs, τ_cap={TAU_CAP:.2f}, τ_img={TAU_IMG:.2f})", fontsize=10)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Caption-identical / Image-different: Can the method tell the images apart?",
             fontsize=11, y=1.01)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_exp2a_v3_discrimination.{ext}",
                dpi=180, bbox_inches="tight", facecolor="white")
print("Saved fig_exp2a_v3_discrimination.png/pdf")
