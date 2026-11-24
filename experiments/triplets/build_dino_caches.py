"""
Caption-identical / Image-different pair analysis (v2: DINO for image similarity)

Pair selection:
  CLIP-text(caps_a, caps_b) > τ₁   [captions are similar]
  DINO(x_a, x_b)            < τ₂   [images are different by DINO]

Evaluation (independent of pair selection):
  EmCom-Diffusion: CLIP(gen_a, x_a) vs CLIP(gen_a, x_b)
  Translation:     ROUGE-L(trans_a, ref_a) vs ROUGE-L(trans_a, ref_b)

Outputs:
  cache_dino_vitb14.npy               — DINOv2 GT embeddings
  t17_dino_pairs.csv                  — qualifying pairs with all scores
  dino_pairs_summary.txt              — key numbers
  fig_exp2a_v3_dino_discrimination.png/pdf
  dino_pairs_gallery/                 — qualitative HTML gallery
"""

import json, sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from rouge_score import rouge_scorer
from scipy import stats
import shutil

BASE = Path("/workspace/project/omoto/emlan-image-generate")
OUT  = BASE / "outputs/experiments_v2"
IMGS = BASE / "outputs/generated_images/ep29"
VAL  = Path("/workspace/project/datas/dataset/coco2017/val2017")
ANN_CAP = Path("/workspace/project/datas/dataset/coco2017/annotations/captions_val2017.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 64
print(f"Device: {DEVICE}")

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading CLIP ViT-L/14 ...")
clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
clip_model.eval()

print("Loading DINOv2 ViT-B/14 ...")
dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
dino_model = dino_model.to(DEVICE).eval()

DINO_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT / "t11_translation_per_sample_scores.csv")
with open(ANN_CAP) as f:
    coco_cap = json.load(f)
img_to_refs = defaultdict(list)
for ann in coco_cap["annotations"]:
    img_to_refs[ann["image_id"]].append(ann["caption"])

N = len(df)
print(f"Eval set size: N={N}")

def gt_path(image_id):
    return VAL / f"{int(image_id):012d}.jpg"

def gen_path(idx, image_id):
    p = IMGS / f"{int(idx):04d}_imgid{int(image_id)}.png"
    if not p.exists():
        ms = list(IMGS.glob(f"*_imgid{int(image_id)}.png"))
        if ms:
            return ms[0]
    return p

# ── Embedding helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def embed_clip_images(paths, desc="CLIP images"):
    feats = []
    for i in tqdm(range(0, len(paths), BATCH), desc=desc):
        imgs = []
        for p in paths[i:i+BATCH]:
            try:
                imgs.append(clip_preprocess(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        t = torch.stack(imgs).to(DEVICE)
        f = clip_model.encode_image(t).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.vstack(feats)

@torch.no_grad()
def embed_dino_images(paths, desc="DINO images"):
    feats = []
    for i in tqdm(range(0, len(paths), BATCH), desc=desc):
        imgs = []
        for p in paths[i:i+BATCH]:
            try:
                imgs.append(DINO_TRANSFORM(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        t = torch.stack(imgs).to(DEVICE)
        f = dino_model(t).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.vstack(feats)

@torch.no_grad()
def embed_clip_texts(texts_list, desc="CLIP texts"):
    feats = []
    for i in tqdm(range(0, len(texts_list), BATCH), desc=desc):
        batch = texts_list[i:i+BATCH]
        img_feats = []
        for caps in batch:
            if not caps:
                img_feats.append(np.zeros(768))
                continue
            tokens = clip.tokenize(caps[:5], truncate=True).to(DEVICE)
            f = clip_model.encode_text(tokens).float()
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            f = f / (f.norm() + 1e-8)
            img_feats.append(f.cpu().numpy())
        feats.extend(img_feats)
    return np.array(feats)

# ── Compute / load embeddings ──────────────────────────────────────────────────
cache_dino_gt  = OUT / "cache_dino_vitb14_gt.npy"
cache_clip_gt  = OUT / "cache_gt_clip_vitl14.npy"
cache_clip_gen = OUT / "cache_gen_clip_vitl14.npy"
cache_clip_cap = OUT / "cache_cap_clip_vitl14.npy"

gt_paths  = [gt_path(iid)  for iid in df["image_id"]]
gen_paths = [gen_path(row["idx"], row["image_id"]) for _, row in df.iterrows()]
all_caps  = [img_to_refs.get(int(iid), [""]) for iid in df["image_id"]]

if cache_dino_gt.exists():
    print("Loading cached DINO GT embeddings ...")
    dino_gt = np.load(cache_dino_gt)
else:
    print("Computing DINO GT embeddings ...")
    dino_gt = embed_dino_images(gt_paths, "DINO GT")
    np.save(cache_dino_gt, dino_gt)

if cache_clip_gt.exists():
    print("Loading cached CLIP GT embeddings ...")
    clip_gt = np.load(cache_clip_gt)
else:
    print("Computing CLIP GT embeddings ...")
    clip_gt = embed_clip_images(gt_paths, "CLIP GT")
    np.save(cache_clip_gt, clip_gt)

if cache_clip_gen.exists():
    print("Loading cached CLIP gen embeddings ...")
    clip_gen = np.load(cache_clip_gen)
else:
    print("Computing CLIP gen embeddings ...")
    clip_gen = embed_clip_images(gen_paths, "CLIP gen")
    np.save(cache_clip_gen, clip_gen)

if cache_clip_cap.exists():
    print("Loading cached CLIP caption embeddings ...")
    clip_cap = np.load(cache_clip_cap)
else:
    print("Computing CLIP caption embeddings ...")
    clip_cap = embed_clip_texts(all_caps, "CLIP captions")
    np.save(cache_clip_cap, clip_cap)

print(f"Embeddings: DINO GT {dino_gt.shape}, CLIP GT {clip_gt.shape}, "
      f"CLIP gen {clip_gen.shape}, CLIP cap {clip_cap.shape}")

# ── Calibrate thresholds from random sample ────────────────────────────────────
print("Calibrating thresholds ...")
rng = np.random.default_rng(42)
sidx = rng.choice(N, size=1000, replace=False)
dino_gt_t  = torch.from_numpy(dino_gt).to(DEVICE)
clip_cap_t = torch.from_numpy(clip_cap).to(DEVICE)
clip_gt_t  = torch.from_numpy(clip_gt).to(DEVICE)
clip_gen_t = torch.from_numpy(clip_gen).to(DEVICE)

with torch.no_grad():
    dino_sub = dino_gt_t[sidx]
    cap_sub  = clip_cap_t[sidx]
    dino_sim_sub = (dino_sub @ dino_gt_t[sidx].T).cpu().numpy()
    cap_sim_sub  = (cap_sub  @ clip_cap_t[sidx].T).cpu().numpy()

mask = ~np.eye(len(sidx), dtype=bool)
dino_vals = dino_sim_sub[mask]
cap_vals  = cap_sim_sub[mask]

print(f"DINO pairwise: mean={dino_vals.mean():.3f}  "
      f"p20={np.percentile(dino_vals,20):.3f}  p80={np.percentile(dino_vals,80):.3f}")
print(f"CLIP-cap pairwise: mean={cap_vals.mean():.3f}  "
      f"p20={np.percentile(cap_vals,20):.3f}  p80={np.percentile(cap_vals,80):.3f}")

TAU_CAP  = float(np.percentile(cap_vals,  80))   # top-20% caption similarity
TAU_DINO = float(np.percentile(dino_vals, 20))   # bottom-20% DINO image similarity
print(f"Thresholds: τ_cap={TAU_CAP:.3f}, τ_dino={TAU_DINO:.3f}")

# ── Scan all pairs ─────────────────────────────────────────────────────────────
print("Scanning pairs ...")
CHUNK     = 500
MAX_PAIRS = 50000
pairs     = []

with torch.no_grad():
    for i in tqdm(range(0, N, CHUNK)):
        if len(pairs) >= MAX_PAIRS:
            break
        i_end  = min(i + CHUNK, N)
        cap_i  = clip_cap_t[i:i_end]
        dino_i = dino_gt_t[i:i_end]

        for j in range(i + 1, N, CHUNK):
            if len(pairs) >= MAX_PAIRS:
                break
            j_end  = min(j + CHUNK, N)
            cap_j  = clip_cap_t[j:j_end]
            dino_j = dino_gt_t[j:j_end]

            cap_sim  = (cap_i  @ cap_j.T).cpu().numpy()
            dino_sim = (dino_i @ dino_j.T).cpu().numpy()

            qi_arr, qj_arr = np.where(
                (cap_sim > TAU_CAP) & (dino_sim < TAU_DINO)
            )
            for qi, qj in zip(qi_arr, qj_arr):
                a, b = i + qi, j + qj
                pairs.append({
                    "a": int(a), "b": int(b),
                    "image_id_a": int(df["image_id"].iloc[a]),
                    "image_id_b": int(df["image_id"].iloc[b]),
                    "cap_sim":    float(cap_sim[qi, qj]),
                    "dino_sim":   float(dino_sim[qi, qj]),
                    # CLIP image sim for comparison
                    "clip_img_sim": float(clip_gt_t[a] @ clip_gt_t[b]),
                    # EmCom discrimination
                    "clip_gen_a_gt_a": float(clip_gen_t[a] @ clip_gt_t[a]),
                    "clip_gen_a_gt_b": float(clip_gen_t[a] @ clip_gt_t[b]),
                    "clip_gen_b_gt_b": float(clip_gen_t[b] @ clip_gt_t[b]),
                    "clip_gen_b_gt_a": float(clip_gen_t[b] @ clip_gt_t[a]),
                    "pred_a":   str(df["predicted_caption"].iloc[a]),
                    "pred_b":   str(df["predicted_caption"].iloc[b]),
                    "ref_a0":   str(df["reference_caption_0"].iloc[a]),
                    "ref_b0":   str(df["reference_caption_0"].iloc[b]),
                    "rouge_l_a": float(df["rouge_l"].iloc[a]),
                    "rouge_l_b": float(df["rouge_l"].iloc[b]),
                })
                if len(pairs) >= MAX_PAIRS:
                    break

print(f"Qualifying pairs: {len(pairs)}")
if len(pairs) == 0:
    sys.exit("No pairs found — adjust thresholds")

pairs_df = pd.DataFrame(pairs)

# ── ROUGE-L cross-evaluation ───────────────────────────────────────────────────
print("Computing ROUGE-L cross-evaluation ...")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rouge_aa, rouge_ab, rouge_ba, rouge_bb = [], [], [], []
for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="ROUGE-L"):
    a, b    = int(row["a"]), int(row["b"])
    pred_a  = str(df["predicted_caption"].iloc[a])
    pred_b  = str(df["predicted_caption"].iloc[b])
    refs_a  = img_to_refs.get(int(df["image_id"].iloc[a]), [""])
    refs_b  = img_to_refs.get(int(df["image_id"].iloc[b]), [""])
    ref_a   = refs_a[0] if refs_a else ""
    ref_b   = refs_b[0] if refs_b else ""
    rouge_aa.append(scorer.score(ref_a, pred_a)["rougeL"].fmeasure)
    rouge_ab.append(scorer.score(ref_b, pred_a)["rougeL"].fmeasure)
    rouge_ba.append(scorer.score(ref_a, pred_b)["rougeL"].fmeasure)
    rouge_bb.append(scorer.score(ref_b, pred_b)["rougeL"].fmeasure)

pairs_df["rouge_aa"] = rouge_aa
pairs_df["rouge_ab"] = rouge_ab
pairs_df["rouge_ba"] = rouge_ba
pairs_df["rouge_bb"] = rouge_bb

# ── Discrimination metrics ─────────────────────────────────────────────────────
pairs_df["emcom_correct_a"] = pairs_df["clip_gen_a_gt_a"] > pairs_df["clip_gen_a_gt_b"]
pairs_df["emcom_correct_b"] = pairs_df["clip_gen_b_gt_b"] > pairs_df["clip_gen_b_gt_a"]
pairs_df["emcom_gap_a"]     = pairs_df["clip_gen_a_gt_a"] - pairs_df["clip_gen_a_gt_b"]
pairs_df["emcom_gap_b"]     = pairs_df["clip_gen_b_gt_b"] - pairs_df["clip_gen_b_gt_a"]

pairs_df["rouge_correct_a"] = pairs_df["rouge_aa"] > pairs_df["rouge_ab"]
pairs_df["rouge_correct_b"] = pairs_df["rouge_bb"] > pairs_df["rouge_ba"]
pairs_df["rouge_gap_a"]     = pairs_df["rouge_aa"] - pairs_df["rouge_ab"]
pairs_df["rouge_gap_b"]     = pairs_df["rouge_bb"] - pairs_df["rouge_ba"]

emcom_ind = (pairs_df["emcom_correct_a"].mean() + pairs_df["emcom_correct_b"].mean()) / 2
rouge_ind = (pairs_df["rouge_correct_a"].mean() + pairs_df["rouge_correct_b"].mean()) / 2
emcom_gap = pairs_df["emcom_gap_a"].mean()
rouge_gap = pairs_df["rouge_gap_a"].mean()

_, p_emcom = stats.wilcoxon(pairs_df["emcom_gap_a"].values, alternative="greater")
_, p_rouge = stats.wilcoxon(pairs_df["rouge_gap_a"].values, alternative="greater")
_, p_diff  = stats.mannwhitneyu(pairs_df["emcom_gap_a"], pairs_df["rouge_gap_a"],
                                 alternative="greater")

# ── Summary ────────────────────────────────────────────────────────────────────
lines = [
    "=" * 70,
    "Caption-identical / Image-different (DINO image similarity)",
    f"N_pairs={len(pairs_df)}, τ_cap={TAU_CAP:.3f}, τ_dino={TAU_DINO:.3f}",
    "=" * 70,
    "",
    "=== EmCom-Diffusion (CLIP-image discrimination) ===",
    f"Individual accuracy: {emcom_ind:.3f} ({emcom_ind*100:.1f}%)",
    f"Mean gap (CLIP):     {emcom_gap:.4f}",
    f"p (gap > 0):         {p_emcom:.2e}",
    "",
    "=== Translation (ROUGE-L discrimination) ===",
    f"Individual accuracy: {rouge_ind:.3f} ({rouge_ind*100:.1f}%)",
    f"Mean gap (ROUGE-L):  {rouge_gap:.4f}",
    f"p (gap > 0):         {p_rouge:.2e}",
    "",
    "=== Comparison ===",
    f"Accuracy Δ:          {emcom_ind - rouge_ind:.3f} ({(emcom_ind-rouge_ind)*100:.1f}pp)",
    f"Gap ratio:           {emcom_gap / (abs(rouge_gap)+1e-9):.2f}x",
    f"MWU EmCom > ROUGE:   p={p_diff:.2e}",
]
summary = "\n".join(lines)
print(summary)
(OUT / "dino_pairs_summary.txt").write_text(summary)
pairs_df.to_csv(OUT / "t17_dino_pairs.csv", index=False)
print(f"Saved t17_dino_pairs.csv ({len(pairs_df)} pairs)")

# ── Figure ─────────────────────────────────────────────────────────────────────
col_e = "#EE854A"
col_r = "#4878D0"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: gap distributions
ax = axes[0]
eg = pairs_df["emcom_gap_a"].values
rg = pairs_df["rouge_gap_a"].values
bins = np.linspace(-0.30, 0.70, 50)
ax.hist(rg, bins=bins, alpha=0.6, color=col_r, label="ROUGE-L", density=True)
ax.hist(eg, bins=bins, alpha=0.6, color=col_e, label="CLIP-image (EmCom-Diff)", density=True)
ax.axvline(0,        color="k",  lw=1.2, ls="--", alpha=0.7)
ax.axvline(eg.mean(), color=col_e, lw=2.2)
ax.axvline(rg.mean(), color=col_r, lw=2.2)
ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 5.0
ax.text(eg.mean()+0.01, ylim_top*0.85,
        f"EmCom\n{eg.mean():.3f}", fontsize=8, color=col_e, va="top")
ax.text(rg.mean()+0.005, ylim_top*0.55,
        f"ROUGE\n{rg.mean():.3f}", fontsize=8, color=col_r, va="top")
ax.set_xlabel("Discrimination gap\n(score on own image − score on other image)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"(a) Gap distribution\n($n$={len(pairs_df):,} pairs, DINO image-different)", fontsize=9.5)
ax.legend(fontsize=8.5)
ax.spines[["top","right"]].set_visible(False)

# Panel B: individual accuracy
ax = axes[1]
accs = [rouge_ind, emcom_ind]
ps   = [p_rouge, p_emcom]
cols = [col_r, col_e]
labels = ["ROUGE-L\n(Translation)", "CLIP-image\n(EmCom-Diff)"]
bars = ax.bar(labels, accs, color=cols, alpha=0.85, width=0.5)
ax.axhline(0.5, color="k", lw=1.4, ls="--", alpha=0.7)
ax.text(1.57, 0.502, "chance (50%)", fontsize=8.5, color="k", va="bottom")
for bar, acc, p in zip(bars, accs, ps):
    star = "***" if p < 0.001 else ("**" if p < 0.01 else "*" if p < 0.05 else "ns")
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.008,
            f"{acc:.1%}\n({star})", ha="center", va="bottom", fontsize=11, fontweight="bold")
# Delta annotation
ax.annotate("", xy=(1, rouge_ind), xytext=(1, emcom_ind),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
ax.text(1.28, (emcom_ind+rouge_ind)/2,
        f"Δ={emcom_ind-rouge_ind:.1%}", fontsize=9.5, color="gray", va="center")
ax.set_ylim(0.45, 0.90)
ax.set_ylabel("Individual discrimination accuracy\nP(own image scored higher)", fontsize=10)
ax.set_title(f"(b) Discrimination accuracy\n"
             f"(τ_cap={TAU_CAP:.2f}, τ_dino={TAU_DINO:.2f}, image-diff by DINO)", fontsize=9.5)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Caption-identical / Image-different: can the evaluation method tell images apart?\n"
             "(Pair selection: CLIP-text caption similarity ↑, DINOv2 image similarity ↓)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_exp2a_v3_dino.{ext}", dpi=180,
                bbox_inches="tight", facecolor="white")
print("Saved fig_exp2a_v3_dino.png/pdf")

# ── Qualitative gallery (HTML) ─────────────────────────────────────────────────
print("Building qualitative gallery ...")
GALLERY = OUT / "dino_pairs_gallery"
GALLERY.mkdir(exist_ok=True)
IMG_DIR  = GALLERY / "images"
IMG_DIR.mkdir(exist_ok=True)

# Select top 30 most discriminating pairs:
# high cap_sim, low dino_sim, high emcom_gap_a, low |rouge_gap_a|
pairs_df["quality_score"] = (
    pairs_df["cap_sim"] * 2
    - pairs_df["dino_sim"] * 2
    + pairs_df["emcom_gap_a"]
    - pairs_df["rouge_gap_a"].abs()
)
top_pairs = pairs_df.nlargest(30, "quality_score").reset_index(drop=True)

def save_thumbnail(src_path, dst_path, size=(224, 224)):
    try:
        img = Image.open(src_path).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        img.save(dst_path)
        return True
    except Exception:
        return False

rows_html = []
for rank, (_, row) in enumerate(top_pairs.iterrows()):
    a, b = int(row["a"]), int(row["b"])
    iid_a, iid_b = int(df["image_id"].iloc[a]), int(df["image_id"].iloc[b])
    idx_a, idx_b = int(df["idx"].iloc[a]),      int(df["idx"].iloc[b])

    fn_gt_a  = f"{rank:02d}_gt_a_{iid_a}.jpg"
    fn_gt_b  = f"{rank:02d}_gt_b_{iid_b}.jpg"
    fn_gen_a = f"{rank:02d}_gen_a_{iid_a}.jpg"
    fn_gen_b = f"{rank:02d}_gen_b_{iid_b}.jpg"

    save_thumbnail(gt_path(iid_a),          IMG_DIR / fn_gt_a)
    save_thumbnail(gt_path(iid_b),          IMG_DIR / fn_gt_b)
    save_thumbnail(gen_path(idx_a, iid_a),  IMG_DIR / fn_gen_a)
    save_thumbnail(gen_path(idx_b, iid_b),  IMG_DIR / fn_gen_b)

    refs_a = img_to_refs.get(iid_a, [""])[:2]
    refs_b = img_to_refs.get(iid_b, [""])[:2]

    def hl(s, q):
        return f'<span style="background:#ffe0b2">{s}</span>' if q else s

    emcom_ok_a = row["emcom_correct_a"]
    rouge_ok_a = row["rouge_correct_a"]

    rows_html.append(f"""
<tr>
  <td style="text-align:center;font-weight:bold;color:#555">#{rank+1}</td>

  <!-- Image A -->
  <td>
    <img src="images/{fn_gt_a}"  width="150"><br>
    <small><b>GT A</b> (img {iid_a})</small><br>
    <small style="color:#2a6e2a">{"<br>".join(refs_a)}</small>
  </td>
  <td>
    <img src="images/{fn_gen_a}" width="150"><br>
    <small><b>Generated A</b></small><br>
    <small style="color:#666">{str(row["pred_a"])[:80]}</small>
  </td>

  <!-- Image B -->
  <td>
    <img src="images/{fn_gt_b}"  width="150"><br>
    <small><b>GT B</b> (img {iid_b})</small><br>
    <small style="color:#2a6e2a">{"<br>".join(refs_b)}</small>
  </td>
  <td>
    <img src="images/{fn_gen_b}" width="150"><br>
    <small><b>Generated B</b></small><br>
    <small style="color:#666">{str(row["pred_b"])[:80]}</small>
  </td>

  <!-- Scores -->
  <td style="font-size:12px;line-height:1.8;min-width:190px">
    <b>Pair similarity</b><br>
    CLIP-cap sim: <b>{row["cap_sim"]:.3f}</b><br>
    DINO img sim: <b>{row["dino_sim"]:.3f}</b><br>
    <br>
    <b>EmCom discrimination</b><br>
    CLIP(gen_A, GT_A): <b>{row["clip_gen_a_gt_a"]:.3f}</b><br>
    CLIP(gen_A, GT_B): <b>{row["clip_gen_a_gt_b"]:.3f}</b><br>
    Gap: {hl(f"<b>+{row['emcom_gap_a']:.3f}</b>" if row["emcom_gap_a"]>0 else f"<b>{row['emcom_gap_a']:.3f}</b>", emcom_ok_a)}<br>
    Correct: {"✅" if emcom_ok_a else "❌"}<br>
    <br>
    <b>ROUGE-L discrimination</b><br>
    ROUGE(trans_A, ref_A): <b>{row["rouge_aa"]:.3f}</b><br>
    ROUGE(trans_A, ref_B): <b>{row["rouge_ab"]:.3f}</b><br>
    Gap: {hl(f"<b>+{row['rouge_gap_a']:.3f}</b>" if row["rouge_gap_a"]>0 else f"<b>{row['rouge_gap_a']:.3f}</b>", rouge_ok_a)}<br>
    Correct: {"✅" if rouge_ok_a else "❌"}
  </td>
</tr>
<tr><td colspan="6" style="border-bottom:2px solid #ddd"></td></tr>
""")

html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Caption-identical / Image-different Gallery (DINO pairs)</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; margin: 20px; }}
  h1   {{ font-size: 18px; }}
  table {{ border-collapse: collapse; }}
  td   {{ padding: 8px; vertical-align: top; border: 1px solid #eee; }}
  img  {{ border-radius: 4px; object-fit: cover; }}
</style>
</head><body>
<h1>Caption-identical / Image-different Pairs — Qualitative Gallery</h1>
<p>
  Pairs selected by: CLIP-text caption similarity &gt; {TAU_CAP:.3f} (top 20%)
  <b>AND</b> DINOv2 image similarity &lt; {TAU_DINO:.3f} (bottom 20%).<br>
  Total qualifying pairs: {len(pairs_df):,}. Showing top 30 by discrimination quality score.<br>
  <b>EmCom-Diffusion individual accuracy: {emcom_ind:.1%}</b> &nbsp;|&nbsp;
  <b>ROUGE-L individual accuracy: {rouge_ind:.1%}</b> &nbsp;|&nbsp;
  Chance: 50%
</p>
<p style="color:#888;font-size:11px">
  Green: reference captions. Orange bg = correct discrimination (own score &gt; cross score).
</p>
<table>
<tr style="background:#f5f5f5;font-weight:bold">
  <td>#</td>
  <td>GT Image A</td><td>Generated A</td>
  <td>GT Image B</td><td>Generated B</td>
  <td>Scores</td>
</tr>
{"".join(rows_html)}
</table>
</body></html>"""

(GALLERY / "index.html").write_text(html)
print(f"Gallery saved: {GALLERY}/index.html")
print(f"\nFinal numbers:")
print(f"  Pairs:            {len(pairs_df):,}")
print(f"  EmCom accuracy:   {emcom_ind:.1%}")
print(f"  ROUGE accuracy:   {rouge_ind:.1%}")
print(f"  Δ accuracy:       {(emcom_ind-rouge_ind)*100:.1f}pp")
print(f"  EmCom gap:        {emcom_gap:.4f}")
print(f"  ROUGE gap:        {rouge_gap:.4f}")
print(f"  Gap ratio:        {emcom_gap/(abs(rouge_gap)+1e-9):.2f}x")
