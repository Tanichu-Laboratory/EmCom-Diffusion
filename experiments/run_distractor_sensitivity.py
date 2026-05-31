"""
Distractor sensitivity experiment using the actual trained Listener (checkpoint_29.pth).

Evaluation:
  - Sender output: EC tokens from ec_corpus_blip_ref_ep29.json (pre-computed)
  - Listener score: text_feat(tokens) @ image_feat(GT_image)
    where text_feat = BERT_receiver + text_proj (embed_dim=256, L2-normalized)
    and   image_feat = DINOv2_CLS + vision_proj (embed_dim=256, L2-normalized)
  - Correct if Listener picks GT image_i over all distractors

Distractor selection:
  (A) Random distractors, varying N
  (B) Hard (CLIP-nearest GT images) vs Random, varying N-way
"""

import argparse
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ec_game.models.emcom import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       required=True, help="checkpoint_29.pth path")
parser.add_argument("--corpus",     required=True, help="EC corpus JSON")
parser.add_argument("--cache_dir",  required=True, help="Dir with cache_gt_clip_vitl14.npy etc.")
parser.add_argument("--output_dir", required=True, help="Dir to write outputs and cache files")
_args = parser.parse_args()

CKPT_PATH = Path(_args.ckpt)
CORPUS    = Path(_args.corpus)
OUT_V2    = Path(_args.cache_dir)
OUT       = Path(_args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── モデルロード ────────────────────────────────────────────────────────────
print("Loading model...")
ckpt  = torch.load(CKPT_PATH, map_location=device)
cfg   = ckpt["config"]
model = build_model(cfg).to(device).eval()
model.load_state_dict(ckpt["model"])
print(f"  Loaded from {CKPT_PATH}")

# ── EC コーパス（val 5000件）────────────────────────────────────────────────
print("Loading EC corpus...")
with open(CORPUS) as f:
    corpus = json.load(f)
val_entries = corpus["val"]
N_TOTAL = len(val_entries)
print(f"  val entries: {N_TOTAL}")

# トークン行列 [N, K]
tokens_all = torch.tensor(
    [e["ec_tokens"] for e in val_entries], dtype=torch.long
)  # (5000, 8)

# ── image_feat キャッシュ or 計算 ──────────────────────────────────────────
CACHE_IMG = OUT / "cache_listener_image_feat.npy"
if CACHE_IMG.exists():
    print(f"Loading image_feat cache: {CACHE_IMG}")
    image_feats = torch.from_numpy(np.load(CACHE_IMG)).to(device)
else:
    print("Computing image_feat for all val images...")
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    BATCH = 64
    feats = []
    batch = []
    for i, e in enumerate(val_entries):
        img = Image.open(e["image_path"]).convert("RGB")
        batch.append(transform(img))
        if len(batch) == BATCH or i == N_TOTAL - 1:
            imgs = torch.stack(batch).to(device)
            with torch.no_grad():
                embeds = model.visual_encoder(imgs)
                feat   = F.normalize(model.vision_proj(embeds[:, 0, :]), dim=-1)
            feats.append(feat.cpu())
            batch = []
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{N_TOTAL}")
    image_feats = torch.cat(feats, dim=0)  # (5000, 256)
    np.save(CACHE_IMG, image_feats.numpy())
    print(f"  Saved: {CACHE_IMG}")
    image_feats = image_feats.to(device)

print(f"image_feats: {image_feats.shape}")

# ── text_feat キャッシュ or 計算 ───────────────────────────────────────────
CACHE_TXT = OUT / "cache_listener_text_feat.npy"
if CACHE_TXT.exists():
    print(f"Loading text_feat cache: {CACHE_TXT}")
    text_feats = torch.from_numpy(np.load(CACHE_TXT)).to(device)
else:
    print("Computing text_feat for all val token sequences...")
    BATCH = 256
    feats = []
    for i in range(0, N_TOTAL, BATCH):
        toks = tokens_all[i:i+BATCH].to(device)
        V    = model.vocab_size
        soft = F.one_hot(toks, V).float()  # [B, K, V]
        with torch.no_grad():
            feat, _ = model._encode_message(soft)   # [B, 256], L2-normalized
        feats.append(feat.cpu())
        if (i + BATCH) % 500 < BATCH:
            print(f"  {min(i+BATCH, N_TOTAL)}/{N_TOTAL}")
    text_feats = torch.cat(feats, dim=0)  # (5000, 256)
    np.save(CACHE_TXT, text_feats.numpy())
    print(f"  Saved: {CACHE_TXT}")
    text_feats = text_feats.to(device)

print(f"text_feats: {text_feats.shape}")

# ── CLIP GT-GT 類似度（hard distractor 選択用）────────────────────────────
print("Loading CLIP GT embeddings for hard distractor selection...")
gt_clip = np.load(OUT_V2 / "cache_gt_clip_vitl14.npy").astype(np.float32)
gt_clip /= (np.linalg.norm(gt_clip, axis=1, keepdims=True) + 1e-8)
clip_sim_mat = torch.from_numpy(gt_clip @ gt_clip.T).to(device)  # (5000,5000)
print(f"clip_sim_mat: {clip_sim_mat.shape}")

# ── 評価関数 ────────────────────────────────────────────────────────────────
RNG      = np.random.default_rng(42)
N_TRIALS = 2000

# ── 全類似度行列を先に計算（Part B 5000-way 最適化・Part C 共用）──────────
print("Precomputing full similarity matrix (5000×5000)...")
sim_all = text_feats @ image_feats.T    # (5000, 5000)
print(f"  sim_all: {sim_all.shape}")

def accuracy_random_n(n_distractors, n_trials=N_TRIALS):
    """ランダム n_distractors 枚 + 正解 の (n+1)-way 正解率"""
    if n_distractors == N_TOTAL - 1:
        # 5000-way: sim_all を直接利用
        idxs = RNG.choice(N_TOTAL, n_trials, replace=False)
        correct = sum(int(sim_all[i].argmax().item() == i) for i in idxs)
        return correct / n_trials
    idxs = RNG.choice(N_TOTAL, n_trials, replace=False)
    correct = 0
    for i in idxs:
        cands = RNG.choice(
            [j for j in range(N_TOTAL) if j != i],
            n_distractors, replace=False
        )
        pool = np.concatenate([[i], cands])
        pool_t = torch.from_numpy(pool).long().to(device)
        sims = text_feats[i] @ image_feats[pool_t].T  # (n+1,)
        correct += int(sims.argmax().item() == 0)
    return correct / n_trials

def accuracy_hard_n(n_distractors, n_trials=N_TRIALS):
    """CLIP-nearest n_distractors 枚 + 正解 の (n+1)-way 正解率"""
    idxs = RNG.choice(N_TOTAL, n_trials, replace=False)
    correct = 0
    if n_distractors == N_TOTAL - 1:
        # 5000-way hard: 全画像が distractor → sim_all を直接利用
        for i in idxs:
            correct += int(sim_all[i].argmax().item() == i)
        return correct / n_trials
    for i in idxs:
        row = clip_sim_mat[i].clone()
        row[i] = -torch.inf
        hard_cands = row.topk(n_distractors).indices.cpu().numpy()
        pool = np.concatenate([[i], hard_cands])
        pool_t = torch.from_numpy(pool).long().to(device)
        sims = text_feats[i] @ image_feats[pool_t].T
        correct += int(sims.argmax().item() == 0)
    return correct / n_trials

# ── (A) N-way 正解率（ランダム）──────────────────────────────────────────
print("\n(A) Varying N-way (random distractors)...")
N_labels = [2, 4, 8, 16, 32, 100, 500, 1000, 5000]
accs_rand = []
for n in N_labels:
    a = accuracy_random_n(n - 1)
    accs_rand.append(a)
    print(f"  {n:5d}-way  acc={a:.3f}")

# ── (B) Hard vs Random（N-way 固定）────────────────────────────────────────
print("\n(B) Hard vs Random distractors...")
Ns_way = [2, 4, 8, 16, 32, 100, 500, 1000, 5000]
accs_hard  = []
accs_rand2 = []
for nw in Ns_way:
    ah = accuracy_hard_n(nw - 1)
    ar = accuracy_random_n(nw - 1)
    accs_hard.append(ah)
    accs_rand2.append(ar)
    print(f"  {nw:4d}-way  hard={ah:.3f}  random={ar:.3f}  gap={ar-ah:.3f}")

# ── (C) Full R@1 (5000-way) ────────────────────────────────────────────────
print("\n(C) Full R@1 (5000-way)...")
ranks = (sim_all > sim_all[torch.arange(N_TOTAL), torch.arange(N_TOTAL)].unsqueeze(1)).sum(1)
r1 = (ranks == 0).float().mean().item()
print(f"  R@1 (5000-way) = {r1:.3f}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
fig.suptitle("Listener Accuracy vs Distractor Set", fontsize=13, fontweight="bold")

# left: N-way
ax = axes[0]
chance = [1 / n for n in N_labels]
ax.semilogx(N_labels, [a * 100 for a in accs_rand],
            "o-", color="#2196F3", lw=2, ms=7, label="Random distractors")
ax.semilogx(N_labels, [c * 100 for c in chance],
            "--", color="#9e9e9e", lw=1.5, label="Chance (1/N)")
ax.axvline(x=5000, color="#e53935", ls=":", lw=1.5, alpha=0.7, label="N=5000")
ax.set_xlabel("N-way", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("(A) N-way accuracy\n(random distractors)", fontsize=10)
ax.set_xticks(N_labels)
ax.set_xticklabels([str(n) for n in N_labels], rotation=30, fontsize=8)
ax.legend(fontsize=9); ax.set_ylim(0, 105)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for n, a in zip(N_labels, accs_rand):
    ax.annotate(f"{a*100:.0f}%", (n, a*100), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#1565c0")

# right: hard vs random
ax2 = axes[1]
x = np.arange(len(Ns_way))
w = 0.32
b1 = ax2.bar(x - w/2, [a * 100 for a in accs_rand2], w,
             label="Random distractors", color="#2196F3", edgecolor="white")
b2 = ax2.bar(x + w/2, [a * 100 for a in accs_hard], w,
             label="Hard distractors\n(CLIP-nearest)", color="#e53935", edgecolor="white")
ax2.plot(x, [100/n for n in Ns_way], "s--", color="#9e9e9e",
         lw=1.2, ms=5, label="Chance", zorder=5)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{n}-way" for n in Ns_way], fontsize=8, rotation=30, ha="right")
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_title("(B) Random vs. Hard Distractors\n(CLIP-nearest hard)", fontsize=10)
ax2.legend(fontsize=9); ax2.set_ylim(0, 105)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                 f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_png = OUT_V2 / "fig_distractor_sensitivity.png"
out_pdf = OUT_V2 / "fig_distractor_sensitivity.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\nSaved {out_png}")

# ── 数値サマリー ───────────────────────────────────────────────────────────
print("\n=== Summary (A) ===")
print(f"{'N-way':>8}  {'Random':>8}  {'Chance':>8}")
for n, a, c in zip(N_labels, accs_rand, chance):
    print(f"{n:>8d}  {a*100:>7.1f}%  {c*100:>7.1f}%")

print(f"\n=== Summary (B) ===")
print(f"{'N-way':>8}  {'Random':>8}  {'Hard':>8}  {'Gap':>8}")
for n, ar, ah in zip(Ns_way, accs_rand2, accs_hard):
    print(f"{n:>8d}  {ar*100:>7.1f}%  {ah*100:>7.1f}%  {(ar-ah)*100:>+7.1f}%")

# ── CSV 保存 ───────────────────────────────────────────────────────────────
import csv
csv_a = OUT_V2 / "distractor_sensitivity_A.csv"
with open(csv_a, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_way", "random_pct", "chance_pct"])
    for n, a, c in zip(N_labels, accs_rand, chance):
        w.writerow([n, round(a * 100, 1), round(c * 100, 1)])
print(f"Saved {csv_a}")

csv_b = OUT_V2 / "distractor_sensitivity_B.csv"
with open(csv_b, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_way", "random_pct", "hard_pct", "gap_pct"])
    for n, ar, ah in zip(Ns_way, accs_rand2, accs_hard):
        w.writerow([n, round(ar * 100, 1), round(ah * 100, 1), round((ar - ah) * 100, 1)])
print(f"Saved {csv_b}")
