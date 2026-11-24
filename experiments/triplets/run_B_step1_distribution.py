"""
Plan B Step 1: 創発言語における「類似トークン・異なる画像」ペアの頻度確認

全5000ペアの EC token Levenshtein 距離 × DINO 画像距離の分布を可視化し、
「edit distance が小さい（≤2）かつ DINO 距離が大きい（≥0.9）」象限のペア数を報告する。

出力:
  experiments_v3/outputs/B_editdist_dino_scatter.png/pdf  ← 分布図
  experiments_v3/outputs/B_quadrant_counts.csv            ← 象限別ペア数
"""

import ast
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist as rf_cdist

# ── パス ──────────────────────────────────────────────────────────────────────
BASE = Path("/workspace/project/omoto/emlan-image-generate")
OUT  = BASE / "experiments_v3/outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE / "outputs/experiments_v2"

# ── データ読み込み ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_DIR / "t11_translation_per_sample_scores.csv")
N = len(df)  # 5000
print(f"  N = {N}")

# EC tokens → tuple of ints (rapidfuzz requires sequences)
seqs = [tuple(ast.literal_eval(x)) for x in df["ec_tokens"]]

# DINO embeddings (normalized)
dino_gt = np.load(DATA_DIR / "cache_dino_vitb14_gt.npy").astype(np.float32)
dino_gt /= (np.linalg.norm(dino_gt, axis=1, keepdims=True) + 1e-8)
print(f"  DINO shape: {dino_gt.shape}")

# ── Edit distance 行列 ─────────────────────────────────────────────────────────
print(f"\nComputing Levenshtein edit distance matrix ({N}×{N})...")
t0 = time.time()
edit_mat = rf_cdist(seqs, seqs, scorer=Levenshtein.distance, workers=-1)
t1 = time.time()
print(f"  Done in {t1-t0:.1f}s. dtype={edit_mat.dtype}, shape={edit_mat.shape}")

# ── DINO 距離行列（= 1 - cosine similarity）─────────────────────────────────
print("\nComputing DINO distance matrix...")
t0 = time.time()
dino_sim_mat = dino_gt @ dino_gt.T  # (N, N)
dino_dist_mat = 1.0 - dino_sim_mat
t1 = time.time()
print(f"  Done in {t1-t0:.1f}s.")

# ── 上三角のみ抽出（対角除く）─────────────────────────────────────────────────
print("\nExtracting upper triangle pairs...")
triu_idx = np.triu_indices(N, k=1)   # ~12.5M pairs
edit_vals  = edit_mat[triu_idx].astype(np.int8)    # 0-8
dino_vals  = dino_dist_mat[triu_idx].astype(np.float32)
total_pairs = len(edit_vals)
print(f"  Total pairs: {total_pairs:,}")

# ── 象限別カウント ─────────────────────────────────────────────────────────────
# 「類似トークン・異なる画像」= 左上象限（高DINO距離・低edit距離）
EDIT_THRESHOLDS  = [0, 1, 2, 3]
DINO_THRESHOLDS  = [0.5, 0.7, 0.8, 0.9]

rows = []
for ed_max in EDIT_THRESHOLDS:
    mask_edit = edit_vals <= ed_max
    n_edit = mask_edit.sum()
    for dino_min in DINO_THRESHOLDS:
        mask_both = mask_edit & (dino_vals >= dino_min)
        n_both = mask_both.sum()
        rows.append({
            "edit_dist_max": ed_max,
            "dino_dist_min": dino_min,
            "n_edit_similar": int(n_edit),
            "n_both": int(n_both),
            "pct_of_edit_similar": float(n_both / n_edit * 100) if n_edit > 0 else 0,
            "pct_of_total": float(n_both / total_pairs * 100),
        })

df_counts = pd.DataFrame(rows)
df_counts.to_csv(OUT / "B_quadrant_counts.csv", index=False)
print("\n=== 象限別ペア数 (上三角, 全対称ペア) ===")
print(df_counts.to_string(index=False))

# ── DINO 距離の分布 (edit_dist 別) ─────────────────────────────────────────────
print("\n=== DINO 距離の統計 (edit_dist 別) ===")
print(f"{'edit_dist':>10}  {'count':>10}  {'dino_mean':>10}  {'dino_p50':>10}  {'dino_p90':>10}")
for ed in range(9):
    mask = edit_vals == ed
    cnt = mask.sum()
    if cnt == 0:
        continue
    d = dino_vals[mask]
    print(f"{ed:>10d}  {cnt:>10,}  {d.mean():>10.4f}  {np.median(d):>10.4f}  {np.percentile(d,90):>10.4f}")

# ── 散布図（2D ヒストグラム）─────────────────────────────────────────────────
print("\nCreating scatter (2D histogram)...")

# edit_dist は離散値 0-8 → x 軸を jitter なしで bin 幅 1 に
# DINO 距離 は 0-1 連続値 → y 軸

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("EC Token Edit Distance vs DINO Image Distance\n(all 12.5M pairs, upper triangle)",
             fontsize=12, fontweight="bold")

# --- left: 2D density heatmap ---
ax = axes[0]
# bins: x = -0.5 to 8.5 (9 bins for edit 0-8), y = 0.0 to 1.0
H, xedges, yedges = np.histogram2d(
    edit_vals.astype(float), dino_vals,
    bins=[np.arange(-0.5, 9.5, 1.0), np.linspace(0, 1, 51)]
)
# log scale for color to handle large variation
im = ax.pcolormesh(xedges, yedges, np.log1p(H.T),
                   cmap="Blues", shading="flat")
plt.colorbar(im, ax=ax, label="log(1 + count)")

# 閾値ライン
ax.axvline(x=2.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="edit_dist ≤ 2")
ax.axhline(y=0.9, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="DINO dist ≥ 0.9")

# 象限注釈
ed2_dino09 = int(df_counts[(df_counts.edit_dist_max==2) & (df_counts.dino_dist_min==0.9)]["n_both"].iloc[0])
pct = float(df_counts[(df_counts.edit_dist_max==2) & (df_counts.dino_dist_min==0.9)]["pct_of_total"].iloc[0])
ax.text(0.5, 0.92, f"★ {ed2_dino09:,} pairs\n({pct:.2f}% of total)",
        transform=ax.transAxes, fontsize=9,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#e65c00", alpha=0.9))

ax.set_xlabel("EC Token Edit Distance (0=identical, 8=max)", fontsize=10)
ax.set_ylabel("DINO Image Distance (1 − cos sim)", fontsize=10)
ax.set_title("Density of (edit dist, DINO dist) pairs", fontsize=10)
ax.set_xticks(range(9))
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- right: 「類似トークン」グループ別の DINO 距離分布 ---
ax2 = axes[1]
colors = ["#1565c0", "#2196F3", "#64B5F6", "#B3E5FC", "#E3F2FD",
          "#CFD8DC", "#90A4AE", "#546E7A", "#263238"]
from matplotlib.patches import Patch

violins = []
labels_v = []
dino_by_edit = []
for ed in range(9):
    mask = edit_vals == ed
    if mask.sum() > 0:
        dino_by_edit.append(dino_vals[mask])
    else:
        dino_by_edit.append(np.array([0.0]))
    labels_v.append(f"{ed}")

vp = ax2.violinplot(dino_by_edit, positions=range(9),
                    showmedians=True, showextrema=False, widths=0.7)
for i, (body, col) in enumerate(zip(vp["bodies"], colors)):
    body.set_facecolor(col)
    body.set_alpha(0.75)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.5)

ax2.axhline(y=0.9, color="orange", linestyle="--", linewidth=1.5, alpha=0.9, label="DINO dist = 0.9")
ax2.set_xticks(range(9))
ax2.set_xticklabels(labels_v)
ax2.set_xlabel("EC Token Edit Distance", fontsize=10)
ax2.set_ylabel("DINO Image Distance", fontsize=10)
ax2.set_title("DINO Distance Distribution per Edit Distance\n(violin plot)", fontsize=10)
ax2.legend(fontsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
out_png = OUT / "B_editdist_dino_scatter.png"
out_pdf = OUT / "B_editdist_dino_scatter.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out_png}")

# ── 追加サマリ: edit_dist ≤ 2 の内訳 ─────────────────────────────────────────
print("\n=== 主要な閾値での要約 ===")
mask_ed2 = edit_vals <= 2
n_ed2 = mask_ed2.sum()
dino_ed2 = dino_vals[mask_ed2]
print(f"edit_dist ≤ 2 のペア数: {n_ed2:,} / {total_pairs:,} ({n_ed2/total_pairs*100:.2f}%)")
print(f"  その中で DINO dist 分布:")
for p in [25, 50, 75, 90, 95]:
    print(f"    p{p:02d} = {np.percentile(dino_ed2, p):.4f}")

mask_target = mask_ed2 & (dino_vals >= 0.9)
print(f"\n「edit_dist ≤ 2 かつ DINO dist ≥ 0.9」: {mask_target.sum():,} ペア ({mask_target.sum()/total_pairs*100:.3f}%)")
print(f"  全ペアに対する割合: {mask_target.sum()/total_pairs*100:.3f}%")
print(f"  edit_dist ≤ 2 ペアに対する割合: {mask_target.sum()/n_ed2*100:.2f}%")

print("\nDone.")
