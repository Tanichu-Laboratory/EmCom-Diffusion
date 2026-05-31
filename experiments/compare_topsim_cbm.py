"""
Plan B Step 2: トリプレット評価による手法比較

タスク:
  アンカー A、陽性 B（DINOで視覚的に類似）、陰性 C（DINOで視覚的に異なる）の3枚組で、
  各手法が「B が A に近い」と正しく答えられるかを測る。

各手法のスコア:
  EmCom-Diffusion: DINO(gen_A, GT_B) vs DINO(gen_A, GT_C)   ← 生成画像とGTのDINO類似度
  TopSim:          -edit_dist(tokens_A, tokens_B) vs -edit_dist(tokens_A, tokens_C)
  CBM:             cos(cat_dist_A, cat_dist_B) vs cos(cat_dist_A, cat_dist_C)
                   ※ cat_dist[i] = val5000からトークン値→COCO80カテゴリ分布のマッピング

2種類のトリプレット:
  [General]  B = CLIP(GT_A, GT_B) 上位50近傍からサンプル、C = 残りからランダム（N=10,000）
  [Hard-edN] edit_dist(A,B) = edit_dist(A,C) = N（TopSimはタイになる）
             → CLIP(GT_A, GT_B) > CLIP(GT_A, GT_C) の方を陽性 B とする

注: グラウンドトゥルース(CLIP GT-GT) と EmComスコア(DINO gen-GT) を独立した指標で測定

出力:
  experiments_v3/outputs/B_triplet_result.csv
  experiments_v3/outputs/B_triplet_bar.png/pdf
"""

import ast, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist as rf_cdist

# ── パス ──────────────────────────────────────────────────────────────────────
BASE     = Path(os.environ.get("PROJ_DIR", "."))
DATA_DIR = BASE / "outputs/experiments_v2"
OUT      = BASE / "experiments_v3/outputs"
COCO_ANN = Path(os.environ.get("COCO_ROOT", "data/coco2017")) / "annotations" / "instances_val2017.json"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ── データ読み込み ──────────────────────────────────────────────────────────────
print("Loading data...")
df   = pd.read_csv(DATA_DIR / "t11_translation_per_sample_scores.csv")
N    = len(df)
seqs = [tuple(ast.literal_eval(x)) for x in df["ec_tokens"]]

dino_gt  = np.load(DATA_DIR / "cache_dino_vitb14_gt.npy").astype(np.float32)
dino_gt  /= (np.linalg.norm(dino_gt, axis=1, keepdims=True) + 1e-8)
gt_clip  = np.load(DATA_DIR / "cache_gt_clip_vitl14.npy").astype(np.float32)
gt_clip  /= (np.linalg.norm(gt_clip, axis=1, keepdims=True) + 1e-8)
dino_gen = np.load(OUT / "cache_dino_vitb14_gen.npy").astype(np.float32)
dino_gen /= (np.linalg.norm(dino_gen, axis=1, keepdims=True) + 1e-8)
print(f"  N={N}, dino_gt={dino_gt.shape}, dino_gen={dino_gen.shape}")

# ── CBM: Bipartite matching（論文定義準拠）────────────────────────────────────
# Paper の定義: bipartite matching between token vocab V and concepts C
#   each token matched to AT MOST ONE concept (Hungarian algorithm)
#   score = cos(concept_freq_A, concept_freq_X) per pair
print("\nBuilding CBM via bipartite matching (token V ↔ COCO concept C)...")
from scipy.optimize import linear_sum_assignment

with open(COCO_ANN) as f:
    inst_data = json.load(f)

# カテゴリ ID → 連続インデックス（80 カテゴリ）
cat_ids = sorted(c["id"] for c in inst_data["categories"])
N_CATS  = len(cat_ids)   # 80
cat2idx = {c: i for i, c in enumerate(cat_ids)}

# image_id → dominant category（最多インスタンス数のカテゴリ）
img_cat_counts = defaultdict(lambda: defaultdict(int))
for ann in inst_data["annotations"]:
    img_cat_counts[ann["image_id"]][ann["category_id"]] += 1

image_ids    = df["image_id"].astype(int).tolist()
dominant_cat = []
for iid in image_ids:
    if iid in img_cat_counts:
        dom = max(img_cat_counts[iid], key=img_cat_counts[iid].get)
        dominant_cat.append(cat2idx[dom])
    else:
        dominant_cat.append(-1)
dominant_cat = np.array(dominant_cat)
print(f"  images with annotation: {(dominant_cat >= 0).sum()}/{N}")

# Step 1: 共起行列 cooccur[v, c]
#   = トークン v を含む画像で カテゴリ c が dominant な枚数（binary: 画像ごとに 1 カウント）
VOCAB   = 256
cooccur = np.zeros((VOCAB, N_CATS), dtype=np.float32)
for i, seq in enumerate(seqs):
    cat_i = dominant_cat[i]
    if cat_i < 0:
        continue
    for v in set(seq):   # 画像ごとにトークン値を重複なしでカウント
        cooccur[v, cat_i] += 1.0

# Step 2: Hungarian algorithm で optimal bipartite matching を求める
#   最大化問題なので -cooccur を渡す
#   サイズ: (256 tokens) × (80 concepts) → 80 トークンがマッチされ 176 はアンマッチ
row_ind, col_ind = linear_sum_assignment(-cooccur)
token_to_concept = dict(zip(row_ind.tolist(), col_ind.tolist()))
print(f"  Matched tokens: {len(token_to_concept)}/{VOCAB}  "
      f"(unmatched: {VOCAB - len(token_to_concept)})")

# マッチングの品質確認
matched_weight = cooccur[row_ind, col_ind].sum()
print(f"  Total matched co-occurrence weight: {matched_weight:.0f}")

# Step 3: 各画像を「matched concept の頻度ベクトル」で表現
#   tokens_i の各トークン v → matched concept c(v)（未マッチは無視）
img_cbm = np.zeros((N, N_CATS), dtype=np.float32)
for i, seq in enumerate(seqs):
    for v in seq:
        if v in token_to_concept:
            img_cbm[i, token_to_concept[v]] += 1.0
    img_cbm[i] /= len(seq)   # 系列長で正規化（= 8）

# L2 正規化
norms = np.linalg.norm(img_cbm, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)
img_cbm_norm = img_cbm / norms   # (N, 80)

# CBM スコア行列
print("  Computing CBM score matrix...")
cbm_mat = img_cbm_norm @ img_cbm_norm.T   # (N, N)
print(f"  cbm_mat: mean={cbm_mat.mean():.4f}  std={cbm_mat.std():.4f}")

# ── 行列の事前計算 ─────────────────────────────────────────────────────────────
print("\nPrecomputing other matrices...")

edit_path = OUT / "B_edit_mat.npy"
if edit_path.exists():
    edit_mat = np.load(edit_path)
    print("  edit_mat: loaded from cache")
else:
    t0 = time.time()
    edit_mat = rf_cdist(seqs, seqs, scorer=Levenshtein.distance, workers=-1)
    np.save(edit_path, edit_mat)
    print(f"  edit_mat: computed in {time.time()-t0:.1f}s")

# GT-GT CLIP 類似度（トリプレット選択のグラウンドトゥルース）
gt_gt_clip = gt_clip @ gt_clip.T         # (N, N)

# EmCom スコア行列: DINO(gen_A, GT_X)
emcom_mat  = dino_gen @ dino_gt.T        # (N, N)

neg_edit   = -edit_mat.astype(np.float32)

# ── スコア関数 ─────────────────────────────────────────────────────────────────
def accuracy(a_idx, b_idx, c_idx, score_mat, tie_val=0.5):
    sb = score_mat[a_idx, b_idx]
    sc = score_mat[a_idx, c_idx]
    return np.where(sb > sc, 1.0, np.where(sb == sc, tie_val, 0.0))

# ── General トリプレット（N=10,000）────────────────────────────────────────────
MIN_GAP  = 0.10   # CLIP(GT_A,GT_B) - CLIP(GT_A,GT_C) の最小ギャップ（General/Hard 共通）
N_SAMPLE = 10_000  # 全条件で揃えるサンプル数

print(f"\nBuilding GENERAL triplets (N={N_SAMPLE}, gap≥{MIN_GAP})...")
TOP_K = 50

triplets_gen = []
anchor_pool = list(RNG.permutation(N)) * 20   # 足りなくなっても回せるよう多めに
ai = 0
while len(triplets_gen) < N_SAMPLE:
    a = int(anchor_pool[ai])
    ai += 1
    clip_row = gt_gt_clip[a].copy()
    clip_row[a] = -np.inf
    top_idx = np.argpartition(clip_row, -TOP_K)[-TOP_K:]
    b = int(RNG.choice(top_idx))
    exclude = set(top_idx.tolist() + [a])
    remain  = [j for j in range(N) if j not in exclude]
    c = int(RNG.choice(remain))
    if gt_gt_clip[a, b] - gt_gt_clip[a, c] >= MIN_GAP:
        triplets_gen.append((int(a), b, c))

a_g   = np.array([t[0] for t in triplets_gen])
b_g   = np.array([t[1] for t in triplets_gen])
c_g   = np.array([t[2] for t in triplets_gen])
gap_g = gt_gt_clip[a_g, b_g] - gt_gt_clip[a_g, c_g]

per_emcom_gen  = accuracy(a_g, b_g, c_g, emcom_mat)
per_topsim_gen = accuracy(a_g, b_g, c_g, neg_edit)
per_cbm_gen    = accuracy(a_g, b_g, c_g, cbm_mat)
acc_emcom_gen  = per_emcom_gen.mean()
acc_topsim_gen = per_topsim_gen.mean()
acc_cbm_gen    = per_cbm_gen.mean()
print(f"  General: EmCom={acc_emcom_gen:.3f}  TopSim={acc_topsim_gen:.3f}  CBM={acc_cbm_gen:.3f}"
      f"  gap_median={np.median(gap_g):.3f}")
np.savez(OUT / "B_per_triplet_general.npz",
         emcom=per_emcom_gen.astype(np.float32),
         topsim=per_topsim_gen.astype(np.float32),
         cbm=per_cbm_gen.astype(np.float32),
         a=a_g.astype(np.int32), b=b_g.astype(np.int32), c=c_g.astype(np.int32))

# ── Hard トリプレット（edit_dist(A,B) == edit_dist(A,C)）────────────────────
print("\nBuilding HARD triplets by edit distance level...")

results = []

for d in [1, 2, 3]:   # ed=0 は EC 言語自体の識別不能問題のため除外
    triplets_hard = []
    for a in range(N):
        partners = np.where((edit_mat[a] == d) & (np.arange(N) != a))[0]
        if len(partners) < 2:
            continue
        clip_partners = gt_gt_clip[a, partners]   # CLIP GT-GT で正解を決める
        n_p = len(partners)
        for bi in range(n_p):
            for ci in range(n_p):
                if bi == ci:
                    continue
                gap = clip_partners[bi] - clip_partners[ci]
                if gap >= MIN_GAP:   # ギャップが十分大きいペアのみ採用
                    triplets_hard.append((a, int(partners[bi]), int(partners[ci])))

    if not triplets_hard:
        print(f"  edit_dist={d}: no hard triplets after gap filter")
        continue

    print(f"  edit_dist={d}: {len(triplets_hard):,} triplets after gap≥{MIN_GAP} filter")

    if len(triplets_hard) > N_SAMPLE:
        idx_s = RNG.choice(len(triplets_hard), N_SAMPLE, replace=False)
        triplets_hard = [triplets_hard[i] for i in idx_s]

    a_h   = np.array([t[0] for t in triplets_hard])
    b_h   = np.array([t[1] for t in triplets_hard])
    c_h   = np.array([t[2] for t in triplets_hard])
    gap_h = gt_gt_clip[a_h, b_h] - gt_gt_clip[a_h, c_h]

    per_emcom  = accuracy(a_h, b_h, c_h, emcom_mat)
    per_topsim = accuracy(a_h, b_h, c_h, neg_edit)
    per_cbm    = accuracy(a_h, b_h, c_h, cbm_mat)
    acc_emcom  = per_emcom.mean()
    acc_topsim = per_topsim.mean()
    acc_cbm    = per_cbm.mean()

    np.savez(OUT / f"B_per_triplet_hard_ed{d}.npz",
             emcom=per_emcom.astype(np.float32),
             topsim=per_topsim.astype(np.float32),
             cbm=per_cbm.astype(np.float32),
             a=a_h.astype(np.int32), b=b_h.astype(np.int32), c=c_h.astype(np.int32))

    print(f"  Hard ed={d} used (N={len(triplets_hard):,}): "
          f"EmCom={acc_emcom:.3f}  TopSim={acc_topsim:.3f}  CBM={acc_cbm:.3f}"
          f"  gap_median={np.median(gap_h):.3f}")

    results.append({
        "condition": f"Hard (ed={d})",
        "edit_dist": d,
        "n_triplets": len(triplets_hard),
        "gap_median": float(np.median(gap_h)),
        "EmCom": acc_emcom,
        "TopSim": acc_topsim,
        "CBM": acc_cbm,
        "Chance": 0.5,
    })

results.insert(0, {
    "condition": "General",
    "edit_dist": -1,
    "n_triplets": len(triplets_gen),
    "gap_median": float(np.median(gap_g)),
    "EmCom": acc_emcom_gen,
    "TopSim": acc_topsim_gen,
    "CBM": acc_cbm_gen,
    "Chance": 0.5,
})

df_res = pd.DataFrame(results)
df_res.to_csv(OUT / "B_triplet_result.csv", index=False)
print("\n=== Results ===")
print(df_res[["condition","n_triplets","gap_median","EmCom","TopSim","CBM","Chance"]].to_string(index=False))

# ── 棒グラフ ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 2.2]})
fig.suptitle("Triplet Discrimination Accuracy: EmCom vs TopSim vs CBM",
             fontsize=13, fontweight="bold")

METHODS = ["EmCom", "TopSim", "CBM"]
COLORS  = {"EmCom": "#1565c0", "TopSim": "#e53935", "CBM": "#f57c00"}

ax0 = axes[0]
vals = [df_res.loc[df_res.condition=="General", m].iloc[0] * 100 for m in METHODS]
bars = ax0.bar(METHODS, vals, color=[COLORS[m] for m in METHODS], width=0.55, edgecolor="white")
ax0.axhline(50, color="gray", linestyle="--", linewidth=1.2, label="Chance (50%)")
ax0.set_ylim(0, 100)
ax0.set_title("General Triplets\n(N=10,000)", fontsize=10)
ax0.set_ylabel("Accuracy (%)", fontsize=10)
ax0.legend(fontsize=8)
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
for bar, v in zip(bars, vals):
    ax0.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.1f}%",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

ax1 = axes[1]
hard_rows = df_res[df_res.edit_dist >= 0].copy()
ds = hard_rows["edit_dist"].tolist()
x  = np.arange(len(ds))
w  = 0.25

for k, m in enumerate(METHODS):
    vals_h = hard_rows[m].values * 100
    br = ax1.bar(x + (k - 1) * w, vals_h, w,
                 label=m, color=COLORS[m], edgecolor="white", alpha=0.9)
    for bar, v in zip(br, vals_h):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=7)

ax1.axhline(50, color="gray", linestyle="--", linewidth=1.2, label="Chance (50%)")
ax1.set_xticks(x)
ns = hard_rows["n_triplets"].tolist()
ax1.set_xticklabels([f"ed={d}\n(N={n:,})" for d, n in zip(ds, ns)], fontsize=8)
ax1.set_ylabel("Accuracy (%)", fontsize=10)
ax1.set_title("Hard Triplets: edit_dist(A,B) = edit_dist(A,C)\n"
              "(TopSim has no edit-distance signal → exactly 50%)", fontsize=10)
ax1.set_ylim(0, 100)
ax1.legend(fontsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "B_triplet_bar.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "B_triplet_bar.pdf", bbox_inches="tight")
print(f"\nSaved {OUT}/B_triplet_bar.png")
print("Done.")
