"""
Tables 4, 5, 6 の点推定 + Bootstrap 95% CI
論文設定に完全準拠:
  Table 4: τ=0.7 (B: CLIP>τ, C: CLIP<τ), ε-sweep, seed=42
  Table 5: τ=0.7 (同上), per edit-distance, seed=42
  Table 6: R@1 distractor sensitivity, seed=42, N_TRIALS=2000
全て per-sample/per-triplet 正誤配列を保存 → bootstrap

環境変数:
  PROJ_DIR  : リポジトリルート (default: カレントディレクトリ)
  COCO_ROOT : MS-COCO データセットルート (default: data/coco2017)
"""

import json, ast, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

BASE = Path(os.environ.get("PROJ_DIR", "."))
V2   = BASE / "outputs/experiments_v2"
V3   = BASE / "experiments_v3/outputs"
V3.mkdir(parents=True, exist_ok=True)

N_BOOT   = 10000
N_SAMPLE = 10_000
N_TRIALS = 2000   # Table 6
TAU      = 0.7
RNG_SEED = 42

# ─── 共通埋め込みロード ────────────────────────────────────────────────────────
def L2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

print("Loading embeddings...")
gt_clip   = L2(np.load(V2 / "cache_gt_clip_vitl14.npy").astype(np.float32))
cap_emb   = L2(np.load(V3 / "cache_gt_clip_captions_vitl14.npy").astype(np.float32))
trans_emb = L2(np.load(V2 / "cache_pred_clip_text_vitl14.npy").astype(np.float32))
dino_gen  = L2(np.load(V3 / "cache_dino_vitb14_gen.npy").astype(np.float32))
dino_gt   = L2(np.load(V2 / "cache_dino_vitb14_gt.npy").astype(np.float32))
N = len(gt_clip)
print(f"  N={N}")

print("Computing similarity matrices...")
gt_gt_clip = gt_clip  @ gt_clip.T   # (N,N) ground truth
cap_sim    = cap_emb  @ cap_emb.T   # (N,N) caption similarity
trans_cap  = trans_emb @ cap_emb.T  # (N,N) translation score
emcom_mat  = dino_gen  @ dino_gt.T  # (N,N) EmCom score

def boot_mean_ci(arr, rng, n_boot=N_BOOT):
    n = len(arr)
    boots = np.array([rng.choice(arr, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4: Translation vs EmCom  (τ=0.7, ε-sweep)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE 4: Translation vs EmCom  (τ=0.7, ε-sweep, seed=42)")
print("="*70)

# 有効トリプレットプールを事前構築（τ=0.7 絶対閾値）
print("Building triplet pools for each ε (τ=0.7)...")
t4_pool = {}
for eps in [0.02, 0.05, 0.10, 0.20]:
    A_list, B_list, C_list = [], [], []
    for a in range(N):
        pos = np.where(gt_gt_clip[a] > TAU)[0]; pos = pos[pos != a]
        neg = np.where(gt_gt_clip[a] < TAU)[0]; neg = neg[neg != a]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = np.abs(cap_sim[a, pos][:, None] - cap_sim[a, neg][None, :])
        pi, ni = np.where(diff <= eps)
        if len(pi) == 0:
            continue
        A_list.append(np.full(len(pi), a, dtype=np.int32))
        B_list.append(pos[pi].astype(np.int32))
        C_list.append(neg[ni].astype(np.int32))
    t4_pool[eps] = (np.concatenate(A_list),
                    np.concatenate(B_list),
                    np.concatenate(C_list))
    print(f"  ε={eps:.2f}: {len(t4_pool[eps][0]):,} triplets")

rng4 = np.random.default_rng(RNG_SEED)
t4_rows = []
print("\n  ε    N_used  Translation [95%CI]           EmCom [95%CI]             Δ(T-E) [95%CI]")
for eps in [0.02, 0.05, 0.10, 0.20]:
    A, B, C = t4_pool[eps]
    n_total = len(A)
    if n_total > N_SAMPLE:
        idx = rng4.choice(n_total, N_SAMPLE, replace=False)
        As, Bs, Cs = A[idx], B[idx], C[idx]
    else:
        As, Bs, Cs = A, B, C
    n_used = len(As)

    tb = trans_cap[As, Bs]; tc = trans_cap[As, Cs]
    trans_correct = np.where(tb > tc, 1.0, np.where(tb == tc, 0.5, 0.0)).astype(np.float32)

    eb = emcom_mat[As, Bs]; ec = emcom_mat[As, Cs]
    emcom_correct = np.where(eb > ec, 1.0, np.where(eb == ec, 0.5, 0.0)).astype(np.float32)

    diff_correct = trans_correct - emcom_correct

    rng_b = np.random.default_rng(RNG_SEED + int(eps * 100))
    t_m, t_lo, t_hi = boot_mean_ci(trans_correct, rng_b)
    e_m, e_lo, e_hi = boot_mean_ci(emcom_correct, rng_b)
    d_m, d_lo, d_hi = boot_mean_ci(diff_correct,  rng_b)

    print(f"  {eps:.2f}  {n_used:>7,}  "
          f"{t_m*100:5.1f}% [{t_lo*100:4.1f},{t_hi*100:4.1f}]  "
          f"{e_m*100:5.1f}% [{e_lo*100:4.1f},{e_hi*100:4.1f}]  "
          f"{d_m*100:+5.1f}% [{d_lo*100:+5.1f},{d_hi*100:+5.1f}]")

    np.savez(V3 / f"paper_t4_eps{int(eps*100):02d}.npz",
             trans_correct=trans_correct, emcom_correct=emcom_correct,
             A_s=As.astype(np.int32), B_s=Bs.astype(np.int32), C_s=Cs.astype(np.int32))
    t4_rows.append({"eps": eps, "N": n_used,
                    "trans_mean": t_m, "trans_lo": t_lo, "trans_hi": t_hi,
                    "emcom_mean": e_m, "emcom_lo": e_lo, "emcom_hi": e_hi,
                    "diff_mean": d_m, "diff_lo": d_lo, "diff_hi": d_hi})

# ε=0.20→0.02 drop の CI
d_loose = np.load(V3 / "paper_t4_eps20.npz")
d_tight = np.load(V3 / "paper_t4_eps02.npz")
rng_drop = np.random.default_rng(RNG_SEED + 99)
print()
for method, key in [("Translation", "trans_correct"), ("EmCom", "emcom_correct")]:
    loose_arr = d_loose[key]; tight_arr = d_tight[key]
    drop_obs = loose_arr.mean() - tight_arr.mean()
    boots = np.array([
        rng_drop.choice(loose_arr, len(loose_arr), replace=True).mean() -
        rng_drop.choice(tight_arr, len(tight_arr), replace=True).mean()
        for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  Drop ε=0.20→0.02  {method}: {drop_obs*100:+.1f}% [{lo*100:+.1f}, {hi*100:+.1f}]")

pd.DataFrame(t4_rows).to_csv(V3 / "paper_bootstrap_t4.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 5: EmCom vs TopSim vs CBM  (τ=0.7, per edit distance)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE 5: EmCom vs TopSim vs CBM  (τ=0.7, seed=42)")
print("="*70)

# edit distance 行列（キャッシュ済み）
print("Loading edit distance matrix...")
edit_mat = np.load(V3 / "B_edit_mat.npy")
neg_edit = -edit_mat.astype(np.float32)

# CBM 行列（再計算 - deterministic）
print("Building CBM matrix...")
df_tok = pd.read_csv(V2 / "t11_translation_per_sample_scores.csv")
seqs = [tuple(ast.literal_eval(x)) for x in df_tok["ec_tokens"]]
COCO_ANN = Path(os.environ.get("COCO_ROOT", "data/coco2017")) / "annotations" / "instances_val2017.json"
with open(COCO_ANN) as f:
    inst_data = json.load(f)
cat_ids = sorted(c["id"] for c in inst_data["categories"])
cat2idx = {c: i for i, c in enumerate(cat_ids)}
N_CATS = len(cat_ids)
img_cat_counts = defaultdict(lambda: defaultdict(int))
for ann in inst_data["annotations"]:
    img_cat_counts[ann["image_id"]][ann["category_id"]] += 1
image_ids = df_tok["image_id"].astype(int).tolist()
dominant_cat = np.array([
    cat2idx[max(img_cat_counts[iid], key=img_cat_counts[iid].get)]
    if iid in img_cat_counts else -1
    for iid in image_ids])
VOCAB = 256
cooccur = np.zeros((VOCAB, N_CATS), dtype=np.float32)
for i, seq in enumerate(seqs):
    if dominant_cat[i] < 0: continue
    for v in set(seq): cooccur[v, dominant_cat[i]] += 1.0
row_ind, col_ind = linear_sum_assignment(-cooccur)
token_to_concept = dict(zip(row_ind.tolist(), col_ind.tolist()))
img_cbm = np.zeros((N, N_CATS), dtype=np.float32)
for i, seq in enumerate(seqs):
    for v in seq:
        if v in token_to_concept: img_cbm[i, token_to_concept[v]] += 1.0
    img_cbm[i] /= len(seq)
norms = np.linalg.norm(img_cbm, axis=1, keepdims=True)
img_cbm_norm = img_cbm / np.where(norms == 0, 1.0, norms)
cbm_mat = img_cbm_norm @ img_cbm_norm.T
print(f"  CBM matrix done. token_to_concept size={len(token_to_concept)}")

def accuracy_arr(a_idx, b_idx, c_idx, score_mat):
    sb = score_mat[a_idx, b_idx]
    sc = score_mat[a_idx, c_idx]
    return np.where(sb > sc, 1.0, np.where(sb == sc, 0.5, 0.0)).astype(np.float32)

rng5 = np.random.default_rng(RNG_SEED)
t5_rows = []
print("\n  Condition    N      EmCom [95%CI]           TopSim [95%CI]          CBM [95%CI]")
for d, n_paper in [(1, None), (2, None), (3, None)]:
    # τ=0.7 基準: B: CLIP>τ, C: CLIP<τ, edit_dist=d
    A_list, B_list, C_list = [], [], []
    for a in range(N):
        partners = np.where((edit_mat[a] == d) & (np.arange(N) != a))[0]
        if len(partners) < 2: continue
        clip_p = gt_gt_clip[a, partners]
        pos_idx = partners[clip_p > TAU]
        neg_idx = partners[clip_p < TAU]
        for b in pos_idx:
            for c in neg_idx:
                A_list.append(a); B_list.append(int(b)); C_list.append(int(c))

    if not A_list:
        print(f"  ed={d}: no triplets"); continue

    Ah = np.array(A_list); Bh = np.array(B_list); Ch = np.array(C_list)
    n_total = len(Ah)
    if n_total > N_SAMPLE:
        idx = rng5.choice(n_total, N_SAMPLE, replace=False)
        Ah, Bh, Ch = Ah[idx], Bh[idx], Ch[idx]

    per_em = accuracy_arr(Ah, Bh, Ch, emcom_mat)
    per_ts = accuracy_arr(Ah, Bh, Ch, neg_edit)
    per_cb = accuracy_arr(Ah, Bh, Ch, cbm_mat)

    rng_b5 = np.random.default_rng(RNG_SEED + d)
    em_m, em_lo, em_hi = boot_mean_ci(per_em, rng_b5)
    ts_m, ts_lo, ts_hi = boot_mean_ci(per_ts, rng_b5)
    cb_m, cb_lo, cb_hi = boot_mean_ci(per_cb, rng_b5)

    cond_name = f"ed={d} (N={len(Ah):,})"
    print(f"  {cond_name:<20}  "
          f"{em_m*100:5.1f}% [{em_lo*100:4.1f},{em_hi*100:4.1f}]  "
          f"{ts_m*100:5.1f}% [{ts_lo*100:4.1f},{ts_hi*100:4.1f}]  "
          f"{cb_m*100:5.1f}% [{cb_lo*100:4.1f},{cb_hi*100:4.1f}]")

    np.savez(V3 / f"paper_t5_ed{d}.npz",
             emcom=per_em, topsim=per_ts, cbm=per_cb,
             a=Ah.astype(np.int32), b=Bh.astype(np.int32), c=Ch.astype(np.int32))
    t5_rows.append({"ed": d, "N": len(Ah),
                    "emcom_mean": em_m, "emcom_lo": em_lo, "emcom_hi": em_hi,
                    "topsim_mean": ts_m, "topsim_lo": ts_lo, "topsim_hi": ts_hi,
                    "cbm_mean": cb_m, "cbm_lo": cb_lo, "cbm_hi": cb_hi})

pd.DataFrame(t5_rows).to_csv(V3 / "paper_bootstrap_t5.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 6: R@1 distractor sensitivity (seed=42, N_TRIALS=2000)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE 6: R@1 distractor sensitivity  (seed=42, N_TRIALS=2000)")
print("="*70)

print("Loading listener features...")
text_feats  = np.load(V3 / "cache_listener_text_feat.npy").astype(np.float32)
image_feats = np.load(V3 / "cache_listener_image_feat.npy").astype(np.float32)
clip_sim_mat = gt_clip @ gt_clip.T
sim_all = text_feats @ image_feats.T   # (N,N)
print(f"  sim_all: {sim_all.shape}")

# 元スクリプト (run_distractor_sensitivity.py) と同一の RNG 消費順序:
#   Part A: accuracy_random_n for N_way=[2,4,8,16,32,100,500,1000,5000]
#   Part B: accuracy_hard_n  → accuracy_random_n for 同 N_way
# この順序を再現することで論文 Table 6 の点推定と完全一致する。
rng6 = np.random.default_rng(RNG_SEED)   # 単一インスタンス・連続消費
N_TOTAL = len(text_feats)
ALL_NW = [2, 4, 8, 16, 32, 100, 500, 1000, 5000]

def _random_correct(nd, n_trials):
    idxs = rng6.choice(N_TOTAL, n_trials, replace=False)
    correct = np.zeros(n_trials, dtype=np.float32)
    if nd == N_TOTAL - 1:
        for k, i in enumerate(idxs):
            correct[k] = float(sim_all[i].argmax() == i)
    else:
        for k, i in enumerate(idxs):
            cands = rng6.choice([j for j in range(N_TOTAL) if j != i],
                                nd, replace=False)
            pool = np.concatenate([[i], cands])
            correct[k] = float((text_feats[i] @ image_feats[pool].T).argmax() == 0)
    return correct

def _hard_correct(nd, n_trials):
    idxs = rng6.choice(N_TOTAL, n_trials, replace=False)
    correct = np.zeros(n_trials, dtype=np.float32)
    if nd == N_TOTAL - 1:
        for k, i in enumerate(idxs):
            correct[k] = float(sim_all[i].argmax() == i)
    else:
        for k, i in enumerate(idxs):
            row = clip_sim_mat[i].copy(); row[i] = -np.inf
            hard_cands = np.argpartition(row, -nd)[-nd:]
            pool = np.concatenate([[i], hard_cands])
            correct[k] = float((text_feats[i] @ image_feats[pool].T).argmax() == 0)
    return correct

# Part A: ランダム（全 N_way）→ RNG 状態を消費
print("  Part A (random, all n-way)...")
_part_a = {nw: _random_correct(nw - 1, N_TRIALS) for nw in ALL_NW}

# Part B: hard → random（全 N_way）→ 正式な per-query 配列を取得
print("  Part B (hard + random, all n-way)...")
perq_hard = {}; perq_rand = {}
for nw in ALL_NW:
    perq_hard[nw] = _hard_correct(nw - 1, N_TRIALS)
    perq_rand[nw] = _random_correct(nw - 1, N_TRIALS)

t6_rows = []
print("\n  n-way    Random [95%CI]            Hard [95%CI]              Gap [95%CI]")
for nw in [32, 100, 1000, 5000]:
    rand_correct = perq_rand[nw]
    hard_correct = perq_hard[nw]
    gap_correct  = rand_correct - hard_correct

    rng_b6 = np.random.default_rng(RNG_SEED + nw + 20000)
    r_m, r_lo, r_hi = boot_mean_ci(rand_correct, rng_b6)
    h_m, h_lo, h_hi = boot_mean_ci(hard_correct, rng_b6)
    g_m, g_lo, g_hi = boot_mean_ci(gap_correct,  rng_b6)

    print(f"  {nw:>5}-way  "
          f"{r_m*100:5.1f}% [{r_lo*100:4.1f},{r_hi*100:4.1f}]  "
          f"{h_m*100:5.1f}% [{h_lo*100:4.1f},{h_hi*100:4.1f}]  "
          f"{g_m*100:+5.1f}% [{g_lo*100:+4.1f},{g_hi*100:+4.1f}]")

    np.savez(V3 / f"paper_t6_nway{nw}.npz",
             rand_correct=rand_correct, hard_correct=hard_correct)
    t6_rows.append({"nway": nw, "N": N_TRIALS,
                    "rand_mean": r_m, "rand_lo": r_lo, "rand_hi": r_hi,
                    "hard_mean": h_m, "hard_lo": h_lo, "hard_hi": h_hi,
                    "gap_mean": g_m, "gap_lo": g_lo, "gap_hi": g_hi})

pd.DataFrame(t6_rows).to_csv(V3 / "paper_bootstrap_t6.csv", index=False)

print(f"\nAll CSV saved to {V3}")
print("Done.")
