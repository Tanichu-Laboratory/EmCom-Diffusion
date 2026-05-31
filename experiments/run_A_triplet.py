"""
Plan A: Translation ベース評価 vs EmCom-Diffusion の比較
Hard 条件: |cap_sim(A,B) - cap_sim(A,C)| <= ε  (GT キャプション CLIP 類似度)
正解基準: CLIP_image_sim(A,B) > 0.7 かつ CLIP_image_sim(A,C) < 0.7

スコア:
  Translation : CLIP_text_sim(trans_A, GT_caps_X)  cross-modal
  EmCom       : DINO_sim(gen_A, GT_X)

サンプリング: 全有効トリプレットを列挙 → 均一にN_SAMPLE件サブサンプル
"""

import os
import numpy as np
from pathlib import Path

BASE = Path(os.environ.get("PROJ_DIR", "."))
V2   = BASE / "outputs/experiments_v2"
V3   = BASE / "experiments_v3/outputs"
V3.mkdir(parents=True, exist_ok=True)

print("Loading embeddings...")
gt_clip   = np.load(V2 / "cache_gt_clip_vitl14.npy").astype(np.float32)
gt_clip  /= (np.linalg.norm(gt_clip,  axis=1, keepdims=True) + 1e-8)
cap_emb   = np.load(V3 / "cache_gt_clip_captions_vitl14.npy").astype(np.float32)
cap_emb  /= (np.linalg.norm(cap_emb,  axis=1, keepdims=True) + 1e-8)
trans_emb = np.load(V2 / "cache_pred_clip_text_vitl14.npy").astype(np.float32)
trans_emb/= (np.linalg.norm(trans_emb, axis=1, keepdims=True) + 1e-8)
dino_gen  = np.load(V3 / "cache_dino_vitb14_gen.npy").astype(np.float32)
dino_gt   = np.load(V2 / "cache_dino_vitb14_gt.npy").astype(np.float32)
dino_gen /= (np.linalg.norm(dino_gen,  axis=1, keepdims=True) + 1e-8)
dino_gt  /= (np.linalg.norm(dino_gt,   axis=1, keepdims=True) + 1e-8)

N = len(gt_clip)
print(f"  N={N}")

print("Computing similarity matrices...")
gt_gt_clip = gt_clip  @ gt_clip.T    # (N,N)
cap_sim    = cap_emb  @ cap_emb.T    # (N,N)
trans_cap  = trans_emb @ cap_emb.T   # (N,N) Translation スコア
emcom_mat  = dino_gen  @ dino_gt.T   # (N,N) EmCom スコア

CLIP_THR = 0.7
N_SAMPLE = 10_000
RNG = np.random.default_rng(42)

print("\n=== Plan A トリプレット評価 ===")
print(f"正解基準: CLIP_image(A,B)>{CLIP_THR} かつ CLIP_image(A,C)<{CLIP_THR}")
print(f"Hard 条件: |cap_sim(A,B) - cap_sim(A,C)| <= ε\n")

results = []
for eps in [0.02, 0.05, 0.10, 0.20]:
    # アンカーごとに有効トリプレットを全列挙
    A_list, B_list, C_list = [], [], []

    for a in range(N):
        pos = np.where((gt_gt_clip[a] > CLIP_THR))[0]
        neg = np.where((gt_gt_clip[a] < CLIP_THR))[0]
        pos = pos[pos != a]
        neg = neg[neg != a]
        if len(pos) == 0 or len(neg) == 0:
            continue

        cs_pos = cap_sim[a, pos]   # (|pos|,)
        cs_neg = cap_sim[a, neg]   # (|neg|,)

        # |cs_pos[i] - cs_neg[j]| <= eps: ブロードキャスト
        diff = np.abs(cs_pos[:, None] - cs_neg[None, :])  # (|pos|,|neg|)
        pi, ni = np.where(diff <= eps)
        if len(pi) == 0:
            continue

        A_list.append(np.full(len(pi), a, dtype=np.int32))
        B_list.append(pos[pi].astype(np.int32))
        C_list.append(neg[ni].astype(np.int32))

    if not A_list:
        print(f"  ε={eps:.2f}: 0 件")
        continue

    A_all = np.concatenate(A_list)
    B_all = np.concatenate(B_list)
    C_all = np.concatenate(C_list)
    n_total = len(A_all)

    # 均一サブサンプル
    if n_total > N_SAMPLE:
        idx = RNG.choice(n_total, N_SAMPLE, replace=False)
        A_s, B_s, C_s = A_all[idx], B_all[idx], C_all[idx]
    else:
        A_s, B_s, C_s = A_all, B_all, C_all

    cap_gap = cap_sim[A_s, B_s] - cap_sim[A_s, C_s]

    # Translation 正解率
    tb = trans_cap[A_s, B_s]
    tc = trans_cap[A_s, C_s]
    trans_correct = np.where(tb > tc, 1.0, np.where(tb == tc, 0.5, 0.0)).astype(np.float32)
    trans_acc = float(trans_correct.mean())

    # EmCom 正解率
    eb = emcom_mat[A_s, B_s]
    ec = emcom_mat[A_s, C_s]
    emcom_correct = np.where(eb > ec, 1.0, np.where(eb == ec, 0.5, 0.0)).astype(np.float32)
    emcom_acc = float(emcom_correct.mean())

    # per-triplet 正誤を保存（ブートストラップ用）
    eps_tag = f"eps{int(eps * 100):02d}"
    np.savez(V3 / f"A_per_triplet_{eps_tag}.npz",
             trans_correct=trans_correct,
             emcom_correct=emcom_correct,
             A_s=A_s.astype(np.int32),
             B_s=B_s.astype(np.int32),
             C_s=C_s.astype(np.int32))

    print(f"  ε={eps:.2f}: {n_total:>12,}件 → {len(A_s):,}件使用  "
          f"cap_gap中央値={np.median(cap_gap):+.4f}  "
          f"Translation={trans_acc:.3f}  EmCom={emcom_acc:.3f}")
    results.append((eps, n_total, len(A_s),
                    float(np.median(cap_gap)), trans_acc, emcom_acc))

print("\n=== Table 4 サマリー ===")
print(f"{'ε':>6}  {'利用可能件数':>14}  {'使用':>8}  "
      f"{'cap_gap中央値':>14}  {'Translation':>12}  {'EmCom':>8}")
for eps, n_tot, n_use, gap, tr, em in results:
    print(f"  {eps:.2f}  {n_tot:>14,}  {n_use:>8,}  "
          f"{gap:>+14.4f}  {tr:>12.1%}  {em:>8.1%}")

if len(results) >= 2:
    loose = results[-1]
    tight = results[0]
    print()
    print(f"Translation: {loose[4]:.1%} → {tight[4]:.1%}  "
          f"({(tight[4]-loose[4])*100:+.1f} pp)")
    print(f"EmCom      : {loose[5]:.1%} → {tight[5]:.1%}  "
          f"({(tight[5]-loose[5])*100:+.1f} pp)")
