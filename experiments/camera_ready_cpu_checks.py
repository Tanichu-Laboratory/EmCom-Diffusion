"""Camera-ready CPU-only checks for ICONIP 2026 #1069.

(1) R2: is tau=0.70 really the 99th percentile of CLIP ViT-L/14 GT-GT cosine
    similarity?  The claim exists only as prose in REPRODUCIBILITY.md.
(2) R1-3: system-level comparison across the controlled capacity-grid
    languages -- does corpus-level TopSim recover the same ordering as
    EmCom-Diffusion?
(3) R1-4: normalized improvement over the input-independent baseline.

No GPU, no writes outside outputs/camera_ready/.
"""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/workspace/project/omoto/emlan-image-generate")
V2 = ROOT / "outputs" / "experiments_v2"
V3 = ROOT / "experiments_v3" / "outputs"
OUT = ROOT / "outputs" / "camera_ready"
OUT.mkdir(parents=True, exist_ok=True)

results = {}


def sep(title):
    print(f"\n{'='*74}\n{title}\n{'='*74}")


# ─────────────────────────────────────────────────────────────────────────
# (1) tau = 0.70 -- which percentile is it really?
# ─────────────────────────────────────────────────────────────────────────
sep("(1) tau = 0.70 : percentile of CLIP ViT-L/14 GT-GT cosine similarity")

emb = np.load(V2 / "cache_gt_clip_vitl14.npy").astype(np.float32)
n = emb.shape[0]
norms = np.linalg.norm(emb, axis=1)
print(f"  cache shape={emb.shape}  ||v|| mean={norms.mean():.6f} "
      f"min={norms.min():.6f} max={norms.max():.6f}")
# normalise defensively (bootstrap_paper.py assumes pre-normalised)
embn = emb / np.clip(norms[:, None], 1e-12, None)
was_normalised = bool(np.allclose(norms, 1.0, atol=1e-3))
print(f"  already L2-normalised: {was_normalised}")

sim = embn @ embn.T
iu = np.triu_indices(n, k=1)          # unordered pairs, self excluded
pairs = sim[iu]
del sim

TAU = 0.70
frac_above = float((pairs > TAU).mean())
pct_of_tau = float((pairs <= TAU).mean() * 100.0)
print(f"  unordered GT-GT pairs      : {pairs.size:,}")
print(f"  mean={pairs.mean():.4f}  std={pairs.std():.4f}  "
      f"min={pairs.min():.4f}  max={pairs.max():.4f}")
print(f"  fraction with sim > 0.70   : {frac_above:.6f}  ({frac_above*100:.4f}%)")
print(f"  --> tau=0.70 is the {pct_of_tau:.3f}th percentile")
for q in (90, 95, 97.5, 99, 99.5, 99.9):
    print(f"  {q:>5}th percentile         : {np.percentile(pairs, q):.4f}")
print(f"  VERDICT: the paper claims 99th percentile; measured "
      f"{pct_of_tau:.2f}th "
      f"({'CONSISTENT' if 98.5 <= pct_of_tau <= 99.5 else 'INCONSISTENT'})")

results["tau_check"] = dict(
    n_images=int(n), n_pairs=int(pairs.size), tau=TAU,
    frac_above_tau=frac_above, percentile_of_tau=pct_of_tau,
    cache_was_l2_normalised=was_normalised,
    percentiles={str(q): float(np.percentile(pairs, q))
                 for q in (90, 95, 97.5, 99, 99.5, 99.9)},
)
del pairs


# ─────────────────────────────────────────────────────────────────────────
# (2) R1-3 : system-level metric agreement across controlled languages
# ─────────────────────────────────────────────────────────────────────────
sep("(2) R1-3 : do corpus-level TopSim and EmCom-Diffusion agree across the "
    "capacity grid?")


def read_single_row_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


# EmCom-Diffusion (DINOv2) per capacity-grid condition
emcom = {}
for p in sorted(glob.glob(str(ROOT / "outputs" / "phase3_table1_K*_V*.csv"))):
    for r in read_single_row_csv(p):
        if r["input_type"] == "EC":
            emcom[r["setting"]] = dict(
                dino=float(r["dino_score"]),
                clip_img=float(r["clip_score_image"]),
                fid=float(r["fid"]),
            )
# K8_V256 lives in pilot_metrics_dino.csv (setting A)
for r in read_single_row_csv(ROOT / "outputs" / "pilot_metrics_dino.csv"):
    if r["setting"] == "A" and r["input_type"] == "EC":
        emcom["K8_V256"] = dict(dino=float(r["dino_score"]),
                                clip_img=float(r["clip_score_image"]),
                                fid=float(r["fid"]))

# corpus-level TopSim
topsim = {}
for r in read_single_row_csv(V2 / "topsim_capacity_grid.csv"):
    cond = r["condition"]
    key = "K8_V256" if cond == "ec_ep29" else cond.replace("ec_", "")
    topsim[key] = float(r["topsim_rho"])

shared = sorted(set(emcom) & set(topsim))
print(f"  conditions with both metrics: {len(shared)} -> {shared}")
print(f"\n  {'condition':<12} {'K':>3} {'V':>5} {'TopSim rho':>11} "
      f"{'EmCom(DINO)':>12} {'CLIP-img':>9} {'FID':>7}")
rows = []
for c in shared:
    K = int(c.split("_")[0][1:])
    V = int(c.split("_")[1][1:])
    rows.append((c, K, V, topsim[c], emcom[c]["dino"],
                 emcom[c]["clip_img"], emcom[c]["fid"]))
for c, K, V, ts, dn, ci, fid in sorted(rows, key=lambda r: -r[4]):
    print(f"  {c:<12} {K:>3} {V:>5} {ts:>11.4f} {dn:>12.4f} "
          f"{ci:>9.4f} {fid:>7.2f}")

ts_v = np.array([r[3] for r in rows])
dn_v = np.array([r[4] for r in rows])
ci_v = np.array([r[5] for r in rows])

rho_d, p_d = spearmanr(ts_v, dn_v)
rho_c, p_c = spearmanr(ts_v, ci_v)
pr_d, pp_d = pearsonr(ts_v, dn_v)
print(f"\n  Spearman TopSim vs EmCom(DINOv2) : rho={rho_d:.4f}  p={p_d:.4f}  "
      f"(n={len(rows)})")
print(f"  Spearman TopSim vs EmCom(CLIP-img): rho={rho_c:.4f}  p={p_c:.4f}")
print(f"  Pearson  TopSim vs EmCom(DINOv2) : r={pr_d:.4f}  p={pp_d:.4f}")
print("\n  Reading: a moderate positive rho means the two metrics order these\n"
      "  controlled languages similarly but are NOT interchangeable, which is\n"
      "  exactly the 'related but non-identical constructs' framing R1 asks for.")

results["system_level"] = dict(
    conditions=[dict(condition=c, K=K, V=V, topsim_rho=ts,
                     emcom_dino=dn, emcom_clip_img=ci, fid=fid)
                for c, K, V, ts, dn, ci, fid in rows],
    spearman_topsim_vs_dino=dict(rho=float(rho_d), p=float(p_d)),
    spearman_topsim_vs_clipimg=dict(rho=float(rho_c), p=float(p_c)),
    pearson_topsim_vs_dino=dict(r=float(pr_d), p=float(pp_d)),
    n=len(rows),
)


# ─────────────────────────────────────────────────────────────────────────
# (3) R1-4 : normalized improvement over the input-independent baseline
# ─────────────────────────────────────────────────────────────────────────
sep("(3) R1-4 : normalized score  (EL - Random) / (SD-NL - Random)")

b1 = {r["condition"]: r for r in read_single_row_csv(V2 / "b1_multi_encoder_n5000.csv")}
pil = {(r["setting"], r["input_type"]): r
       for r in read_single_row_csv(ROOT / "outputs" / "pilot_metrics_dino.csv")}

enc = [("CLIP-img", "clip_mean"), ("DINOv2", "dino_mean"),
       ("SigLIP", "siglip_mean"), ("MAE", "mae_mean")]
print(f"  {'metric':<10} {'EL':>8} {'Random':>8} {'Fixed':>8} {'SD-NL':>8} "
       f"{'delta':>9} {'norm':>7}")
norm_rows = []
for name, col in enc:
    el = float(b1["A"][col]); rnd = float(b1["A_random"][col])
    fx = float(b1["A_fixed"][col]); nl = float(b1["NL_ft"][col])
    delta = el - rnd
    norm = delta / (nl - rnd) if (nl - rnd) != 0 else float("nan")
    print(f"  {name:<10} {el:>8.4f} {rnd:>8.4f} {fx:>8.4f} {nl:>8.4f} "
          f"{delta:>+9.4f} {norm:>7.3f}")
    norm_rows.append(dict(metric=name, el=el, random=rnd, fixed=fx, sd_nl=nl,
                          delta=delta, normalized=norm))

# CLIP-text from pilot_metrics_dino.csv
el = float(pil[("A", "EC")]["clip_score_text"])
rnd = float(pil[("A", "Random")]["clip_score_text"])
fx = float(pil[("A", "Fixed")]["clip_score_text"])
nl = float(pil[("SD-NL-ft", "NL")]["clip_score_text"])
delta, norm = el - rnd, (el - rnd) / (nl - rnd)
print(f"  {'CLIP-text':<10} {el:>8.4f} {rnd:>8.4f} {fx:>8.4f} {nl:>8.4f} "
      f"{delta:>+9.4f} {norm:>7.3f}")
norm_rows.append(dict(metric="CLIP-text", el=el, random=rnd, fixed=fx,
                      sd_nl=nl, delta=delta, normalized=norm))

three = [r["normalized"] for r in norm_rows
         if r["metric"] in ("CLIP-img", "DINOv2", "SigLIP")]
print(f"\n  Three perceptual encoders: normalized = "
      f"{min(three):.3f} - {max(three):.3f} (mean {np.mean(three):.3f})")
print("  Note: DINOv2 has the largest RAW gap (11x Random) but the LOWEST\n"
      "  normalized value -- raw ratios overstate the separation.")

results["normalized"] = norm_rows

# also: capacity ablation (R1-2 evaluator-capacity answer, already on disk)
sep("(3b) R1-2 : trainable-module ablation already on disk (n=5,000, with CI)")
print(f"  {'setting':<10} {'trainable':<44} {'DINOv2 EC':>18} {'Random':>9}")
desc = {"A": "EC embedding only",
        "B": "EC embedding + U-Net LoRA r16 (= SD-NL set)",
        "C": "EC emb + LoRA r16 + CLIP top-4 (> SD-NL set)",
        "SD-NL-ft": "U-Net LoRA r16 only (CLIP frozen)"}
cap = []
for s in ("A", "B", "C", "SD-NL-ft"):
    it = "NL" if s == "SD-NL-ft" else "EC"
    r = pil[(s, it)]
    d = float(r["dino_score"]); lo = float(r["dino_ci_lo"]); hi = float(r["dino_ci_hi"])
    rr = pil[(s, "Random")]["dino_score"] if (s, "Random") in pil else "-"
    print(f"  {s:<10} {desc[s]:<44} {d:.4f} [{lo:.4f},{hi:.4f}]  {rr:>9}")
    cap.append(dict(setting=s, trainable=desc[s], dino=d, ci=[lo, hi],
                    random_dino=(float(rr) if rr != "-" else None)))
results["capacity_ablation"] = cap

with open(OUT / "camera_ready_cpu_checks.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[saved] {OUT / 'camera_ready_cpu_checks.json'}")
