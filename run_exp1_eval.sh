#!/usr/bin/env bash
# Experiment 1, stage 2: generate + score the four evaluator seeds.
#
# The four conditions run CONCURRENTLY, one per GPU. This is only safe because
# _make_latents() in diffusion/inference.py seeds each sample from its own
# global index, so the images for a given index do not depend on batch size,
# batch boundaries, or which device produced them (unit-tested).
#
# Why pilot_A is regenerated: the published pilot_A images were produced before
# the initial-latent seeding bug was fixed, so their sampling noise came from an
# unseeded global RNG. To isolate evaluator-TRAINING variance (what R1-2 asks
# for) every seed must share identical generation noise, so all four are
# regenerated with --seed 42 --latent_seed 42. The original pilot_A images are
# left untouched in outputs/generated_images/A.

set -euo pipefail

export COCO_ROOT=/workspace/project/datas/dataset/coco2017
export PROJECT_DIR=/workspace/project/omoto/emlan-image-generate
export TOKENIZERS_PARALLELISM=false

REPO=/workspace/project/omoto/emcom-diffusion
CR=$PROJECT_DIR/outputs/camera_ready
GEN=$CR/exp1_generated
LOGDIR=$CR/exp1_logs
EC_JSON=$PROJECT_DIR/outputs/ec_corpus_blip_ref_ep29.json
N=5000

mkdir -p "$GEN" "$LOGDIR"
cd "$REPO"

LABELS=(seed42 seed0 seed1 seed2)
CKPT_DIRS=(pilot_A pilot_A_seed0 pilot_A_seed1 pilot_A_seed2)

PIDS=()
for i in 0 1 2 3; do
  LABEL=${LABELS[$i]}
  CKPT=$PROJECT_DIR/checkpoints/${CKPT_DIRS[$i]}/step_010000

  if [ ! -d "$CKPT" ]; then
    echo "[exp1-eval] MISSING $CKPT -- skipping $LABEL"; continue
  fi
  if [ -d "$GEN/$LABEL" ] && [ "$(ls -1 "$GEN/$LABEL" 2>/dev/null | wc -l)" -ge "$N" ]; then
    echo "[exp1-eval] $LABEL already has $N images -- skipping"; continue
  fi

  echo "[exp1-eval] launching $LABEL on GPU $i at $(date -Is)"
  CUDA_VISIBLE_DEVICES=$i python -m diffusion.inference \
      --ckpts "$CKPT" --labels "$LABEL" \
      --n_samples $N --input_type ec \
      --seed 42 --latent_seed 42 \
      --n_steps 50 --guidance 7.5 --batch_size 16 \
      --ec_json "$EC_JSON" \
      --out_img_dir "$GEN" \
      --out_html "$CR/exp1_${LABEL}.html" \
      > "$LOGDIR/gen_${LABEL}.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then echo "[exp1-eval] a generation job (pid $p) FAILED"; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] || { echo "[exp1-eval] aborting: see $LOGDIR/gen_*.log"; exit 1; }

echo "[exp1-eval] all generation done at $(date -Is); image counts:"
for L in "${LABELS[@]}"; do
  echo "  $L: $(ls -1 "$GEN/$L" 2>/dev/null | wc -l)"
done

echo "[exp1-eval] ===== scoring at $(date -Is) ====="
python experiments/eval_similarity.py \
    --ec_json "$EC_JSON" \
    --gen_dir "$GEN" \
    --conditions seed42 seed0 seed1 seed2 \
    --encoders clip dino siglip \
    --n_samples $N --seed 42 --n_bootstrap 1000 \
    --out_csv "$CR/exp1_seed_variance.csv" \
    > "$LOGDIR/score_all.log" 2>&1

echo "[exp1-eval] DONE at $(date -Is). scores:"
cat "$CR/exp1_seed_variance.csv"
