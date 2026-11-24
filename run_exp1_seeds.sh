#!/usr/bin/env bash
# Experiment 1 (reviewer R1-2): evaluator-training seed variance.
#
# Runs Setting A (EC token embedding only -- the Table 2 configuration) three
# more times with optimization seeds 0/1/2. Everything else is held identical
# to the published pilot_A run:
#   4-GPU DDP, batch_size 16 per GPU, grad_accumulation 1  -> effective batch 64
#   total_steps 10000, warmup 500, AdamW lr 1e-4 wd 1e-2, grad clip 1.0, fp16
# Runs are SEQUENTIAL so each one gets all 4 GPUs and therefore the exact same
# effective batch size as pilot_A. Running them in parallel on 1 GPU each would
# require grad_accumulation 4 to keep the budget equal and would be slower
# overall, so sequential is both faster and protocol-identical.
#
# Together with the published pilot_A (seed 42) this gives n=4 evaluator seeds.

set -euo pipefail

export COCO_ROOT=/workspace/project/datas/dataset/coco2017
export PROJECT_DIR=/workspace/project/omoto/emlan-image-generate
export TOKENIZERS_PARALLELISM=false

REPO=/workspace/project/omoto/emcom-diffusion
LOGDIR=$PROJECT_DIR/outputs/camera_ready/exp1_logs
mkdir -p "$LOGDIR"

cd "$REPO"

for SEED in 0 1 2; do
  RUN=pilot_A_seed${SEED}
  CKPT=$PROJECT_DIR/checkpoints/$RUN

  if [ -d "$CKPT/step_010000" ]; then
    echo "[exp1] $RUN already complete -- skipping"
    continue
  fi

  echo "[exp1] ===== $RUN starting at $(date -Is) ====="

  # sample VRAM every 60s so we can report peak memory (reviewer R1-5)
  ( while true; do
      echo "$(date +%s),$(nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits | tr '\n' ';')"
      sleep 60
    done ) > "$LOGDIR/${RUN}_vram.csv" 2>/dev/null &
  VRAM_PID=$!
  # make sure the sampler dies with this run even on failure
  trap 'kill $VRAM_PID 2>/dev/null || true' EXIT

  START=$(date +%s)
  # `accelerate` console script is not installed in this environment; the
  # library is, so invoke the launcher as a module.
  python -m accelerate.commands.launch \
      --config_file accelerate_configs/4gpu_ddp.yaml \
      -m diffusion.train \
      --config configs/diffusion_ec_embed_seed${SEED}.yaml \
      > "$LOGDIR/${RUN}_train.log" 2>&1
  END=$(date +%s)

  kill $VRAM_PID 2>/dev/null || true
  trap - EXIT

  PEAK=$(awk -F'[,;]' '{for(i=3;i<=NF;i+=2) if($i+0>m) m=$i+0} END{print m}' \
         "$LOGDIR/${RUN}_vram.csv" 2>/dev/null || echo "NA")
  echo "$RUN,$((END-START)),$PEAK" >> "$LOGDIR/cost.csv"
  echo "[exp1] ===== $RUN done at $(date -Is): $((END-START))s, peak ${PEAK} MiB ====="
done

echo "[exp1] ALL RUNS COMPLETE at $(date -Is)"
echo "[exp1] cost summary:"; cat "$LOGDIR/cost.csv" 2>/dev/null || true
