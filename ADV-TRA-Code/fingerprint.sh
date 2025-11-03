PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 \
LOGFILE=logs/$(date +"%Y%m%d_%H%M%S")_trajectory.log && \
{
  echo "========== ADV-TRA RUN =========="
  echo "Timestamp: $(date)"
  echo "Parameters:"
  echo "  --length 8"
  echo "  --max_iteration 500"
  echo "  --initial_stepsize 0.2"
  echo "  --tra_lr 0.01"
  echo "  --factor_lc 0.90"
  echo "  --factor_re 0.995"
  echo "  --threshold 0.5"
  echo "  --TIL_Checkpoint"
  echo "================================="
  echo
} | tee "$LOGFILE" && \
python -u main.py \
  --length 8 \
  --max_iteration 500 \
  --num_trajectories 100 \
  --tra_classes 10 \
  --initial_stepsize 0.2 \
  --tra_lr 0.01 \
  --factor_lc 0.90 \
  --factor_re 0.995 \
  --threshold 0.5 \
  --device cuda:2 \
  --model_path "./til" \
  --fingerprint_path "./til" \
  2>&1 | tee -a "$LOGFILE"