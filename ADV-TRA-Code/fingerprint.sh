PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 \
LOGFILE=logs/$(date +"%Y%m%d_%H%M%S")_trajectory.log && \
{
  echo "========== ADV-TRA RUN =========="
  echo "Timestamp: $(date)"
  echo "Parameters:"
  echo "  --length 6"
  echo "  --max_iteration 2000"
  echo "  --initial_stepsize 0.20"
  echo "  --tra_lr 0.20"
  echo "  --factor_lc 0.98"
  echo "  --factor_re 0.995"
  echo "  --threshold 0.5"
  echo "================================="
  echo
} | tee "$LOGFILE" && \
python -u main.py \
  --length 6 \
  --max_iteration 1000 \
  --initial_stepsize 0.20 \
  --tra_lr 0.20 \
  --factor_lc 0.98 \
  --factor_re 0.995 \
  --threshold 0.5 \
  --device cuda:2 \
  2>&1 | tee -a "$LOGFILE"