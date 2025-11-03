PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 \
LOGFILE=logs/$(date +"%Y%m%d_%H%M%S")_trajectory.log && \

{
  echo "========== VERIFICATION RUN =========="
  echo "Timestamp: $(date)"
  echo "Parameters:"
  echo "RTAL attack"
  echo "================================="
  echo
} | tee "$LOGFILE" && \
python -u main.py \
  --suspect_path ./results/stolen/RTAL/RTAL_stolen.pth \
  --device cuda:2 \
  --length 8 \
  --num_trajectories 47 \
  --dataset cifar10 \
  --fingerprint_path ./ver \
  2>&1 | tee -a "$LOGFILE"