python finetune_attack.py \
  --attack RTAL \
  --source ./results/cifar10/source_model.pth \
  --out_dir ./results/stolen/RTAL \
  --dataset cifar10 \
  --data_path ./results \
  --device cuda:2 \
  --epochs 50 \
  --batch_size 128