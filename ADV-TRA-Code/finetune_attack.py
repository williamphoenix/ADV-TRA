#!/usr/bin/env python3
"""
finetune_attack_remain.py

A finetuning attack script which **only** uses the reserved attacker split
(X_remain, y_remain) produced by allocate_data(...).

Usage examples:
  python finetune_attack_remain.py --attack FTLL --source ./results/cifar10/source_model.pth --out_dir ./results/stolen --device cuda:2
  python finetune_attack_remain.py --attack RTAL --source ./results/cifar10/source_model.pth --out_dir ./results/stolen --device cuda:2
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import build_model

# ---------------------------
# Data loader: ONLY X_remain / y_remain
# ---------------------------
def load_aux_dataset_from_remain(args):
    cache_path = os.path.join(args.data_path, args.dataset, "allocated_data", "data_log.pth")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Allocated data not found at {cache_path}. Run allocate_data(args) first.")
    cache = torch.load(cache_path, map_location="cpu")

    if "X_remain" not in cache or "y_remain" not in cache:
        raise KeyError(f"data_log.pth missing X_remain/y_remain keys. Keys present: {list(cache.keys())}")

    X_remain = cache["X_remain"]
    y_remain = cache["y_remain"]

    if X_remain is None or len(X_remain) == 0:
        raise ValueError("X_remain is empty — cannot run attacker training with no data.")

    # print a short sanity summary
    print(f"[info] Loaded attacker dataset (remain): X_remain={tuple(X_remain.shape)}  y_remain={tuple(y_remain.shape)}")

    loader = DataLoader(
        TensorDataset(X_remain, y_remain),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=False,
    )
    return loader

# ---------------------------
# Utilities
# ---------------------------
def device_for(args):
    if torch.cuda.is_available() and "cuda" in args.device:
        return torch.device(args.device)
    return torch.device("cpu")

def robust_load_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)

def find_last_linear(model):
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last = (name, module)
    return last

def reinit_last_layer(model):
    pair = find_last_linear(model)
    if pair is None:
        raise RuntimeError("No nn.Linear layer found to reinitialize.")
    _, lin = pair
    nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)

def set_trainable_layers(model, attack_type):
    # default all trainable
    for p in model.parameters():
        p.requires_grad = True

    last_pair = find_last_linear(model)
    last_name, last_lin = (last_pair if last_pair is not None else (None, None))

    if attack_type == "FTLL":
        for p in model.parameters():
            p.requires_grad = False
        if last_lin is None:
            raise RuntimeError("FTLL: couldn't find final linear layer.")
        for p in last_lin.parameters():
            p.requires_grad = True

    elif attack_type == "FTAL":
        pass

    elif attack_type == "RTLL":
        if last_lin is None:
            raise RuntimeError("RTLL: couldn't find final linear layer.")
        reinit_last_layer(model)
        for p in model.parameters():
            p.requires_grad = False
        for p in last_lin.parameters():
            p.requires_grad = True

    elif attack_type == "RTAL":
        if last_lin is None:
            raise RuntimeError("RTAL: couldn't find final linear layer.")
        reinit_last_layer(model)
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # set BN/Dropout to eval if their module params are all frozen (stability)
    for _, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if params and not any(p.requires_grad for p in params):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
                module.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable

# ---------------------------
# Training loop
# ---------------------------
def finetune(args):
    args.dataset = args.dataset.lower()
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "imagenet":
        args.num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    device = device_for(args)
    print(f"[info] using device: {device}")

    model = build_model(args).to(device)
    robust_load_state(model, args.source)
    params = set_trainable_layers(model, args.attack)

    # paper-like lr choices: FT* small, RT* larger
    if args.attack in ("FTLL", "FTAL"):
        lr = 1e-3
    else:
        lr = 1e-2

    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)

    train_loader = load_aux_dataset_from_remain(args)

    # training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        scheduler.step()
        avg_loss = running_loss / total if total else 0.0
        acc = 100.0 * correct / total if total else 0.0
        print(f"[{args.attack}] Epoch {epoch+1:03d}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.5f}  loss={avg_loss:.4f}  acc={acc:.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"{args.attack}_stolen.pth")
    torch.save({"model_state": model.state_dict(), "attack": args.attack, "args": vars(args)}, save_path)
    print(f"[info] Saved {args.attack} stolen-model to: {save_path}")
    return save_path

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to source model .pth")
    parser.add_argument("--out_dir", type=str, default="./results/stolen", help="Where to save the stolen model")
    parser.add_argument("--attack", type=str, choices=["FTLL", "FTAL", "RTLL", "RTAL"], required=True)
    parser.add_argument("--data_path", type=str, default="./results", help="Base path containing <dataset>/allocated_data/data_log.pth")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    finetune(args)

if __name__ == "__main__":
    main()