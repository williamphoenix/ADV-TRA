#!/usr/bin/env python3
# finetune_attack.py
# Usage examples:
#  python finetune_attack.py --attack FTLL --source ./results/cifar10/source_model.pth --out_dir ./results/stolen
#  python finetune_attack.py --attack RTAL --source ./results/cifar10/source_model.pth --out_dir ./results/stolen

import argparse
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from utils.utils import build_model  # assumes same project layout as your repo

def load_aux_dataset(args):
    # paper uses a reserved 'remain' split for attacker data
    cache_path = os.path.join(args.data_path, args.dataset, "allocated_data", "data_log.pth")
    cache = torch.load(cache_path)
    # attacker uses X_remain / y_remain per paper
    X = cache["X_remain"]
    y = cache["y_remain"]
    return DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

def reinit_last_layer(model):
    """
    Reinitialize the final linear layer weights and bias.
    Handles common patterns (resnet: model.fc, torchvision models: classifier, linear).
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        nn.init.kaiming_normal_(model.fc.weight, nonlinearity='linear')
        if model.fc.bias is not None:
            nn.init.zeros_(model.fc.bias)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        nn.init.kaiming_normal_(model.classifier.weight, nonlinearity='linear')
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
    else:
        # fallback: search for last linear module
        last_lin = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                last_lin = module
                break
        if last_lin is not None:
            nn.init.kaiming_normal_(last_lin.weight, nonlinearity='linear')
            if last_lin.bias is not None:
                nn.init.zeros_(last_lin.bias)
        else:
            raise RuntimeError("Couldn't find final linear layer to reinit on model")

def set_trainable_layers(model, attack_type):
    """
    Returns list of model parameters to pass to optimizer according to attack_type:
      FTLL: freeze all except last layer
      FTAL: all layers trainable
      RTLL: reinit last layer then train only last layer
      RTAL: reinit all layers then train all layers (paper: re-init then train all)
    """
    if attack_type == "FTLL":
        for p in model.parameters():
            p.requires_grad = False
        # enable last linear
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            # last linear fallback
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name.count(".") == 0:
                    for p in module.parameters():
                        p.requires_grad = True
    elif attack_type == "FTAL":
        for p in model.parameters():
            p.requires_grad = True
    elif attack_type == "RTLL":
        reinit_last_layer(model)
        # freeze others
        for p in model.parameters():
            p.requires_grad = False
        # last layer trainable
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            # fallback enable last linear found earlier
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    for p in module.parameters():
                        p.requires_grad = True
                    break
    elif attack_type == "RTAL":
        # paper describes "re-initialize the last layer before FTAL" for RTAL,
        # but they also say "re-initialize the last layer before FTAL" — we'll reinit last layer then fine-tune all params.
        reinit_last_layer(model)
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    params = [p for p in model.parameters() if p.requires_grad]
    return params

def finetune(args):
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else "cpu")
    # build model skeleton with same architecture and num_classes
    args.dataset = args.dataset.lower()
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "imagenet":
        args.num_classes = 1000
    else:
        raise ValueError("Unknown dataset")

    model = build_model(args)
    model = model.to(device)

    # load source weights
    src = torch.load(args.source, map_location="cpu")
    # typical source saved is state_dict only; try to be flexible
    if isinstance(src, dict) and ("model_state" in src or "state_dict" in src):
        state = src.get("model_state", src.get("state_dict", src))
        model.load_state_dict(state)
    elif isinstance(src, dict):
        # maybe it is state_dict already
        try:
            model.load_state_dict(src)
        except Exception as e:
            # try to find inner keys
            keys = list(src.keys())
            # if saved as plain weights tensor dict, attempt load
            model.load_state_dict(src)
    else:
        model.load_state_dict(src)

    # configure attack-specific trainable params
    params = set_trainable_layers(model, args.attack)

    # choose lr per paper:
    if args.attack in ("FTLL", "FTAL"):
        lr = 0.001
    else:  # RTLL, RTAL
        lr = 0.01

    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)

    # loader: attacker uses reserved 'remain' split from allocate_data
    train_loader = load_aux_dataset(args)

    # training loop (50 epochs per paper)
    model.train()
    for epoch in range(args.epochs):
        total = 0
        correct = 0
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = nn.functional.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        scheduler.step()
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"[{args.attack}] Epoch {epoch+1}/{args.epochs} loss={running_loss/total:.4f} acc={acc:.2f}%")

    # save stolen model copy
    os.makedirs(args.out_dir, exist_ok=True)
    save_name = f"{args.attack}_stolen.pth"
    save_path = os.path.join(args.out_dir, save_name)
    torch.save({"model_state": model.state_dict(), "attack": args.attack, "args": vars(args)}, save_path)
    print(f"Saved {args.attack} stolen-model to: {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to source model .pth (source_model.pth)")
    parser.add_argument("--out_dir", type=str, default="./results/stolen", help="Directory to save stolen model")
    parser.add_argument("--attack", type=str, choices=["FTLL","FTAL","RTLL","RTAL"], required=True)
    parser.add_argument("--data_path", type=str, default="./results", help="Base results/data path (must contain allocated_data)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10|cifar100|imagenet")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)  # paper uses 50
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    finetune(args)