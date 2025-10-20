# -*- coding: utf-8 -*-

import argparse
import torch
import os
from utils.data_process import allocate_data
import numpy as np
import random 
from utils.adv_gen import generate_trajectory, verify_trajectory
from utils.utils import build_model, train_model


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed) 
random.seed(seed)  
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

has_split = True



def parse_args():
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument("--device", type=str, default="cuda:0", help="Which GPU to use (CPU or GPU)")
    
    # source model
    parser.add_argument("--dataset", type=str, default="cifar10", help="Which dataset to evaluate (cifar10, cifar100, or ImageNet)")
    parser.add_argument("--data_path", type=str, default="./results", help="The path to store the allocated data")
    parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the dataset") 
    parser.add_argument("--num_train", type=int, default=50000, help="The number of training data for the source model")
    parser.add_argument("--num_attack", type=int, default=5000, help="The number of data for lauching removal attacks")
    parser.add_argument("--initial_lr", type=float, default=0.1, help="Initial learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs of source model training") 
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size for each iteration")
    parser.add_argument("--model_path", type=str, default="./results", help="The path where the source model is saved")

    # trajectory config
    parser.add_argument("--num_trajectories", type=int, default=100, help="The number of trajectories") 
    parser.add_argument("--max_iteration", type=int, default=300, help="The maximum number of iterations") 
    parser.add_argument("--length", type=int, default=4, help="The length of bilateral trajectories")
    parser.add_argument("--factor_lc", type=float, default=0.9, help="Length control factor to adjust the step size of each step")
    parser.add_argument("--factor_re", type=float, default=0.95, help="Reduction factor")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for fingerprint determination")
    parser.add_argument("--initial_stepsize", type=float, default=0.05, help="Initial value for the step size")
    parser.add_argument("--tra_lr", type=float, default=0.05, help="The learning rate for optimizing the trajectories")
    parser.add_argument("--fingerprint_path", type=str, default="./results", help="The path where the fingerprinting samples are saved")
    parser.add_argument("--tra_classes", type=int, default=10, help="The number of classes traversed by the trajectory")
    parser.add_argument("--suspect_path", type=str, default="./suspect_models/model.pth", help="The path of the suspect model")
    
    return parser.parse_args()



def main(args):
    # normalize dataset casing & num_classes
    args.dataset = args.dataset.lower()
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "imagenet":
        args.num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ensure allocated data exists (CIFAR path)
    data_log_path = os.path.join(args.data_path, args.dataset, "allocated_data", "data_log.pth")
    if not os.path.exists(data_log_path):
        print(f"[info] Allocated data not found at {data_log_path}. Running allocate_data(args).")
        allocate_data(args)
    assert os.path.exists(data_log_path), f"Missing {data_log_path}"


    # build + load CIFAR-10 model
    #model = build_model(args).to(args.device)
    #train_model(model, args)

    # run trajectories + verification
    generate_trajectory(args)
    #assert os.path.exists(args.suspect_path), f"Suspect model not found: {args.suspect_path}"
    #verify_trajectory(args)





if __name__ == '__main__':
    args = parse_args()
    # better determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)





