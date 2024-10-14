# -*- coding: utf-8 -*-
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_data(dataset, data_root):

    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                              ])  
       

        train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform
                                               )


        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=True,
                                                transform=transform
                                                )
        all_dataset = torch.utils.data.dataset.ConcatDataset([train_set, test_set])
    
    if dataset == 'cifar100':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(15),
                                        transforms.ToTensor()
                                              ])  

        train_set = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform
                                               )


        test_set = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=True,
                                                transform=transform
                                                )
        all_dataset = torch.utils.data.dataset.ConcatDataset([train_set, test_set])
                
    if dataset == "ImageNet":
        test_set = torchvision.datasets.ImageNet(data_root,
                                                split = 'val',
                                                transform=transforms.Compose([transforms.Resize(256),
                                                                              transforms.CenterCrop(224),
                                                                              transforms.ToTensor()
                                                                              ])
                                                )
        return test_set

    return all_dataset








class DatasetSplit(Dataset):
    def __init__(self, dataset, num_data):
        self.dataset = dataset
        idxs = np.arange(len(dataset))
        self.idxs = np.random.choice(idxs,num_data,replace=False)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



def allocate_data(args):
    dataset = get_data(args.dataset, data_root="./Data")
    list_loader = list(dataset)
    
    if len(list_loader) < args.num_attack + args.num_train:
        raise Exception("Data used for training and attack is in excess of total data")

    X = []
    y = []
    for data in list_loader:
        X.append(data[0].unsqueeze(0))
        y.append(data[1])
    X = torch.cat(X, axis=0)
    y = torch.tensor(y)
    
    # shuffle the dataset
    if args.shuffle == True:
        idx = torch.randperm(len(list_loader))
    else:
        idx = torch.arange(len(list_loader))
        
    X = X[idx]
    y = y[idx]
    
    X_train = X[0:args.num_train]
    X_remain = X[args.num_train:]
    y_train = y[0:args.num_train]
    y_remain = y[args.num_train:]
    
    X_attack = X_remain[0:args.num_attack]
    X_remain = X_remain[args.num_attack:]
    y_attack = y_remain[0:args.num_attack]
    y_remain = y_remain[args.num_attack:]
    
    # allocate data
    data_log = {}
    data_log["X_train"] = X_train
    data_log["y_train"] = y_train
    data_log["X_attack"] = X_attack
    data_log["y_attack"] = y_attack
    data_log["X_remain"] = X_remain
    data_log["y_remain"] = y_remain
    # save data
    data_dir = args.data_path + '/' + args.dataset + '/allocated_data'
    os.makedirs(data_dir, exist_ok=True)
    torch.save(data_log, data_dir + '/data_log.pth')










