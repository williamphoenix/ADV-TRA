# -*- coding: utf-8 -*-

import torch
import torchvision
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from utils.models import wideresnet
import copy
import torch.nn as nn



def build_model(args):

    if args.dataset == "cifar10":
        model = torchvision.models.resnet18(num_classes=args.num_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  
        model.maxpool = nn.MaxPool2d(1, 1, 0) 
        
    elif args.dataset == "cifar100":
        model = wideresnet()
        
    elif args.dataset == "ImageNet":
        model = torchvision.models.resnet50(num_classes=args.num_classes, 
                                            weights=torchvision.models.ResNet50_Weights.DEFAULT)
    
    return model



def train_model(model, args):
    
    train_ldr = DataLoader(TensorDataset(torch.load('./results/' + args.dataset + '/allocated_data/data_log.pth')["X_train"],
                                         torch.load('./results/' + args.dataset + '/allocated_data/data_log.pth')["y_train"]), 
                           batch_size = args.batch_size, shuffle=True)

    test_ldr = DataLoader(TensorDataset(torch.load('./results/' + args.dataset + '/allocated_data/data_log.pth')["X_remain"],
                                         torch.load('./results/' + args.dataset + '/allocated_data/data_log.pth')["y_remain"]), 
                           batch_size = args.batch_size, shuffle=False)
    
    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_lr,
                                momentum=0.9,
                                weight_decay = 5e-4,
                                nesterov = True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[60, 120, 160], 
                                                     gamma=0.2)
    
    # train the ori_model
    logs = {}
    loss_record = []
    acc_record = []
    test_acc_record = []
    model.to(args.device)
    for epoch in range(args.epochs):
        print("epoch=",epoch)
        model.train()
        loss_meter = 0
        acc_meter = 0
        runcount = 0
        
        for batch_idx, (x, y) in enumerate(train_ldr):
            optimizer.zero_grad()
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            runcount += x.size(0)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            
            pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            acc_meter += pred.eq(y.view_as(pred)).sum().item()
            loss_meter += (loss.item()*x.size(0))

        loss_meter /= runcount
        acc_meter /= (runcount/100)
            
        loss_record.append(loss_meter)
        acc_record.append(acc_meter)
        print("Training acc:", acc_meter)
        
        # adjust the scheduler 
        scheduler.step()
        
        # calculate test acc
        test_acc_meter = test_acc_dataldr(model, args.device, test_ldr)
        test_acc_record.append(test_acc_meter)
        print("test acc:", test_acc_meter)
        
    logs["loss_record"] = loss_record
    logs["acc_record"] = acc_record
    logs["test_acc_record"] = test_acc_record
        
    w = copy.deepcopy(model.state_dict())
    model_dir = args.model_path + '/' + args.dataset + '/'
    
    torch.save(w, model_dir + 'source_model.pth')
    torch.save(logs, model_dir + 'source_model_logs.pth')
    
    return model



def test_acc_dataldr(model, device, data_ldr):
    test_acc_meter = 0
    runcount = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_ldr):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            test_acc_meter += pred.eq(target.view_as(pred)).sum().item()
            runcount += data.size(0)

    test_acc_meter /= (runcount/100)
    return test_acc_meter

    



    





