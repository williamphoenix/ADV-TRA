# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import os
import copy
from torch.utils.data import DataLoader
from utils.utils import build_model
from utils.data_process import get_data




def next_adv_sample(image, step_size, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Prevent images from going out of range
    location = ((image - step_size*sign_data_grad)>1.0) + ((image - step_size*sign_data_grad)<0)
    location = (1.0 - location*1.0)
    sign_data_grad = sign_data_grad*location
    perturbed_image = image - step_size*sign_data_grad

    # Return the perturbed image
    return perturbed_image



# Check that the required trajectories are generated
def capture(status, status2, stepsize_list):
    if status == 0 and status2 == 0:
        return stepsize_list
    else:
        return None


def generate_unilateral_tra(model, image, target, args, num_epoch=300, coe3=1, coe4=1, decay=0.90, lr=0.001):
    length = args.length
    half_length = int(length/2)
    stepsize_list = [ torch.tensor(args.initial_stepsize).to(args.device) - 
                     torch.tensor(0.002*i).to(args.device) for i in range(half_length)]

    best_stepsize_list = None
    
    for epoch in range(num_epoch):
        loss_all = torch.tensor(0.0).to(args.device)
        images_tra = []
        grad_list = []
        images_tra.append(image.detach().clone())
        
        process_image = images_tra[0]
        for i in range(half_length):
            model.eval()
            model.zero_grad()
            # Set requires_grad attribute of tensor
            process_image.requires_grad = True
            stepsize_list[i].requires_grad = False
            
            # Forward pass the data through the model
            output = model(process_image)
            # Calculate the loss
            loss = F.nll_loss(output, target)
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = process_image.grad.data.detach().clone()
            grad_list.append(data_grad)
            
            process_image.grad.zero_()
            
            # generate the next adversarial sample
            process_image = next_adv_sample(process_image, stepsize_list[i], data_grad)
            process_image = process_image.detach().clone()
            process_image.requires_grad = False
            images_tra.append(process_image.detach().clone())

        pred_list = []
        status1 = 0  # judge whether the other sample crosses the decision boundary
        status2 = 0  # judge whether the last sample crosses the decision boundary
        for j in range(half_length):
            if j == 0:
                adv_image = images_tra[0]
            else:
                adv_image = next_adv_sample(images_tra[j], stepsize_list[j], grad_list[j]) 
            stepsize_list[j].requires_grad = True

            # Re-classify the perturbed image
            output = model(adv_image)
            output = F.softmax(output,dim=1)
            final_pred = output.max(1, keepdim=True)[1]
            pred_list.append(final_pred[0][0].item())
            
            # update status1
            if j == half_length-1:
                status1 = 0
                if final_pred != target and status2 == 0:
                    status1 = 1
                    
               
                # record best step size and quit early
                best_stepsize_list = capture(status1, status2, stepsize_list)
                if epoch>num_epoch/4 and (best_stepsize_list != None):
                    print("target:", target)
                    print(status1, status2)
                    print(pred_list)
                    print('All requirements are satisfied, quit early!')
                    return best_stepsize_list
                    
            
            # update status2
            else: 
                if final_pred == target:
                    status2 = 1 | status2   

                # The step size decays close to the decision boundary
                loss_3 = (stepsize_list[j]*decay - stepsize_list[j+1])**2
                loss_all += loss_3*coe3

            # Ensure the step sizes are all greater than zero
            if stepsize_list[j] < 0:
                loss_4 = - stepsize_list[j]*coe4
                loss_all += loss_4
    
        if loss_all != 0:      
            loss_all.backward()

        for i in range(half_length):
            if stepsize_list[i].grad != None:
                stepsize_list[i] = stepsize_list[i] - lr*stepsize_list[i].grad
                 
            if status1 == 1:
                stepsize_list[i] = stepsize_list[i]*(1/args.factor_lc - 0.049*(epoch/num_epoch)) 
                
            if status2 == 1:
                stepsize_list[i] = stepsize_list[i]*(args.factor_lc + 0.049*(epoch/num_epoch)) 
                
            stepsize_list[i] = stepsize_list[i].detach().clone()
            stepsize_list[i].requires_grad = False
    
        for i in stepsize_list:
            if i > 1.0:
                print("The gradient is too large")
                raise ValueError
        
    return best_stepsize_list




def generate_bilateral_tra(model, stepsize_list, image, target):
    
    stepsize_list = stepsize_list + stepsize_list[::-1]
    process_image = image
    
    images_tra = []
    images_tra.append(process_image)
    grad_list = []
    for i in range(len(stepsize_list)):
        model.eval()
        model.zero_grad()
        
        # Set requires_grad attribute of tensor. Important for Attack
        process_image.requires_grad = True
        
        # Forward pass the data through the model
        output = model(process_image)
        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect ``datagrad``
        data_grad = process_image.grad.data.detach().clone()
        grad_list.append(data_grad)

        # Zero all existing gradients
        process_image.grad.zero_()
        
        # generate the next adversarial sample
        process_image = next_adv_sample(process_image, stepsize_list[i], data_grad)
        process_image = process_image.detach().clone()
        process_image.requires_grad = False
        images_tra.append(process_image.detach().clone())
        
    
    pred_list = []
    for adv_image in images_tra:
        output = model(adv_image)
        output = F.softmax(output,dim=1)
        final_pred = output.max(1, keepdim=True)[1]
        pred_list.append(final_pred[0][0].item())
        
    images_tra = torch.cat(images_tra)
    return images_tra, pred_list




def generate_all_classes(model, image, label, args):
    
    # Exclude the correct label
    class_list = np.arange(args.num_classes)
    class_list = np.delete(class_list, label.cpu())
    
    # Randomly select the categories to cross
    if args.tra_classes >  args.num_classes:
        raise Exception("The classes traversed by the trajectory are larger than the total number of classes!")
    class_list = np.random.choice(class_list, args.tra_classes - 1, replace=False)
    
    # start generate trajectories
    tra_log, pred_log = [], []
    model.to(args.device)
    for class_i in class_list:
        target = torch.tensor(class_i).to(args.device).unsqueeze(0).to(torch.long)
        # generate a unilateral trajectory
        stepsize_list = generate_unilateral_tra(model, image, target, args, num_epoch=args.max_iteration, coe3=1, coe4=1, 
                                                decay=args.factor_re, lr=args.tra_lr)
        # Check whether it is generated successfully
        if stepsize_list == None:
            return None
        # bilateralize the trajectory
        tra, pred = generate_bilateral_tra(model, stepsize_list, image, target)
        
        # Record the last sample of this trajectory as the beginning of the next trajectory
        image = tra[-1].detach().clone().unsqueeze(0)
        
        # Record the prediction of the current trajectory
        tra_pred = model(tra)
        tra_pred = tra_pred.max(1, keepdim=True)[1].reshape(-1) 
        tra_log.append(tra.detach())
        pred_log.append(copy.deepcopy(tra_pred))
    
    return tra_log, pred_log



def generate_trajectory(args):
    if args.dataset == "ImageNet":
        dataset = get_data(args.dataset, "./Data")
        
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        X = []
        y = []
        count = 0
        for data in data_loader:
            X.append(data[0])
            y.append(data[1])
            count += len(data[0])
            if count > 3*args.num_trajectories:
                break
        X = torch.cat(X,axis=0)
        y = torch.cat(y,axis=0)

    else:
        data_log = torch.load(args.data_path + '/' + args.dataset + '/allocated_data' + '/data_log.pth')
        X = data_log["X_train"]
        y = data_log["y_train"]
    
    # Reserve more samples for substitute
    images = X[0:2*args.num_trajectories].to(args.device)
    labels = y[0:2*args.num_trajectories].to(args.device)
    
    source_model = build_model(args)
    source_model.load_state_dict(torch.load(args.model_path + '/' + args.dataset + '/source_model.pth', map_location=args.device))
    source_model.to(args.device)
    num_finger = 0
    
    for image, label in zip(images, labels):
        image, label = image.unsqueeze(0).to(args.device), label.unsqueeze(0).to(args.device)
        temp = generate_all_classes(source_model, image, label, args)
        if temp == None:
            print("This basic sample cannot generate a stable adversarial trajectory, change to the next one")
            continue
        tra_log, pred_log = temp
        num_finger += 1
        
        # save the trajectories
        save_dir = args.fingerprint_path + '/' + args.dataset +'/trajectory_' + str(args.length) + '/' + str(num_finger)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tra_log, save_dir+"/tra_log.pth")
        torch.save(pred_log, save_dir+"/pred_log.pth")
        
        # Checks that a specified number of tracks are generated
        if args.num_trajectories == num_finger:
            break

    return None




def verify_trajectory(args):
    model = build_model(args)
    model.to(args.device)
    model.load_state_dict(torch.load(args.suspect_path))
    model.eval()
    
    # calculate dectetion rate w.r.t. the suspect model
    count_pos = 0
    for idx in range(1, args.num_trajectories+1):
        tra_log = torch.load(args.fingerprint_path + '/' + args.dataset + "/trajectory_" + str(args.length) + "/" + str(idx) + "/tra_log.pth")
        ori_pred = torch.load(args.fingerprint_path + '/' + args.dataset + "/trajectory_" + str(args.length) + "/" + str(idx) + "/pred_log.pth")
        tra_log = torch.cat(tra_log)
        ori_pred = torch.cat(ori_pred)
        
        tra_pred = model(tra_log)
        tra_pred = tra_pred.max(1, keepdim=True)[1].reshape(-1)
        # calclate the mutation rate for the trajectory of the suspect model
        mutation = ((tra_pred != ori_pred) * 1.0).mean()
        print(mutation)
        if mutation < args.threshold:
            count_pos += 1
    
    detection_rate = count_pos/args.num_trajectories
    print("The fingerprint detection rate of the suspect model is: ", detection_rate)




