
import loss_calculation
from loss_calculation import calc_loss
from collections import defaultdict
import copy
import time
import torch
import numpy as np

# 1)model_name:  
#     - 'Unet' 
#     - 'MiDaS'

# 2)regime:  
#     - 'mean' - per image
#     - 'none' - per pixel
#
# 3) weights:
#    - yes (want to save)
#    - no (don't want to save)
#
# 4) dataloaders - name of dataloader

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))
    
def train_model_with_superloss(model, model_name, regime, weights, coef_noise, lam, dataloaders, 
                               device, optimizer, scheduler, batch_size, num_epochs):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    epoches = []
    #----------------------
    supp_loss_train = []
    us_loss_train = []
    acc_train = []
    supp_loss_val = []
    us_loss_val = []
    acc_val = []

    mae_train = []
    mae_val = []

    iou_train = []
    iou_val = []
    #-----------------
    loss_usual = []
    tau = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)    

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if model_name == 'MiDaS':
                        outputs = outputs.unsqueeze(1)

                    l, loss = calc_loss(outputs, labels, metrics, tau = tau, batch_size = batch_size, lam=lam,
                                      option = 'with_superloss', model = model_name, regime = regime)
                    
                    loss_usual.append(l.cpu().detach().numpy())
                    tau = np.mean(np.array(loss_usual))
                    

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
            
            print_metrics(metrics, epoch_samples, phase)
            
            if phase == 'train':
                us_loss_train.append(metrics['loss']/epoch_samples)
                supp_loss_train.append(metrics['super_loss']/epoch_samples)
                
                if model_name == 'MiDaS':
                    acc_train.append(metrics['RMSE']/epoch_samples)
                    mae_train.append(metrics['MAE']/epoch_samples)
                if model_name == 'Unet':
                    acc_train.append(metrics['dice']/epoch_samples)
                    iou_train.append(metrics['IOU']/epoch_samples)
                    
                

            if phase == 'val':
                
                us_loss_val.append(metrics['loss']/epoch_samples)
                supp_loss_val.append(metrics['super_loss']/epoch_samples)
                
                if model_name == 'MiDaS':
                    acc_val.append(metrics['RMSE']/epoch_samples)
                    mae_val.append(metrics['MAE']/epoch_samples)
                if model_name == 'Unet':
                    acc_val.append(metrics['dice']/epoch_samples)
                    iou_val.append(metrics['IOU']/epoch_samples)


            epoch_loss = metrics['loss'] / epoch_samples

            
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    if weights == 'yes':
        torch.save(model.state_dict(), "weights_path_" + model_name + '_' + regime + '_superloss_' + str(coef_noise) + '_lam_' + str(lam) + ".pth")  #### save weights
        torch.save(model, "model_" + model_name + '_' + regime + '_superloss_' + str(coef_noise) + '_lam_' + str(lam) + ".pth")  #### save model
        
    if model_name == 'MiDaS':
        return model, us_loss_train, us_loss_val, acc_train, acc_val, supp_loss_train, supp_loss_val, mae_train, mae_val
    
    if model_name == 'Unet':   
        return model, us_loss_train, us_loss_val, acc_train, acc_val, supp_loss_train, supp_loss_val, iou_train, iou_val
                
    
