
# 1)option:  
#     - 'with_loss' 
#     - 'with_superloss'

# 2)model:  
#     - 'Unet' 
#     - 'MiDaS'

# 3)regime:  
#     - 'mean' - per image
#     - 'none' - per pixel

import Superloss
import torch
import torch.nn as nn
import torch.nn.functional as F 
# from loss_dice import dice_loss

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, tau, batch_size, lam, option, model, regime):
    
    if model == 'MiDaS':
        if option == 'with_loss':
            loss_f = torch.nn.SmoothL1Loss(reduction= 'mean')
            loss = loss_f(pred, target)
            metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
            
        if option == 'with_superloss':
            l,loss = Superloss.SuperLoss(tau = tau, loss_name = 'huber', regime = regime, 
                                         lam = lam, batch_size = batch_size).sup_loss(pred, target)
            metrics['loss'] += l.data.cpu().numpy() * target.size(0)
            metrics['super_loss'] += loss.data.cpu().numpy() * target.size(0)
        
        criterion = torch.nn.MSELoss()
        RMSE = torch.sqrt(criterion(pred, target))
        metrics['RMSE'] += RMSE.data.cpu().numpy() * target.size(0)

        criterion_L1 = torch.nn.L1Loss()
        MAE = criterion_L1(pred, target)
        metrics['MAE'] += MAE.data.cpu().numpy() * target.size(0)
        
        if option == 'with_loss':
            return loss
        if option == 'with_superloss':
            return l, loss
        
     
    if model == 'Unet':
        if regime == 'mean':
            if option == 'with_loss':
                
                bce_weight = 0.5
                bce = F.binary_cross_entropy_with_logits(pred, target)
                pred = F.sigmoid(pred)
                dice = dice_loss(pred, target)

                loss = bce * bce_weight + dice * (1 - bce_weight)
                
                metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
                metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
                
                pred1 = pred > 0.75
                IOU = iou_pytorch(pred1, target)
                metrics['IOU'] += IOU.mean().item()
                
                return loss
            
            if option == 'with_superloss':
                l,loss = Superloss.SuperLoss(tau = tau, loss_name = 'dice', regime = regime, 
                                         lam = lam, batch_size = batch_size).sup_loss(pred, target)
                
                metrics['loss'] += l.data.cpu().numpy() * target.size(0)
                metrics['super_loss'] += loss.data.cpu().numpy() * target.size(0)
                
                pred = F.sigmoid(pred)
                dice = dice_loss(pred, target)
                metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
                
                pred1 = pred > 0.75
                IOU = iou_pytorch(pred1, target)
                metrics['IOU'] += IOU.mean().item()
                
                return l,loss
        
        if regime == 'none':   
            if option == 'with_loss':
                bce = F.binary_cross_entropy_with_logits(pred, target)
                loss = bce
                metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
                
                pred = F.sigmoid(pred)
                dice = dice_loss(pred, target)
                metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
                
                pred1 = pred > 0.75
                IOU = iou_pytorch(pred1, target)
                metrics['IOU'] += IOU.mean().item()
                
                return loss
            
            if option == 'with_superloss':
                l,loss = Superloss.SuperLoss(tau = tau, loss_name = 'bce', regime = regime, 
                                         lam = lam, batch_size = batch_size).sup_loss(pred, target)
                
                metrics['loss'] += l.data.cpu().numpy() * target.size(0)
                metrics['super_loss'] += loss.data.cpu().numpy() * target.size(0)
                
                pred = F.sigmoid(pred)
                dice = dice_loss(pred, target)
                metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
                
                pred1 = pred > 0.75
                IOU = iou_pytorch(pred1, target)
                metrics['IOU'] += IOU.mean().item()
                
                return l,loss   
