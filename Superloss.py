

#initialization:
#1) regime: 
#    - none (pixel cersion)
#    - mean (whole picture)
#2) loss_name:
#    - 'cross_entropy' - for classification tasks
#    - 'dice' - for segmentation task
#    - 'bse','bse_usual' - for pixel segmentation task
#    - 'L1', 'mse', 'huber' - other possible versions


import math
import torch
import torch.nn.functional as F
import numpy as np

class SuperLoss():

    # initialization
    def __init__(self, tau, loss_name, regime, batch_size, lam ):
        super(SuperLoss, self).__init__()
        self.lam = lam
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.tau = tau
        self.regime = regime  

    #different types of losses
    def loss_usual (self, pred, targets):
      
        #cross-entropy for classification
        if self.loss_name == 'cross_entropy':
            loss_f = torch.nn.CrossEntropyLoss(reduction = self.regime)#.detach()
            loss = loss_f(pred, targets)

            
         # weighted DICE and BCE for segmentation
        elif self.loss_name =='dice':
            self.smooth = 1
#             self.pred = pred    
            bce = F.binary_cross_entropy_with_logits(pred, targets, reduction = 'mean') 
            
            self.targets = targets
            self.pred = F.sigmoid(pred)       
            self.pred = self.pred.contiguous()
            self.targets = self.targets.contiguous()    

            intersection = (self.pred * self.targets).sum(dim=2).sum(dim=2)                         
            dice = (2.*intersection + self.smooth)/(self.pred.sum(dim=2).sum(dim=2) + self.targets.sum(dim=2).sum(dim=2)  + self.smooth)  

            loss_dice = 1 - dice
            loss_dice = loss_dice.mean()
            
            bce_weight = 0.5
            loss = bce * bce_weight + loss_dice * (1 - bce_weight)
            
            

        elif self.loss_name == 'bce':
            bce = F.binary_cross_entropy_with_logits(pred, targets, reduction = self.regime)
            loss = bce

            
        ##### другой вариант для bce #####
        elif self.loss_name == 'bce_usual':
            pred = F.sigmoid(pred)
            bce = F.binary_cross_entropy(pred, targets, reduction = 'none')
            loss = bce

        #some usual losses for different tasks
        elif self.loss_name == 'L1':
            loss_f = torch.nn.L1Loss( reduction= self.regime)
            loss = loss_f(pred, targets)

        elif self.loss_name == 'mse':
            loss_f = torch.nn.MSELoss(reduction = self.regime)
            loss = loss_f(pred, targets)

        elif self.loss_name == 'huber':
            # loss_f = torch.nn.HuberLoss(reduction='mean')
            loss_f = torch.nn.SmoothL1Loss(reduction = self.regime)
            loss = loss_f(pred, targets)


        else: 
            raise Exception('No such function')

        return loss


    #Lambert function: f(w)=we^{w} -> f^-1
    # https://github.com/jackd/lambertw/blob/master/tf_impl.py
    def lambert(self,z):
        self.step_tol = 1e-6
        
        def body(self, w, step):
            ew = torch.exp(w)
            numer = w*ew - z
            step = numer/(ew*(w+1) - (w+2)*numer/(2*w + 2))
            w = w - step
            return w, step

        w = torch.log(torch.tensor(1) + z)
        step = w
        
        def differ(self,step):
            return torch.max(torch.abs(step))
            
        max_iter = 0
        diff = 100
        while diff > self.step_tol or max_iter < 15:
            w,step =  body(self, w, step)
            diff = differ(self,step)
            max_iter += 1

        return w

    # optimization for sigma
    def sigma(self,loss):
        x = torch.ones(loss.size())*(-2/math.exp(1.))
        x = x.cuda()
       
        y = 0.5*torch.max(x,  (loss - self.tau)/self.lam)
        #y = y.cpu().detach().numpy()
        sigma = torch.exp(-self.lambert(torch.tensor(y, requires_grad = True)))
        #sigma = sigma.type(torch.float32, requires_grad=True)
        
        #sigma = sigma.real.type(torch.float32)
        sigma = sigma.cuda()

        return sigma
    

    def sup_loss(self,pred, targets):
        
        loss = self.loss_usual(pred, targets)
        
        sigma = self.sigma(loss)
        super_loss = (loss - self.tau)*sigma + self.lam*((torch.log(sigma))**2)
        #print(loss,super_loss)
        if self.regime == 'none':
            #super_loss =  super_loss.sum()/self.batch_size
            loss = loss.mean()
            super_loss = super_loss.mean()
        return loss, super_loss
