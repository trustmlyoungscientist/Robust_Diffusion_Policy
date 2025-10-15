# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, te_alpha_d=19.9, te_alpha_min=0.1, epsilon_t=1e-5):
        self.te_alpha_d = te_alpha_d
        self.te_alpha_min = te_alpha_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.te_alpha_d * (t ** 2) + self.te_alpha_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class RobustEDMLoss:
    def __init__(self, P_mean=-1.2, Px_std=1.2, Py_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.Px_std = Px_std
        self.Py_std = Py_std
        self.sigma_data = sigma_data

    
    def __call__(self, net, images, raw_labels=None,target=None, index=None, augment_pipe=None, te_alpha=0.1, flag=1, cond_scale=1):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma_x = (rnd_normal * self.Px_std + self.P_mean).exp() * self.step
        weight_x = (sigma_x ** 2 + self.sigma_data ** 2) / (sigma_x * self.sigma_data) ** 2
        x, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        xn = torch.randn_like(x) * sigma_x

        target_labels = torch.zeros_like(raw_labels).to(raw_labels.device)
        target_labels[torch.arange(raw_labels.shape[0]),torch.argmax(target[index].data.detach(),dim=1)] = 1
        target_labels = (target_labels - target_labels.mean(dim=1, keepdim=True)) / target_labels.std(dim=1, keepdim=True) * cond_scale
        sigma_y = sigma_x.squeeze(2).squeeze(2)

        c1  = sigma_y / self.sigma_data ** 2
        c2 = -(self.sigma_data ** 2 + sigma_y ** 2).sqrt()/self.sigma_data
        c3 = self.sigma_data ** 2 / (sigma_y ** 2 + self.sigma_data ** 2)
        y = torch.randn_like(raw_labels) * 0.5
        raw_labels = (raw_labels - raw_labels.mean(dim=1, keepdim=True)) / raw_labels.std(dim=1, keepdim=True) * cond_scale
        if flag:
            yn = target_labels * sigma_y

            D_xn, F_y = net(x + xn, sigma_x, y+yn, augment_labels=augment_labels,flag=flag)
            
            yn_pred = c1 * y + c2 * F_y
            yn_pred_norm = (yn_pred - yn_pred.mean(dim=1, keepdim=True)) / yn_pred.std(dim=1, keepdim=True) * cond_scale
            yn_pred_norm_ = yn_pred_norm.data.detach()
            target[index] = (1- flag * te_alpha) * target[index] + flag * te_alpha * yn_pred_norm_

        else:
            D_xn, F_y = net(x + xn, sigma_x, target_labels, augment_labels=augment_labels,flag=flag)
            yn_pred = c1 * y + c2 * F_y
            yn_pred_norm = (yn_pred - yn_pred.mean(dim=1, keepdim=True)) / yn_pred.std(dim=1, keepdim=True) * cond_scale

        loss_label = flag * (c3 * ((yn_pred_norm -  raw_labels) ** 2)).mean()
        loss = (weight_x * ((D_xn - x) ** 2)).mean()
        return loss,loss_label




