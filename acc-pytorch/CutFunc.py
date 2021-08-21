import os
import numpy as np 
import torch
import torch.nn as nn
import random
from scipy.stats import norm, powerlaw, expon

def random_grad_cut(percent, tensor):
    num = tensor.shape[1]
    cut_index = random.sample(range(num), int(percent*num))
    mask_index = torch.tensor(cut_index)
    mask_index = mask_index.to('cuda')
    tensor_copy = tensor.clone()
    tensor_copy.index_fill_(1, mask_index, 0)
    del mask_index
    return tensor_copy

def l1_grad_cut(percent, tensor):
    new_tensor = tensor.abs().cpu().numpy()
    L1_norm = np.sum(new_tensor, axis=(0,2,3))
    arg_sort = np.argsort(L1_norm)
    cut_index = arg_sort[:int(percent*len(arg_sort))]
    mask_index = torch.tensor(cut_index.tolist())
    mask_index = mask_index.to('cuda')
    tensor_copy = tensor.clone()
    tensor_copy.index_fill_(1, mask_index, 0)

    del mask_index, L1_norm, arg_sort, cut_index, mask_index
    return tensor_copy

def cut_by_index(percent, tensor, cut_index=None):
    if cut_index is None:
        return l1_grad_cut(percent, tensor)
    tensor_copy = tensor.clone()
    cut_index = cut_index.to('cuda')
    tensor_copy.index_fill_(1, cut_index, 0)
    
    del cut_index
    return tensor_copy
 

def l1_grad_cut_with_th(percent, tensor):
    abs_tensor = tensor.abs().cpu().numpy()
    L1_norm = np.sum(abs_tensor, axis=(0,2,3))
    arg_max = np.argsort(L1_norm)
    th = np.mean(L1_norm)
    if th == 0:
        new_grad = tensor.clone()
        return new_grad.zero_()

    th = expon.ppf(1-percent, 0, th)
    arg_max_del = []
    for index in arg_max:
        if L1_norm[index] < th and random.random() < 1 - percent:
            arg_max_del.append(index)

    if len(arg_max_del) == len(arg_max):
        new_grad = tensor.clone()
        return new_grad.zero_()
    elif len(arg_max_del) == 0:
        return tensor

    mask_gradient_del = torch.tensor(arg_max_del)
    mask_gradient_del = mask_gradient_del.to('cuda')

    new_grad = tensor.clone()
    new_grad.index_fill_(1, mask_gradient_del, 0)
    del mask_gradient_del
    return new_grad
