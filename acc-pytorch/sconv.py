import os
import torch
import torch.nn as nn
from .CutFunc import *
import sys
import random

sys.path.append("..")
sys.path.append("../..")
from utils import *

class sconv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=bias, padding_mode=padding_mode)
        self._hook_handlers = []
        self.grad_cut = False
        self.cut_percent = 0.
        self.skel_cut = False
        
        self.set_skel_process = False
        self.use_skel_process = False
        self.act_accumulation = None
        self.random_percent = 0.0
        
        self.mask_by_act = None
        self.cut_based = "act"

    def set_random_percent(self, random_percent):
        self.random_percent = random_percent
    
    def set_cut_based(self, cut_based):
        self.cut_based = cut_based
        
    def enable_set_skel_process(self, cut_method):
        self.set_skel_process = True
        self.cut_method = cut_method
        del self.act_accumulation
        self.act_accumulation = None
        del self.mask_by_act
        self.mask_by_act = None
    
    def disable_set_skel_process(self):
        self.set_skel_process = False
        
    def enbale_use_skel_process(self):
        self.use_skel_process = True
        
        arg_sort = None
        if self.cut_based == "act":
            assert(not self.act_accumulation is None)
            arg_sort = np.argsort(self.act_accumulation)

        pre_cut_index = arg_sort[:int(self.cut_percent*len(arg_sort))]
        cut_index = []
        if self.random_percent == 0:
            cut_index = pre_cut_index
        else:
            for index in pre_cut_index:
                if random.random() > self.random_percent:
                    cut_index.append(index)
        
        mask_index = torch.tensor(cut_index)
        mask_index = mask_index.to('cuda')
        self.mask_by_act = mask_index
    
    def disable_use_skel_process(self):
        self.use_skel_process = False
        
    def enable_skel_cut(self, cut_method):
        self.skel_cut = True
        self.cut_method = cut_method

    def set_cut_percent(self, cut_percent):
        self.cut_percent = cut_percent

    def enable_grad_cut(self, cut_method):
        self.grad_cut = True
        self.cut_method = cut_method

    def disable_cut(self):
        self.skel_cut = False
        self.grad_cut = False

    def forward(self, input_tensor):
        output = super().forward(input_tensor)

        if not self.training:
            return output

        if self.grad_cut or self.skel_cut or self.use_skel_process or self.set_skel_process:
            self._hook_handlers.append(output.register_hook(self.grad_hooker))
        
        if self.skel_cut:
            self.set_mask_by_act(output)
            
        if self.set_skel_process and self.cut_based == "act":
            self.accumulate_act(output)
       
        return output
    
    def accumulate_act(self, act):
        new_act = act.clone().detach().abs()
        new_act_cpu = new_act.cpu().numpy()
        kernel_act = np.sum(new_act_cpu, axis=(0,2,3))
        if self.act_accumulation is None:
            self.act_accumulation = kernel_act
        else:
            self.act_accumulation += kernel_act
        del new_act, new_act_cpu
        
    def set_mask_by_act(self, act):
        new_act = act.clone().detach().abs()
        new_act_cpu = new_act.cpu().numpy()
        L1_norm = np.sum(new_act_cpu, axis=(0,2,3))
        arg_sort = np.argsort(L1_norm)
        cut_index = arg_sort[:int(self.cut_percent*len(arg_sort))]
        mask_index = torch.LongTensor(cut_index.tolist())
        mask_index = mask_index.to('cuda')
        self.mask_by_act = mask_index
        del new_act, new_act_cpu, L1_norm, arg_sort, cut_index

    def grad_hooker(self, grad):
        new_grad = grad

        torch.cuda.empty_cache()
        
        if self.skel_cut or self.use_skel_process:
            new_grad = new_grad.index_fill_(1, self.mask_by_act, 0)

        del grad
        return new_grad

    def _remove_hook_handlers(self):
        for hook in self._hook_handlers:
            hook.remove()
            
    # def combine_weight(self, origin_model_layer, critical_ratio):
    #     if not isinstance(origin_model_layer, sconv):
    #         assert(0)
    #     if self.mask_by_act is None:
    #         return

    #     new_weight_critical = self.weight.data.clone()
    #     new_weight_uncritical = self.weight.data.clone()
    #     origin_weight_clone = origin_model_layer.weight.data.clone()
        
    #     new_weight_critical = new_weight_critical.to('cuda').index_fill_(0, self.mask_by_act, 0)
        
    #     uncritical_list = self.mask_by_act.tolist()
    #     critical_set = set(range(0,new_weight_critical.shape[0])) - set(uncritical_list)
    #     critical_mask = torch.LongTensor(list(critical_set)).to('cuda')
       
    #     new_weight_uncritical.index_fill_(0, critical_mask, 0)
    #     origin_weight_clone.index_fill_(0, critical_mask, 0)
        
    #     combined_weight = new_weight_critical + origin_weight_clone * (1-critical_ratio) + new_weight_uncritical * critical_ratio
        
    #     self.weight.data = combined_weight
        
    #     del new_weight_uncritical
    #     del origin_weight_clone
    #     del critical_mask
    #     del new_weight_critical
    #     torch.cuda.empty_cache()


    def update_skel_weight(self, server_weight):
        if self.mask_by_act is None:
            self.weight.data = server_weight.weight.data
        else :
            print('[DEBUG] Download Only Skeleton Network ......')
            new_weight = server_weight.weight.data.clone()
            worker_weight = self.weight.data
            un_skel_list = self.mask_by_act.tolist()
            skel_set = set(range(0,new_weight.shape[0])) - set(un_skel_list)
            skel_mask = torch.LongTensor(list(skel_set)).to('cuda')
            new_weight = new_weight.index_fill_(0, self.mask_by_act, 0)
            worker_weight = worker_weight.index_fill_(0, skel_mask, 0)
            self.weight.data = new_weight + worker_weight