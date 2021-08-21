import torch
import torch.nn as nn
from torch.autograd import Variable
from .sconv import *

class sparseNet(nn.Module):
    def __init__(self):
        super(sparseNet, self).__init__()
        
    def enable_set_skel_process(self, cut_method):
        for m in self.modules():
            if isinstance(m, sconv):   
                m.enable_set_skel_process(cut_method)
            
    def enbale_use_skel_process(self):
        for m in self.modules():
            if isinstance(m, sconv): 
                m.enbale_use_skel_process()

    def disable_set_skel_process(self):
        for m in self.modules():
            if isinstance(m, sconv):   
                m.disable_set_skel_process()

    def disable_use_skel_process(self):
        for m in self.modules():
            if isinstance(m, sconv):   
                m.disable_use_skel_process()

    def enable_grad_cut(self, cut_method):
        for m in self.modules():
            if isinstance(m, sconv):   
                m.enable_grad_cut(cut_method)

    def enable_skel_cut(self, cut_method):
        for m in self.modules():
            if isinstance(m, sconv):   
                m.enable_skel_cut(cut_method)

    def set_cut_percent(self, cut_percent):
        self.cut_percent = cut_percent
        for m in self.modules():
            if isinstance(m, sconv):   
                m.set_cut_percent(cut_percent)

    def set_cut_based(self, cut_based):
        for m in self.modules():
            if isinstance(m, sconv):   
                m.set_cut_based(cut_based)
    
    def disable_grad_cut(self):  
        for m in self.modules():
            if isinstance(m, sconv):   
                m.disable_cut()

    def disable_grad_cut(self, disable_indexs: list):
        index = -1  
        for m in self.modules():
            if isinstance(m, sconv):
                index += 1                      # conv index begin from 0
                if index in disable_indexs:
                    m.disable_cut()

    def model_update_skel(self, server_model):
        for model_layer, server_model_layer in zip(self.modules(), server_model.modules()):
            if isinstance(model_layer, sconv):
                model_layer.update_skel_weight(server_model_layer)