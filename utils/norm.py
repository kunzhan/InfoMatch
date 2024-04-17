import ipdb

import torch
# L1，L2正则化函数
def l1_norm(model):
       l1_norm = 0
       for layer,param in model.state_dict().items(): 
              #ipdb.set_trace()
              l1_norm += torch.norm(param.cpu().float(),1)
       return l1_norm

def l2_norm(model):
       l2_norm = 0
       for layer,param in model.state_dict().items(): 
              #ipdb.set_trace()
              l2_norm += torch.norm(param.cpu().float(),1)
       return l2_norm
