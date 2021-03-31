import torch.nn as nn
import torch.nn.functional as F
import torch

import Feed_and_Loss.loss as my_loss
import Models.resnet as resnet

""" !!!!!!!!!!!!!!!!!!!!!!!!!! 주의  !!!!!!!!!!!!!!!!!!!!!!!!!!  """
""" 
    이 페이지는 v1의 구성이된 네트워크에다 encoder for g-buffer를 넣고자 함. 
    따라서, 하나의 추가적인 네트워크가 있어서 high level feature를 추가적으로 잡음.    
"""
