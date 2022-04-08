import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseModel(nn.Module):

    # initialization
    def __init__(self, device="cpu", hidden_dim=256):
        '''
        Args:
            device : a string. Tensor mode "cpu" or "gpu".
            hidden_dim : no idea for the moment.
        Returns: 
            None.
        '''
        super().__init__()
        self.device = device
        
        # define the model architecture here 


    # forward propagation
    def forward(self, input, query_points):
        '''
        Args:
            input : a 3d pytorch tensor. Mode "cpu" or "gpu".
            hidden_dim : no idea for the moment.
        Returns: 
            None.
        '''
        
        # apply all layers here

    
