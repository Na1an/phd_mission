import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseModel(nn.Module):

    # initialization
    def __init__(self, grid_size, voxel_size, global_height, device="cpu", hidden_dim=256):
        '''
        Args:
            grid_size : a interger/float. The side length of a grid.
            voxel_size : a float. The resolution of the voxel. 
            global_height : a float. The max height of the raw data.
            device : a string. Tensor mode "cpu" or "gpu".
            hidden_dim : no idea for the moment.
        Returns: 
            None.
        '''
        super().__init__()
        self.device = device
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.global_height = global_height
        
        self.x_v = int((grid_size+0.000000000001)//voxel_size)
        self.y_v = int((grid_size+0.000000000001)//voxel_size)
        self.z_v = int((global_height+0.000000000001)//voxel_size)
        self.nb_vox = self.x_v*self.y_v*self.z_v

        # define architecture here

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

    # show voxel shape
    def show_voxel_shape(self):
        print("model: x_v=", self.x_v, "y_v=",self.y_v, "z_v=",self.z_v, " -> nb_vox=", self.nb_vox)
        return None

    
