import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

# PointNet++
class Pointnet_plus(nn.Module):
    def __init__(self, num_classes):
        super(Pointnet_plus, self).__init__()
        # 3+12 : (x,y,z) + (verticality, specificity, pca1, linearity) * 3
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 15 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, points_for_pointnet):
        '''
        ps_fs: points + features.
        '''
        l0_ps_fs = points_for_pointnet[:,:3,:]
        l1_ps_fs, l1_points = self.sa1(l0_ps_fs, points_for_pointnet)
        l2_ps_fs, l2_points = self.sa2(l1_ps_fs, l1_points)
        l3_ps_fs, l3_points = self.sa3(l2_ps_fs, l2_points)
        l4_ps_fs, l4_points = self.sa4(l3_ps_fs, l3_points)
        l3_points = self.fp4(l3_ps_fs, l4_ps_fs, l3_points, l4_points)
        l2_points = self.fp3(l2_ps_fs, l3_ps_fs, l2_points, l3_points)
        l1_points = self.fp2(l1_ps_fs, l2_ps_fs, l1_points, l2_points)
        features = self.fp1(l0_ps_fs, l1_ps_fs, None, l1_points)

        return features

# MPVCNN: multi-scale Point Voxel CNN
class PointWiseModel(nn.Module):
    # initialization
    def __init__(self, device, num_classes=2, hidden_dim=128):
        '''
        Args:
            device : 'cuda' GPU or 'cpu' CPU.
            num_classes: a integer. The number of class.
            hidden_dim: a integer. The hidden layer dimension.
        Returns: 
            None.
        '''
        super().__init__()
        # define architecture here
        self.actvn = nn.ELU()
        self.maxpool = nn.MaxPool3d(2)
        self.num_classes = num_classes
        self.show_net_shape = False
        
        # feature_size was setting 3 times for multi-scale learning/multi receptive field
        # + 128 
        feature_size = 128
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim*2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        # self.num_classes : the number of class, e.g. 2 for leaf&wood.
        self.fc_out = nn.Conv1d(hidden_dim, self.num_classes, 1)

        # point-based branch: PointNet
        self.point_base_model = Pointnet_plus(num_classes=self.num_classes)

    # forward propagation
    def forward(self, points, pointwise_features, v, points_for_pointnet):
        '''
        Args:
            Args.
        Returns: 
            None.
        '''
        features = self.point_base_model(points_for_pointnet)
        net_out = self.actvn(self.fc_0(features))
        net_out = self.actvn(self.fc_1(net_out))
        net_out = self.fc_out(net_out)
        out = net_out.squeeze(1)
        return out

