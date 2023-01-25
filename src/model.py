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

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
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

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l0_points

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
        #self.actvn = nn.ReLU()
        self.actvn = nn.ELU()
        self.maxpool = nn.MaxPool3d(2)
        self.num_classes = num_classes
        self.show_net_shape = False
        
        #nn.Conv3d(N batch_size, Cin 输入图像通道数, D深度/高度, H图像高, W图像宽, padding=1)
        # 给voxel skeleton准备的net，只为提取feature
        #class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        """ 
        in_channels(int) – 输入信号的通道，就是输入中每帧图像的通道数
        out_channels(int) – 卷积产生的通道，就是输出中每帧图像的通道数
        kernel_size(int or tuple) - 过滤器的尺寸，假设为(a,b,c)，表示的是过滤器每次处理 a 帧图像，该图像的大小是b x c。
        stride(int or tuple, optional) - 卷积步长，形状是三维的，假设为(x,y,z)，表示的是三维上的步长是x，在行方向上步长是y，在列方向上步长是z。
        padding(int or tuple, optional) - 输入的每一条边补充0的层数，形状是三维的，假设是(l,m,n)，表示的是在输入的三维方向前后分别padding l 个全零二维矩阵，在输入的行方向上下分别padding m 个全零行向量，在输入的列方向左右分别padding n 个全零列向量。
        dilation(int or tuple, optional) – 卷积核元素之间的间距，这个看看空洞卷积就okay了
        groups(int, optional) – 从输入通道到输出通道的阻塞连接数；没用到，没细看
        bias(bool, optional) - 如果bias=True，添加偏置；没用到，没细看 
        """

        # voxel-based branch: if-net
        # kernel_size = 3
        self.conv_1 = nn.Conv3d(1, 16, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(16, 32, 3, padding=1)  # out: [32]
        self.conv_2 = nn.Conv3d(32, 64, 3, padding=1)  # out: 64
        self.conv_2_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: [64]
        self.conv_3 = nn.Conv3d(64, 64, 3, padding=1)  # out: 32
        self.conv_3_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: [64]

        # feature_size was setting 3 times for multi-scale learning/multi receptive field
        # +3 : intensity added + roughness added + ncr added ... 
        # + 128 : output of fc of the pointwise_features
        # + (128 + 64) : pointnet segmentation output
        feature_size = 1 + (32 + 64 + 64) + 3 + 128 + (128)

        # conditionnal VAE, co-variabale, regression
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim*2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        #self.fc_2 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        #self.fc_out = nn.Conv1d(hidden_dim, 1, 1) chang  1 to self.num_classes
        self.fc_out = nn.Conv1d(hidden_dim, self.num_classes, 1)

        self.conv1_1_bn = nn.BatchNorm3d(32)
        self.conv2_1_bn = nn.BatchNorm3d(64)
        self.conv3_1_bn = nn.BatchNorm3d(64)

        # point_feature_size = 7
        #self.mlp_0 = nn.Conv1d(7, hidden_dim*2, 1)
        self.mlp_0 = nn.Conv1d(3, hidden_dim*2, 1)
        self.mlp_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.mlp_2 = nn.Conv1d(hidden_dim, 128, 1)

        # point-based branch: PointNet
        self.point_base_model = Pointnet_plus(num_classes=self.num_classes)

    # forward propagation
    def forward(self, points, pointwise_features, v, points_for_pointnet):
        '''
        Args:
            p: points, a 3d pytorch tensor.
            pointwise_features: pointwise_features[0] = intensity.
            v: voxel net.
            points_for_pointnet: feed this data to poinnet++. [batch_size, coordiantes+features+coordinates_scaled, nb of points] e.g. [4,9,5000] 
        Returns: 
            None.
        '''
        
        '''
        [*] points, p.shape=torch.Size([4, 20000, 3])
        [*] pointwise_features.shape=torch.Size([4, 20000, nb_of_features])
        [*] v_cuboid, v.shape=torch.Size([4, 25, 25, 250])
        print("[*] points, p.shape={}".format(p.shape))
        print("[*] intensity.shape", intensity.shape)
        print("[*] v_cuboid, v.shape={}".format(v.shape))
        '''
        
        pointnet_features = self.point_base_model(points_for_pointnet)
        #print("[*] point_features={}".format(point_features.shape))

        # swap x y z to z y x
        p = points[:,:,[2,1,0]]
        p[:,:] = p[:,:] + 0.5
        v = v.unsqueeze(1)
        #v = torch.permute(v, dims=[0,1,4,2,3])
        '''
        v = v.permute((0,1,4,2,3))
        '''
        p = p.unsqueeze(1).unsqueeze(1)
        
        '''
        [*] points, p.shape=torch.Size([4, 1, 1, 20000, 3])
        [*] v_cuboid, v.shape=torch.Size([4, 1, 25, 25, 250])
        '''


        # grid_sample 
        # align_corner = True, consider the center of pixels/voxels 
        # align_corner = False, consider the corner of pixels/voxels
        # 
        # feature_0
        feature_0 = F.grid_sample(v, p, align_corners=True)
        
        net = self.actvn(self.conv_1(v))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)

        # feature_1
        feature_1 = F.grid_sample(net, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)
        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)

        '''
        print("feature_1.shape {}".format(feature_1.shape))
        print("after first conv_2, net=", net.shape)
        print("after first conv_2_1, net=", net.shape)
        '''

        # feature_3
        feature_2 = F.grid_sample(net, p, align_corners=True)
        net = self.maxpool(net)
        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)

        '''
        print("feature_2.shape {}".format(feature_2.shape))
        print("after first conv_3, net=", net.shape)
        print("after first conv_3_1, net=", net.shape)
        '''

        # feature_3
        feature_3 = F.grid_sample(net, p, align_corners=True)

        # pointwise_features = [intensity, roughness, ncr, return_number, number_of_returns, rest_return, ratio_return]
        '''
        feature_intensity = pointwise_features[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        feature_roughness = pointwise_features[1].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        feature_ncr = pointwise_features[2].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        feature_return_nb = pointwise_features[3].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        feature_number_of_returns = pointwise_features[4].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        feature_rest_return = pointwise_features[5].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        feature_ratio_return = pointwise_features[6].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        '''

        pointwise_features = pointwise_features.permute(0,2,1)
        # mlp
        feature_mlp = self.actvn(self.mlp_0(pointwise_features))
        feature_mlp = self.actvn(self.mlp_1(feature_mlp))
        feature_mlp = self.actvn(self.mlp_2(feature_mlp))

        #features = torch.cat((feature_0, feature_1, feature_1_ks5, feature_1_ks7, feature_2, feature_2_ks5, feature_2_ks7, feature_3, feature_3_ks5, feature_3_ks7, feature_intensity), dim=1)  # (B, features, 1,7,sample_num)
        #features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_intensity, point_features, feature_elevation), dim=1)  # (B, features, 1,7,sample_num)
        
        ############################
        # pointnet feature removed #
        ############################
        #features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_intensity, point_features), dim=1)
        '''
        features = torch.cat((
            feature_0, 
            feature_1, 
            feature_2, 
            feature_3, 
            feature_roughness, 
            feature_intensity, 
            feature_ncr, 
            feature_return_nb, 
            feature_number_of_returns,
            feature_rest_return,
            feature_ratio_return,
            point_features
        ), dim=1)
        '''
        features = torch.cat((
            feature_0, 
            feature_1, 
            feature_2, 
            feature_3
        ), dim=1)

        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        
        '''
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
        print("feature_3.shape {}".format(feature_3.shape))
        print("features.shape {}".format(features.shape))
        print("features.shape {}".format(features.shape))
        >> point_features.shape=torch.Size([4, 320, 5000])
        features.shape=torch.Size([4, 641, 5000]) feature_mlp.shape=torch.Size([4, 256, 5000]) pointwise_features=torch.Size([4, 7, 5000])
        features.shape torch.Size([4, 904, 5000])
        '''
        
        features = torch.cat((features, feature_mlp, pointwise_features, pointnet_features), dim=1)
        net_out = self.actvn(self.fc_0(features))
        net_out = self.actvn(self.fc_1(net_out))
        #net_out = self.actvn(self.fc_2(net_out))
        net_out = self.fc_out(net_out)
        out = net_out.squeeze(1)

        return out

'''
feature_0.shape torch.Size([4, 1, 1, 7, 5000])
after first conv_1, net= torch.Size([4, 32, 100, 10, 10])
after first conv_1_1, net= torch.Size([4, 64, 100, 10, 10])
feature_1.shape torch.Size([4, 64, 1, 7, 5000])
after first conv_2, net= torch.Size([4, 128, 50, 5, 5])
after first conv_2_1, net= torch.Size([4, 128, 50, 5, 5])
feature_2.shape torch.Size([4, 128, 1, 7, 5000])
after first conv_3, net= torch.Size([4, 128, 25, 2, 2])
after first conv_3_1, net= torch.Size([4, 128, 25, 2, 2])
feature_3.shape torch.Size([4, 128, 1, 7, 5000])
features.shape torch.Size([4, 321, 1, 7, 5000])
features.shape torch.Size([4, 2247, 5000])
out shape: torch.Size([4, 5000])
'''

