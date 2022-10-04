import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# PointNet++
# spatial transformer networks k-D
# transform the feature
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
 
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
 
        self.k = k
 
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
 
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
 
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

# PointNet
class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        if self.global_feat:
            x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        else:
            return x, trans, trans_feat

# MPVCNN: multi-scale Point Voxel CNN
class PointWiseModel(nn.Module):
    # initialization
    def __init__(self, device, num_classes=2, hidden_dim=256):
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
        self.conv_1 = nn.Conv3d(1, 32, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 64
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 128
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 128
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 128
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 128

        '''
        # kernel_size = 5
        self.conv_1_ks5 = nn.Conv3d(1, 32, 5, padding=1)  # out: 32
        self.conv_1_1_ks5 = nn.Conv3d(32, 64, 5, padding=1)  # out: 64
        self.conv_2_ks5 = nn.Conv3d(64, 128, 5, padding=1)  # out: 128
        self.conv_2_1_ks5 = nn.Conv3d(128, 128, 5, padding=1)  # out: 128
        self.conv_3_ks5 = nn.Conv3d(128, 128, 5, padding=1)  # out: 128
        self.conv_3_1_ks5 = nn.Conv3d(128, 128, 5, padding=1)  # out: 128

        # kernel_size = 7
        self.conv_1_ks7 = nn.Conv3d(1, 32, 7, padding=1)  # out: 32
        self.conv_1_1_ks7 = nn.Conv3d(32, 64, 7, padding=1)  # out: 64
        self.conv_2_ks7 = nn.Conv3d(64, 128, 7, padding=1)  # out: 128
        self.conv_2_1_ks7 = nn.Conv3d(128, 128, 7, padding=1)  # out: 128
        self.conv_3_ks7 = nn.Conv3d(128, 128, 7, padding=1)  # out: 128
        self.conv_3_1_ks7 = nn.Conv3d(128, 128, 7, padding=1)  # out: 128
        '''

        # feature_size was setting 3 times for multi-scale learning/multi receptive field
        # +7 : intensity added + roughness added + ncr added ... 
        # + 256 : output of fc of the pointwise_features
        # + (256 + 64) : pointnet segmentation output
        feature_size = 1 + (64 + 128 + 128) + 7 + 256 + (256 + 64)

        # conditionnal VAE, co-variabale, regression
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim*4, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*4, hidden_dim*2, 1)
        self.fc_2 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        #self.fc_out = nn.Conv1d(hidden_dim, 1, 1) chang  1 to self.num_classes
        self.fc_out = nn.Conv1d(hidden_dim, self.num_classes, 1)

        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        # point_feature_size = 7
        self.mlp_0 = nn.Conv1d(7, hidden_dim*2, 1)
        self.mlp_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.mlp_2 = nn.Conv1d(hidden_dim, 256, 1)
        
        '''
        self.conv1_1_bn_ks5 = nn.BatchNorm3d(64)
        self.conv2_1_bn_ks5 = nn.BatchNorm3d(128)
        self.conv3_1_bn_ks5 = nn.BatchNorm3d(128)

        self.conv1_1_bn_ks7 = nn.BatchNorm3d(64)
        self.conv2_1_bn_ks7 = nn.BatchNorm3d(128)
        self.conv3_1_bn_ks7 = nn.BatchNorm3d(128)
        '''

        # point-based branch: PointNet
        self.point_base_model = PointNetfeat(global_feat=True, feature_transform=True)

    # forward propagation
    def forward(self, points, pointwise_features, v):
        '''
        Args:
            p: points, a 3d pytorch tensor.
            pointwise_features: pointwise_features[0] = intensity.
            v: voxel net.
        Returns: 
            None.
        '''
        
        '''
        [*] points, p.shape=torch.Size([4, 20000, 3])
        [*] intensity.shape=torch.Size([4, 20000])
        [*] pointwise_features.shape=torch.Size([4, 20000, nb_of_features])
        [*] v_cuboid, v.shape=torch.Size([4, 25, 25, 250])
        print("[*] points, p.shape={}".format(p.shape))
        print("[*] intensity.shape", intensity.shape)
        print("[*] v_cuboid, v.shape={}".format(v.shape))
        '''
        p_pn = points.transpose(1,2)
        
        point_features,_,_ = self.point_base_model(p_pn)
        #print(">> point_features.shape={}".format(point_features.shape))
        point_features = point_features.unsqueeze(2).unsqueeze(2)
        

        # swap x y z to z y x
        p = points[:,:,[2,1,0]]
        v = v.unsqueeze(1)
        #v = torch.permute(v, dims=[0,1,4,2,3])
        '''
        v = v.permute((0,1,4,2,3))
        '''
        p = p.unsqueeze(1).unsqueeze(1)
        
        '''
        [*] points, p.shape=torch.Size([4, 1, 1, 20000, 3])
        [*] intensity.shape torch.Size([4, 20000])
        [*] v_cuboid, v.shape=torch.Size([4, 1, 25, 25, 250])
        '''
        # displacements
        #p = torch.cat([p + d for d in self.displacments], dim=2)
        
        #print("what is points, p.shape={}".format(p.shape))
        #print("what is v_cuboid, v.shape={}".format(v.shape))

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
        #feature_1_ks5 = F.grid_sample(net5, p, align_corners=True)
        #feature_1_ks7 = F.grid_sample(net7, p, align_corners=True)

        net = self.maxpool(net)
        #net5 = self.maxpool(net5)
        #net7 = self.maxpool(net7)

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
        #feature_2_ks5 = F.grid_sample(net5, p, align_corners=True)
        #feature_2_ks7 = F.grid_sample(net7, p, align_corners=True)

        net = self.maxpool(net)
        #net5 = self.maxpool(net5)
        #net7 = self.maxpool(net7)

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
        #feature_3_ks5 = F.grid_sample(net5, p, align_corners=True)
        #feature_3_ks7 = F.grid_sample(net7, p, align_corners=True)

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
            feature_3,
            point_features
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
        features = torch.cat((features, feature_mlp, pointwise_features), dim=1)
        net_out = self.actvn(self.fc_0(features))
        net_out = self.actvn(self.fc_1(net_out))
        net_out = self.actvn(self.fc_2(net_out))
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
    
'''
# spatial transformer networks 3-D 
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
 
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
 
 
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
 
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
 
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
'''

# Point Segmentation
class PointNetSemSeg(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetSemSeg, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat