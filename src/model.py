import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseModel(nn.Module):

    # initialization
    def __init__(self, device, hidden_dim=256):
        '''
        Args:
            
        Returns: 
            None.
        '''
        super().__init__()
        # define architecture here
        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)
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
        self.conv_1 = nn.Conv3d(1, 32, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8

        # feature_size was setting 7 for displacements
        #feature_size = (1 + 64 + 128 + 128 ) * 7 
        # intensity added
        feature_size = (1 + 64 + 128 + 128+1)
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim*2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        #self.fc_out = nn.Conv1d(hidden_dim, num_classes, 1)
        # plus log_softmax, x = F.log_softmax(x, dim=1), we will have a more flexible model -> predict more class 
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)

        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        '''
        displacment = 0.035
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)
        self.device = device
        self.displacments = torch.Tensor(displacments).to(self.device)
        '''

    # forward propagation
    def forward(self, p, intensity, v):
        '''
        Args:
            p: points, a 3d pytorch tensor.
            intensity: intensity.
            v: voxel net.
        Returns: 
            None.
        '''

        '''
        print("[*] points, p.shape={}".format(p.shape))
        print("[*] v_cuboid, v.shape={}".format(v.shape))
        print("[*] intensity.shape", intensity.shape)
        '''
        v = v.unsqueeze(1)
        #v = torch.permute(v, dims=[0,1,4,2,3])
        '''
        v = v.permute((0,1,4,2,3))
        p_x = p[...,0].clone().detach()
        p_y = p[...,1].clone().detach()
        p_z = p[...,2].clone().detach()
        p[...,0] = p_z
        p[...,1] = p_x
        p[...,2] = p_y
        '''
        p = p.unsqueeze(1).unsqueeze(1)
        #p = torch.cat([p + d for d in self.displacments], dim=2)
        
        print("what is points, p.shape={}".format(p.shape))
        print("what is v_cuboid, v.shape={}".format(v.shape))
        
        
        # feature_0
        feature_0 = F.grid_sample(v, p)
        net = self.actvn(self.conv_1(v))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        '''
        print("feature_0.shape", feature_0.shape) # feature_0.shape torch.Size([4, 1, 1, 7, 5000])
        print("after first conv_1, net=", net.shape)
        print("after first conv_1_1, net=", net.shape)
        '''

        # feature_1
        feature_1 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
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
        feature_2 = F.grid_sample(net, p)
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
        feature_3 = F.grid_sample(net, p)

        # here every channel corresponse to one feature.
        # !!!!!!!!!!!!!!!!!! see here, it is easy to add one extra feature non?
        feature_intensity = intensity.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_intensity), dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        
        '''
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
        print("feature_3.shape {}".format(feature_3.shape))
        print("features.shape {}".format(features.shape))
        print("features.shape {}".format(features.shape))
        '''

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)
        '''
        print("out shape:", out.shape)
        print("out is {}", out)
        '''

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
        return out
    
