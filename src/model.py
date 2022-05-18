import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseModel(nn.Module):

    # initialization
    def __init__(self, hidden_dim=256):
        '''
        Args:
            
        Returns: 
            None.
        '''
        super().__init__()
        # define architecture here
        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)
        
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

        feature_size = (1 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim*2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)

        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        displacment = 0.035
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments)

    # forward propagation
    def forward(self, p, v):
        '''
        Args:
            input : a 3d pytorch tensor.
        Returns: 
            None.
        '''
        print(">>>> so we start our forward")
        
        v = v.unsqueeze(1)
        p = p.unsqueeze(1).unsqueeze(1)
        print("what is points, p.shape={}".format(p.shape))
        print("what is v_cuboid, v.shape={}".format(v.shape))

        print("<<<< maybe end here")
        
        
        # apply all layers here

    

''' 
def old_code():
    grid_size : a interger/float. The side length of a grid.
            voxel_size : a float. The resolution of the voxel. 
            global_height : a float. The max height of the raw data.
            device : a string. Tensor mode "cpu" or "gpu".
            hidden_dim : no idea for the moment.
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.global_height = global_height
        
        self.x_v = int((grid_size+0.000000000001)//voxel_size)
        self.y_v = int((grid_size+0.000000000001)//voxel_size)
        self.z_v = int((global_height+0.000000000001)//voxel_size)
        self.nb_vox = self.x_v*self.y_v*self.z_v
    # show voxel shape
    def show_voxel_shape(self):
        print("model: x_v=", self.x_v, "y_v=",self.y_v, "z_v=",self.z_v, " -> nb_vox=", self.nb_vox)
        return None
'''