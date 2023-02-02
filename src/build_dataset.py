import torch
import numpy as np
from torch.utils.data import Dataset

def check_nan_in_array(feature_name,a):
    print(feature_name + "shape={} nan size={}".format(a.shape, a[np.isnan(a)].shape))
    return None
# dataset for training
class TrainDataSet(Dataset):
    '''
    Args:
        #points : (x,y,z,label) numpy.ndarray.
        device : cpu or gpu.
        samples: (nb_sample, 5000, 4 :x + y + z + label).
        sample_cuboid_index: (nb_sample, index of nb_cuboid).
        voxelized_cuboids: (nb_voxel, 4:x+y+z+[1 or 0]).
    '''
    def __init__(self, samples, sample_voxel_net_index, samples_voxelized, device, num_classes=2):
        self.samples = samples
        self.sample_voxel_net_index = sample_voxel_net_index
        self.samples_voxelized = samples_voxelized
        self.device = device
        self.adjust_label = 1
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)
    
    # return sample_points, sample_label, voxel_skeleton
    def __getitem__(self, index):
        #samples : [[x,y,z,label,reflectance,gd,ier,PCA1,linearity,verticality], ...]
        #samples_voxelized : [[x,y,z,point_density], ...]
        # (x,y,z,label), label index is 3
        points = self.samples[index][:,:3]
        
        # for input data, leave is 2, wood is 1
        # make wood = 1 leave = 0
        labels = self.samples[index][:,3]
        labels[labels==2] = 0

        '''
        # features
        reflectance = self.samples[index][:,4]
        gd = self.samples[index][:,5]
        ier = self.samples[index][:,6]
        pca1 = self.samples[index][:,7]
        linearity = self.samples[index][:,8]
        verticality = self.samples[index][:,9]
        '''

        # convert to one-hot form
        labels = np.eye(self.num_classes)[labels.astype(int)].transpose()

        # put them into self.device
        points = torch.from_numpy(points.copy()).type(torch.float).to(self.device)

        #pointwise_features = [reflectance,gd,ier,PCA1,linearity,verticality]
        #pointwise_features = torch.from_numpy(self.samples[index][:,4:].copy()).type(torch.float).to(self.device)
        pointwise_features = torch.from_numpy(self.samples[index][:,[7,8,9]].copy()).type(torch.float).to(self.device)
        voxel_net = torch.from_numpy(self.samples_voxelized[self.sample_voxel_net_index[index]]).type(torch.float).to(self.device)
        return points, pointwise_features, labels, voxel_net

    # print the info
    def show_info(self):
        print(">> [Train/Val] DataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))
        print(">>> samples_voxelized.shape={}".format(self.samples_voxelized.shape))
        #print(">>> points.shape = {}, pointwise_features.shape={}, labels.shape={}, voxel_net.shape={}".format(points.shape, pointwise_features.shape, labels.shape, voxel_net.shape))

# dataset for testing
class TestDataSet(Dataset):
    '''
    Args:
        #points : (x,y,z,label) numpy.ndarray.
        device : cpu or gpu.
        samples: (nb_sample, 5000, 4 :x + y + z + label).
        sample_cuboid_index: (nb_sample, index of nb_cuboid).
        voxelized_cuboids: (nb_voxel, 4:x+y+z+[1 or 0]).
    '''
    def __init__(self, samples, sample_cuboid_index, device, num_classes=2):
        self.samples = samples
        self.sample_cuboid_index = sample_cuboid_index
        self.device = device
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)
    
    # return sample_points, sample_label, voxel_skeleton
    def __getitem__(self, index):
        # (x,y,z,label), label index is 3
        points = self.samples[index][:,:3]
        # 4->intensity, 5->roughness, 6->normal_change_rate
        intensity = self.samples[index][:,4]
        roughness = self.samples[index][:,5]
        ncr = self.samples[index][:,6]
        '''
        return_number = self.samples[index][:,7]
        number_of_returns = self.samples[index][:,8]
        rest_return = self.samples[index][:,9]
        ratio_return = self.samples[index][:,10]
        '''

        index_sw = self.samples[index][:,3][0]

        #pointwise_features = [intensity, roughness, ncr, return_number, number_of_returns, rest_return, ratio_return]
        pointwise_features = torch.from_numpy(self.samples[index][:,4:].copy()).type(torch.float).to(self.device)

        #v_cuboid = torch.from_numpy(self.voxelized_cuboids[self.sample_cuboid_index[index]]).type(torch.int).to(self.device)
        index_of_voxel_net = self.sample_cuboid_index[index]

        #return points, v_cuboid
        return points, pointwise_features, index_of_voxel_net, index_sw

    # print the info
    def show_info(self):
        print(">> TestDataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))
        print(">>> samples_cuboid_index.shape={}".format(len(self.sample_cuboid_index)))
        return None
