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
    def __init__(self, samples, sample_voxel_net_index, device, num_classes=2):
        self.samples = samples
        self.sample_voxel_net_index = sample_voxel_net_index
        self.device = device
        self.adjust_label = 1
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)
    
    # return sample_points, sample_label, voxel_skeleton
    def __getitem__(self, index):
        #samples : [[x,y,z,label,reflectance, 12*features, (17) gd, (18) ier, x_raw, y_raw, z_raw], ...]
        # (x,y,z,label), label index is 3
        points = self.samples[index][:,:3]
        points_raw = self.samples[index][:,-3:]

        # for input data, leave is 2, wood is 1
        # make wood = 0 leave = 1
        labels = self.samples[index][:,3]
        labels = labels - 1

        # convert to one-hot form
        labels = np.eye(self.num_classes)[labels.astype(int)].transpose()

        # put them into self.device
        points = torch.from_numpy(points.copy()).type(torch.float).to(self.device)
        pointwise_features = torch.from_numpy(self.samples[index][:,5:17].copy()).type(torch.float).to(self.device)
        
        return points, pointwise_features, labels, 0, points_raw
        

    # print the info
    def show_info(self):
        print(">> [Train/Val] DataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))

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
    def __init__(self, samples, sample_voxel_net_index, device, sample_position, num_classes=2):
        self.samples = samples
        self.sample_voxel_net_index = sample_voxel_net_index
        self.device = device
        self.num_classes = num_classes
        self.sample_position = sample_position

    def __len__(self):
        return len(self.samples)
    
    # return sample_points, sample_label, voxel_skeleton
    def __getitem__(self, index):
        #samples : [[x,y,z,label,reflectance, 12*features, (17) gd, (18) ier, x_raw, y_raw, z_raw], ...]
        # (x,y,z,label), label index is 3
        points = self.samples[index][:,:3]
        points_raw = self.samples[index][:,-3:]
        
        # for input data, leave is 2, wood is 1
        # make wood = 0 leave = 1
        labels = self.samples[index][:,3]
        labels = labels - 1
        sp = self.sample_position[self.sample_voxel_net_index[index]]

        # convert to one-hot form
        #labels = np.eye(self.num_classes)[labels.astype(int)].transpose()

        # put them into self.device
        points = torch.from_numpy(points.copy()).type(torch.float).to(self.device)
        pointwise_features = torch.from_numpy(self.samples[index][:,5:17].copy()).type(torch.float).to(self.device)
        gd = torch.from_numpy(self.samples[index][:,17].copy()).type(torch.float).to(self.device)
        id_comp = torch.from_numpy(self.samples[index][:,18].copy()).type(torch.float).to(self.device)
        return points, pointwise_features, labels, sp, points_raw, gd, id_comp

    # print the info
    def show_info(self):
        print(">> TestDataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))
        print(">>> self.sample_position.shape={}".format(len(self.sample_position)))
        return None
