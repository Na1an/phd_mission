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
        points_raw = self.samples[index][:,-3:]

        # for input data, leave is 2, wood is 1
        # make wood = 0 leave = 1
        labels = self.samples[index][:,3]
        labels = labels - 1

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
        pointwise_features = torch.from_numpy(self.samples[index][:,[7,8,9,10]].copy()).type(torch.float).to(self.device)
        #voxel_net = torch.from_numpy(self.samples_voxelized[self.sample_voxel_net_index[index]]).type(torch.float).to(self.device)
        
        return points, pointwise_features, labels, 0, points_raw
        

    # print the info
    def show_info(self):
        print(">> [Train/Val] DataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))
        #print(">>> samples_voxelized.shape={}".format(self.samples_voxelized.shape))
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
    def __init__(self, samples, sample_voxel_net_index, samples_voxelized, device, sample_position, samples_rest, num_classes=2):
        self.samples = samples
        self.sample_voxel_net_index = sample_voxel_net_index
        self.samples_voxelized = samples_voxelized
        self.device = device
        self.adjust_label = 1
        self.num_classes = num_classes
        self.sample_position = sample_position
        #self.samples_rest = samples_rest

    def __len__(self):
        return len(self.samples)
    
    # return sample_points, sample_label, voxel_skeleton
    def __getitem__(self, index):
        #samples : [[x,y,z,label,reflectance,gd,ier,PCA1,linearity,verticality], ...]
        #samples_voxelized : [[x,y,z,point_density], ...]
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
        
        #pointwise_features = [reflectance,gd,ier,PCA1,linearity,verticality]
        #pointwise_features = torch.from_numpy(self.samples[index][:,4:].copy()).type(torch.float).to(self.device)
        pointwise_features = torch.from_numpy(self.samples[index][:,[7,8,9,10]].copy()).type(torch.float).to(self.device)
        #voxel_net = torch.from_numpy(self.samples_voxelized[self.sample_voxel_net_index[index]]).type(torch.float).to(self.device)
        '''
        sample_rest = self.samples_rest[self.sample_voxel_net_index[index]]
        # just as label
        if len(sample_rest) == 0:
            sample_rest = []
        else:
            if np.max(sample_rest[:,3]) == 2: 
                sample_rest[:,3] = sample_rest[:,3] - 1
            else:
                sample_rest[:,3] = sample_rest[:,3] - 100
        '''

        return points, pointwise_features, labels, 0, sp, 0, points_raw

    # print the info
    def show_info(self):
        print(">> TestDataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))
        print(">>> self.sample_position.shape={}".format(len(self.sample_position)))
        return None
