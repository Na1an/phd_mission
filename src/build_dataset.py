import torch
from torch.utils.data import Dataset

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
    def __init__(self, samples, sample_cuboid_index, device):
        self.samples = samples
        self.sample_cuboid_index = sample_cuboid_index
        #self.voxelized_cuboids = voxelized_cuboids
        self.device = device
        self.adjust_label = 2

    def __len__(self):
        return len(self.samples)
    
    # return sample_points, sample_label, voxel_skeleton
    def __getitem__(self, index):
        # (x,y,z,label), label index is 3
        points = self.samples[index][:,:3]
        # minus self.adjust_label because the original data, label=3 -> leaf, label=2 -> wood  
        # now, label=1 -> leaf, label=0 -> wood
        intensity = self.samples[:,3]
        labels = self.samples[index][:,4] - self.adjust_label

        # put them into self.device
        points = torch.from_numpy(points.copy()).type(torch.float).to(self.device)
        labels = torch.from_numpy(labels.copy()).type(torch.float).to(self.device)
        intensity = torch.from_numpy(intensity.copy()).type(torch.float).to(self.device)
        #v_cuboid = torch.from_numpy(self.voxelized_cuboids[self.sample_cuboid_index[index]]).type(torch.int).to(self.device)
        index_of_voxel_net = self.sample_cuboid_index[index]

        #return points, labels, v_cuboid
        return points, intensity, labels, index_of_voxel_net

    # print the info
    def show_info(self):
        print(">> TrainDataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))
        print(">>> samples_cuboid_index.shape={}".format(len(self.sample_cuboid_index)))
        return None

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
    def __init__(self, samples, sample_cuboid_index, device):
        self.samples = samples
        self.sample_cuboid_index = sample_cuboid_index
        self.device = device

    def __len__(self):
        return len(self.samples)
    
    # return sample_points, sample_label, voxel_skeleton
    def __getitem__(self, index):
        # (x,y,z,label), label index is 3
        points = self.samples[index][:,:3]
        index_sw = self.samples[index][:,3][0]

        # put them into self.device
        points = torch.from_numpy(points.copy()).type(torch.float).to(self.device)
        #v_cuboid = torch.from_numpy(self.voxelized_cuboids[self.sample_cuboid_index[index]]).type(torch.int).to(self.device)
        index_of_voxel_net = self.sample_cuboid_index[index]

        #return points, v_cuboid
        return points, index_of_voxel_net, index_sw

    # print the info
    def show_info(self):
        print(">> TrainDataSet is prepared:")
        print(">>> device={}".format(self.device))
        print(">>> samples.shape={}".format(self.samples.shape))
        print(">>> samples_cuboid_index.shape={}".format(len(self.sample_cuboid_index)))
        return None