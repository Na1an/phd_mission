from torch.utils.data import Dataset, DataLoader

# dataset for training
class TrainDataSet(Dataset):
    '''
    Args:
        #points : (x,y,z,label) numpy.ndarray.
        device : cpu or gpu.
        voxel_skeleton_cuboid: a list of voxel cuboid. (nb_cuboid, x, y, z, 1 or 0).
    '''
    def __init__(self, samples, sample_cuboid_index, voxel_skeleton_cuboid, device):
        #现在开始搞voxel_skeleton_cuboid
        #points: 也许把points直接搞成samples (nb_sample, 5000, 4 :x + y + z + label)
        #再整一个 sample_cuboid_index (nb_sample, index of nb_cuboid)
        #self.data = points[:,:3]
        #self.label = points[:,3]
        self.samples = []
        self.device = device
        self.voxel_skeleton_cuboid = voxel_skeleton_cuboid

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

# 这里的training dataset是训练集
# len返回的是小块6×6×6的数据的norm
# getitem返回的是6×6×6小块的数据， x:coordinates + y:label
class TrainingDataset(Dataset):
    def __init__(self, root_dir, points_per_box, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + '*.npy')
        self.points_per_box = points_per_box
        self.label_index = 3
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        if point_cloud.shape[0] > self.points_per_box:
            shuffle_index = list(range(point_cloud.shape[0]))
            random.shuffle(shuffle_index)
            point_cloud = point_cloud[shuffle_index[:self.points_per_box]]

        x = point_cloud[:, :3]
        y = point_cloud[:, self.label_index] -1
        x, y = augmentations(x, y)
        if np.all(y != 0):
            y[y == 2] = 3  # if no ground is present, CWD is relabelled as stem.
        x = torch.from_numpy(x.copy()).type(torch.float).to(self.device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

        # Place sample at origin
        global_shift = torch.mean(x[:, :3], axis=0)
        x = x - global_shift

        data = Data(pos=x, x=None, y=y)
        return data
