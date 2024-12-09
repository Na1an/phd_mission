import utility
from build_dataset import *
from preprocess_data import read_data_from_directory
from model import *

if __name__ == "__main__":
    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    soul_net = PointWiseModel(device=my_device)
    soul_net.summary()
    # validation dataset
    # val_dataset = read_data_from_directory(
    #                             "/home/yuchen/Documents/PhD/phd_mission/src/data", 
    #                             voxel_sample_mode='mc', 
    #                             voxel_size_ier=0.6,
    #                             label_name='WL', 
    #                             sample_size=3000, 
    #                             augmentation=False)
    # val_dataset = TrainDataSet(val_dataset, 0, my_device)
    # val_dataset.show_info()
    # points, pointwise_features, label, _, points_raw = val_dataset[0]
    # print("points.shape", points.shape)
    # points_for_pointnet = torch.cat([points.transpose(2,1), pointwise_features.transpose(2,1)], dim=1)
    # y = soul_net(points_for_pointnet.float())