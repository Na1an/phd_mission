import argparse as ap
from model import *
from train import *
from preprocess_data import *

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (TLS data with labels).", type=str)
    parser.add_argument("--grid_size", help="The sliding window size.", type=float, default=5.0)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--sample_size", help="The sample size : number of points in one-time training.", type=int, default=5000)
    args = parser.parse_args()

    # take arguments
    raw_data_path = args.data_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    sample_size = args.sample_size

    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    # (1) preprocess data and get set of sliding window coordinates
    print("> input data:", raw_data_path)
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data(raw_data_path, "llabel", detail=True)
    print("\n> data_preprocess.shape =", data_preprocessed.shape)
    print("> grid_size:", grid_size)
    print("> voxel_size:", voxel_size)
    print("> voxel sample mode is:", voxel_sample_mode)
    
    # create a model
    global_height = z_max - z_min # the absolute height
    my_model = PointWiseModel(device=my_device, grid_size=grid_size, voxel_size=voxel_size, global_height=global_height)
    my_model.show_voxel_shape()
    
    # trainer
    my_trainer = Trainer(my_model, my_device)

    # sliding window
    coords_sw = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size)
    (d1,d2,_) = coords_sw.shape
    nb_cuboid = d1 * d2
    #print("> coords.shape={}, size={}".format(coords_sw.shape, coords_sw.size))
    
    global_height = z_max - z_min
    samples, sample_cuboid_index, voxel_skeleton_cuboid = prepare_dataset(data_preprocessed, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size)
    print(">>> samples.shape={}, sample_cuboid_index.shape={}, voxel_skele.len={}".format(samples.shape, sample_cuboid_index.shape, len(voxel_skeleton_cuboid)))
    
    '''
    print(samples[0:10])
    print(samples[0].shape)
    print(sample_cuboid_index[0:10])
    print(sample_cuboid_index[-1])
    '''
    print("\n###### End ######")
        