import argparse as ap
from model import *
from trainer import *
from build_dataset import *
from preprocess_data import *

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (TLS data with labels).", type=str)
    parser.add_argument("--grid_size", help="The sliding window size.", type=float, default=5.0)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--sample_size", help="The sample size : number of points in one-time training.", type=int, default=5000)
    parser.add_argument("--nb_epoch", help="The epoch number.", type=int, default=300)
    args = parser.parse_args()

    # take arguments
    raw_data_path = args.data_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    sample_size = args.sample_size
    nb_epoch = args.nb_epoch

    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    
    # (2) prepare train dataset and validation dataset
    samples_train, sample_cuboid_index_train, train_voxel_nets = prepare_procedure(raw_data_path, grid_size, voxel_size, voxel_sample_mode, sample_size, detail=True)
    train_dataset = TrainDataSet(samples_train, sample_cuboid_index_train, my_device)
    train_dataset.show_info()
    
    samples_val, sample_cuboid_index_val, val_voxel_nets = prepare_procedure(raw_data_path, grid_size, voxel_size, voxel_sample_mode, sample_size, detail=True)
    val_dataset = TrainDataSet(samples, sample_cuboid_index, my_device)
    val_dataset.show_info()

    # (3) create model and trainning

    # create a model
    global_height = z_max - z_min # the absolute height, set to 50 for the moment
    my_model = PointWiseModel()

    my_trainer = Trainer(
                my_model, 
                my_device, 
                train_dataset=train_dataset,
                train_voxel_nets=train_voxel_nets,
                val_dataset=val_dataset,
                val_voxel_nets=val_voxel_nets,
                batch_size=4,
                num_workers=0)

    my_trainer.train_model(nb_epoch=nb_epoch)
    
    print("\n###### End ######")

'''
    # (1) preprocess data and get set of sliding window coordinates
    print("> input data:", raw_data_path)
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data(raw_data_path, "llabel", detail=True)
    print("\n> data_preprocess.shape =", data_preprocessed.shape)
    print("> grid_size:", grid_size)
    print("> voxel_size:", voxel_size)
    print("> voxel sample mode is:", voxel_sample_mode)

    # sliding window
    coords_sw = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size)
    (d1,d2,_) = coords_sw.shape
    nb_cuboid = d1 * d2
    #print("> coords.shape={}, size={}".format(coords_sw.shape, coords_sw.size))
    
    #global_height = z_max - z_min
    global_height = 50
    samples, sample_cuboid_index, voxel_skeleton_cuboid = prepare_dataset(data_preprocessed, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=False)
    print(">>> samples.shape={}, sample_cuboid_index.shape={}, voxel_skele.len={}".format(samples.shape, len(sample_cuboid_index), len(voxel_skeleton_cuboid)))
    print("len(voxel_skeleton_cuboid) =", len(voxel_skeleton_cuboid), " ", type(voxel_skeleton_cuboid))
    print("v_k_c[0]=type",type(voxel_skeleton_cuboid[0]))
    print("v_k_c[0]=",voxel_skeleton_cuboid[0])
    voxel_nets = analyse_voxel_in_cuboid(voxel_skeleton_cuboid, int(global_height/voxel_size), int(grid_size/voxel_size))
    print("voxel_nets.shape=", voxel_nets.shape)
    unique,count = np.unique(voxel_nets, return_counts=True)
    data_count = dict(zip(unique, count))
    print("> data_count", data_count)
'''
        