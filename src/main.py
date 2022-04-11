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
    args = parser.parse_args()

    # take arguments
    raw_data_path = args.data_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    
    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    # create a model 
    my_model = PointWiseModel(device=my_device)
    my_trainer = Trainer(my_model, my_device)

    # (1) preprocess data and get set of sliding window coordinates
    print("> input data:", raw_data_path)
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data(raw_data_path, "WL", detail=True)
    print("\n> data_preprocess.shape =", data_preprocessed.shape)
    print("> grid_size:", grid_size)
    print("> voxel_size:", voxel_size)
    print("> voxel sample mode is:", voxel_sample_mode)

    coords_sw = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size)
    (d1,d2,_) = coords_sw.shape
    #print("> coords :", coords_sw)

    # to do : add more overlap between the cubes
    # beta version
    w_nb = 1

    # record of point numbers in each cube
    tmp = []
    global_height = z_max - z_min
    for i in range(d1):
        for j in range(d2):
            # (1) cut data to cubes
            # local origin
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')
            w_nb = w_nb + 1
            
            # find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data_preprocessed, local_x, local_x+grid_size, local_y, local_y+grid_size)

            # shift points to local origin (0, 0, 0)
            local_points = data_preprocessed[local_index]
            #if local_points.size < 1550000:
            if local_points.size < 10000:
                print(">> Local_points is empty, no points founds here!")
                continue
            
            local_z_min = np.min(local_points[:,2])
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_points[:,2] = local_points[:,2] - np.min(local_points[:,2])
            local_abs_height = np.max(local_points[:,2])
            print(">> local abs height : ", local_abs_height)
            print(">> local data.shape :", local_points.shape)
            print(">> local data shifted")
            tmp.append(local_points.size)
            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, grid_size, global_height, voxel_size, voxel_sample_mode)
            #print(key_points_in_voxel[0:10])
            #print(key_points_in_voxel.shape)
            print(">> nb_points_per_voxel.shape :",nb_points_per_voxel.shape)
            print(">> voxel.shape :",voxel.shape)
            print(voxel[0:10])
            #visualize_voxel_key_points(key_points_in_voxel, nb_points_per_voxel, "key points in voxel - cuboid "+str(w_nb))
            visualize_voxel_key_points(voxel, nb_points_per_voxel, "voxel - cuboid "+str(w_nb))
            # (2) put cube data to device : (cpu or gpu)


            # (3) feed it to the model

            break
        
    #print(tmp)
    print("\n###### End ######")
        