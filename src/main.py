import argparse as ap
from preprocess_data import *

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (TLS data with labels).", type=str)
    parser.add_argument("--grid_size", help="The sliding window size.", type=float, default=5.0)
    args = parser.parse_args()

    # take parameters
    raw_data_path = args.data_path
    grid_size = args.grid_size

    # (1) preprocess data and get set of sliding window coordinates
    path = "/home/yuchen/Documents/PhD/data_for_project/22-03-23_tls_labelled_data/res_bis_crop.las"
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data(path, "WL", detail=True)
    print("\n> data_preprocess.shape =", data_preprocessed.shape)

    coords_sw = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size)
    (d1,d2,_) = coords_sw.shape
    print("> coords :", coords_sw)

    # to do : add more overlap between the cubes
    # beta version
    w_nb = 1
    for i in range(d1):
        for j in range(d2):
            # local origin
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')
            # find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data_preprocessed, local_x, local_x+grid_size, local_y, local_y+grid_size)
            # shift points to local origin (0, 0, 0)
            local_points = data_preprocessed[local_index]
            local_z_min = np.min(local_points[:,2])
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_points[:,2] = local_points[:,2] - np.min(local_points[:,2])
            local_abs_height = np.max(local_points[:,2])
            print(">> local abs height : ", local_abs_height)
            print(">> local data.shape :", local_points.shape)
            print(">> local data shifted")
            #print(local_points[0:10])

            w_nb = w_nb + 1
            #break
        #break

    print("\n###### End ######")
        