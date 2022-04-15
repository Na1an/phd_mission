import argparse as ap
from process_data import *

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("tls_path", help="The path of TLS data.", type=str)
    parser.add_argument("dls_path", help="The path of DLS data.", type=str)
    parser.add_argument("dtm_path", help="The path of dtm.", type=str)
    parser.add_argument("--grid_size", help="The sliding window size.", type=float, default=5.0)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    args = parser.parse_args()

    # take arguments
    tls_path = args.tls_path
    dls_path = args.dls_path
    dtm_path = args.dtm_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    
    # set by default
    voxel_sample_mode = 'cmc'

    # (1) preprocess data and get set of sliding window coordinates
    print("> input data tls_path:", tls_path)
    print("> input data dls_path:", dls_path)
    print("> input data dtm_path:", dtm_path)

    print("\n> grid_size:", grid_size)
    print("> voxel_size:", voxel_size)
    #print("> voxel sample mode is:", voxel_sample_mode)
    
    tls_data_processed, x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t = read_data(tls_path, detail=True)
    print("\n> tls_data_data_preprocess.shape =", tls_data_processed.shape)
    
    dls_data_processed, x_min_d, x_max_d, y_min_d, y_max_d, z_min_d, z_max_d = read_data(dls_path, detail=True)
    print("\n> dls_data_data_preprocess.shape =", tls_data_processed.shape)

    dtm_data_processed, x_min, x_max, y_min, y_max, z_min, z_max = read_data(dtm_path, detail=True)
    print("\n> dtm_data_data_preprocess.shape =", tls_data_processed.shape)
    
    
    # coordinates features
    tls_height = z_max_t - z_min_t
    dls_height = z_max_d - z_min_d 
    dtm_height = z_max - z_min # the absolute height dtm

    # find overlap region
    x_min_overlap = max(x_min, x_min_t)
    y_min_overlap = max(y_min, y_min_t)
    x_max_overlap = min(x_max, x_max_t)
    y_max_overlap = min(y_max, y_max_t)

    # index interesting
    #index_tls = get_region_index(tls_data_processed, x_min_overlap, x_min_overlap+grid_size, y_min_overlap, y_min_overlap+grid_size)
    #index_dls = get_region_index(dls_data_processed, x_min_overlap, x_min_overlap+grid_size, y_min_overlap, y_min_overlap+grid_size)
    index_dtm = get_region_index(dtm_data_processed, x_min_overlap, x_min_overlap+grid_size, y_min_overlap, y_min_overlap+grid_size)

    #tls_points = tls_data_processed[index_tls]
    #dls_points = tls_data_processed[index_dls]
    dtm_points = tls_data_processed[index_dtm]

    dtm_points[:,0] = dtm_points[:,0] - x_min_t
    dtm_points[:,1] = dtm_points[:,1] - y_min_t

    dtm_voxel_key_points, dtm_nb_points_per_voxel, dtm_voxel = voxel_grid_sample(dtm_points, voxel_size, voxel_sample_mode)
    print(">> voxel.shape :",voxel.shape)
    print(">> nb_points_per_voxel.shape :",nb_points_per_voxel.shape)
    #print(voxel[0:10])
    visualize_voxel_key_points(dtm_voxel, dtm_nb_points_per_voxel, "voxel dtm")

    '''
    # sliding window
    coords_sw = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size)
    (d1,d2,_) = coords_sw.shape
    #print("> coords :", coords_sw)

    # to do : add more overlap between the cubes
    # beta version
    # record of point numbers in each cube
    w_nb = 0
    tmp = []
    global_height = z_max - z_min

    for i in range(d1):
        for j in range(d2):
            w_nb = w_nb + 1
            # (1) cut data to cubes
            # local origin
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')
            
            # find index of the tls_data_processed in this sliding window
            local_index = get_region_index(tls_data_processed, local_x, local_x+grid_size, local_y, local_y+grid_size)

            # shift points to local origin (0, 0, 0)
            local_points = tls_data_processed[local_index]
            #if local_points.size < 1550000:
            if local_points.size < 10:
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
            #print(">> local_points:", local_points[0:10])
            tmp.append(local_points.size)

            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            print(">> voxel.shape :",voxel.shape)
            print(">> nb_points_per_voxel.shape :",nb_points_per_voxel.shape)
            #print(voxel[0:10])
            visualize_voxel_key_points(voxel, nb_points_per_voxel, "voxel - cuboid "+str(w_nb))

            #print(key_points_in_voxel[0:10])
            #print(key_points_in_voxel.shape)
            #visualize_voxel_key_points(key_points_in_voxel, nb_points_per_voxel, voxel_sample_mode + " key points in voxel - cuboid "+str(w_nb) + "ratio nb_point/max(nb_point)")
        
    #print(tmp)
    print("\n###### End ######")
    '''