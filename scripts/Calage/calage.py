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
    parser.add_argument("--layer_bot", help="The bottom of layer.", type=float, default=2.0)
    parser.add_argument("--layer_top", help="The top of layer.", type=float, default=8.0)
    args = parser.parse_args()

    # take arguments
    tls_path = args.tls_path
    dls_path = args.dls_path
    dtm_path = args.dtm_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    layer_bot = args.layer_bot
    layer_top = args.layer_top
    
    # set by default
    voxel_sample_mode = 'mc'
    layer_height = layer_top - layer_bot

    # (1) preprocess data and get set of sliding window coordinates
    print("> input data tls_path:", tls_path)
    print("> input data dls_path:", dls_path)
    print("> input data dtm_path:", dtm_path)

    print("\n> grid_size:", grid_size)
    print("> voxel_size:", voxel_size)
    print("> layer_bot={}, layer_top={}, layer_height={}".format(layer_bot,layer_top, layer_top-layer_bot))
    print("> layer_bot_voxel={}, layer_top_voxel={}, voxel_height={}".format((layer_bot+0.000001)//voxel_size,(layer_top+0.000001)//voxel_size, ((layer_top-layer_bot)+0.000001)//voxel_size))
    #print("> voxel sample mode is:", voxel_sample_mode)

    print("> input data dls_path:", dls_path)
    tls_data_processed, x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t = read_data(tls_path, detail=True)
    print("\n> tls_data_preprocess.shape =", tls_data_processed.shape)

    dls_data_processed, x_min_d, x_max_d, y_min_d, y_max_d, z_min_d, z_max_d = read_data(dls_path, detail=True)
    print("\n> dls_data_preprocess.shape =", dls_data_processed.shape)

    dtm_data_processed, x_min, x_max, y_min, y_max, z_min, z_max = read_data(dtm_path, detail=True)
    print("\n> dtm_data_preprocess.shape =", dtm_data_processed.shape)
    
    
    # coordinates features
    tls_height = z_max_t - z_min_t
    dls_height = z_max_d - z_min_d 
    dtm_height = z_max - z_min # the absolute height dtm

    # find overlap region
    x_min_overlap = max(x_min, x_min_t)
    y_min_overlap = max(y_min, y_min_t)
    x_max_overlap = min(x_max, x_max_t)
    y_max_overlap = min(y_max, y_max_t)

    print("> x_min_overlap={}, y_min_overlap={}, x_max_overlap={}, y_max_overlap={}".format(x_min_overlap, y_min_overlap, x_max_overlap, y_max_overlap))
    print("> overlap region shape = {} * {}". format(x_max_overlap - x_min_overlap, y_max_overlap - y_min_overlap))

    # index interesting
    index_tls = get_region_index(tls_data_processed, x_min_overlap, x_min_overlap+grid_size, y_min_overlap, y_min_overlap+grid_size)
    index_dls = get_region_index(dls_data_processed, x_min_overlap, x_min_overlap+grid_size, y_min_overlap, y_min_overlap+grid_size)
    index_dtm = get_region_index(dtm_data_processed, x_min_overlap, x_min_overlap+grid_size, y_min_overlap, y_min_overlap+grid_size)

    tls_points = tls_data_processed[index_tls]
    dls_points = dls_data_processed[index_dls]
    dtm_points = dtm_data_processed[index_dtm]

    # normalization, shift to (0,0)
    tls_points[:,0] = tls_points[:,0] - x_min_overlap
    tls_points[:,1] = tls_points[:,1] - y_min_overlap
    dls_points[:,0] = dls_points[:,0] - x_min_overlap
    dls_points[:,1] = dls_points[:,1] - y_min_overlap
    dtm_points[:,0] = dtm_points[:,0] - x_min_overlap
    dtm_points[:,1] = dtm_points[:,1] - y_min_overlap

    print("> dtm_points.shape = {}".format(dtm_points.shape))

    # 3-1. voxelize dtm
    dtm_voxel_key_points, dtm_nb_points_per_voxel, dtm_voxel = voxel_grid_sample(dtm_points, voxel_size, voxel_sample_mode)
    print(">> dtm_voxel.shape :",dtm_voxel.shape)
    print(">> dtm_nb_points_per_voxel.shape :",dtm_nb_points_per_voxel.shape)
    #print(">> dtm_voxel_key_points:", dtm_voxel_key_points[0:10])
    #visualize_voxel_key_points(dtm_voxel, dtm_nb_points_per_voxel, "voxel dtm")
    
    # 3-2. find the bottom of the dtm
    index_dtm_bottom_voxel = bottom_voxel(dtm_voxel)
    visualize_voxel_key_points(dtm_voxel[index_dtm_bottom_voxel], dtm_nb_points_per_voxel[index_dtm_bottom_voxel], "voxelized dtm bottom")
    #write_data(dtm_voxel[index_dtm_bottom_voxel], "dtm_bottom")

    # 3-3 voxelize tls
    tls_voxel_key_points, tls_nb_points_per_voxel, tls_voxel, tls_voxel_grid = voxel_grid_sample(tls_points, voxel_size, voxel_sample_mode, get_voxel_grid=True)
    #visualize_voxel_key_points(tls_voxel, tls_nb_points_per_voxel, "voxelized tls ")
    dls_voxel_key_points, dls_nb_points_per_voxel, dls_voxel, dls_voxel_grid = voxel_grid_sample(dls_points, voxel_size, voxel_sample_mode, get_voxel_grid=True)
    #visualize_voxel_key_points(dls_voxel, dls_nb_points_per_voxel, "voxelized dls ")
    
    #write_data(tls_voxel, "voxel_tls")
    #write_data(dls_voxel, "voxel_dls")

    # (layer_height, n, 3)
    '''
    voxel_layer = slice_voxel_data(dtm_voxel[index_dtm_bottom_voxel], layer_bot, layer_top, voxel_size, tls_voxel_grid)
    
    print("> voxel_layer.shape =", voxel_layer.shape)
    print(">> voxel_mayer[0:20]", voxel_layer[0:20])
    visualize_voxel_key_points(voxel_layer, voxel_layer, "what we want is here!", only_points=True)
    write_data(voxel_layer, "voxel_layer")
    '''
    layer_tls, layer_dls, nb_voxel_tls, nb_voxel_dls, nb_voxel_coi, coi_voxel = slice_voxel_data_and_find_coincidence(dtm_voxel[index_dtm_bottom_voxel], layer_bot, layer_top, voxel_size, tls_voxel_grid, dls_voxel_grid)
    print("> voxel_layer_tls.shape =", layer_tls.shape, "voxel_layer_dls.shape =", layer_dls.shape)
    #visualize_voxel_key_points(layer_tls, layer_tls, "voxel_layer_tls", only_points=True)
    visualize_voxel_key_points(layer_dls, layer_dls, "voxel_layer_dls", only_points=True)
    visualize_voxel_key_points(coi_voxel, coi_voxel, "both", only_points=True)

    write_data(layer_tls, "layer_tls", x_min_overlap, y_min_overlap)
    write_data(layer_dls, "layer_dls", x_min_overlap, y_min_overlap)
    write_data(coi_voxel*voxel_size, "coi_voxel", x_min_overlap, y_min_overlap)
    