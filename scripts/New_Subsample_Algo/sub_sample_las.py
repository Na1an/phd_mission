import random
import argparse as ap
from preprocess_subsample import *
from datetime import datetime

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("tls_path", help="The path of TLS data.", type=str)
    parser.add_argument("dls_path", help="The path of DLS data.", type=str)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--nb_sample", help="How many sample we want.", type=int, default=4)
    args = parser.parse_args()

    # take arguments
    tls_path = args.tls_path
    dls_path = args.dls_path
    voxel_size = args.voxel_size
    nb_sample = args.nb_sample

    # (1) preprocess data and get set of sliding window coordinates
    print("> input data tls_path:", tls_path)
    print("> input data dls_path:", dls_path)

    print("> input data tls_path:", tls_path)
    tls_data_processed, x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t = read_data_tls(tls_path, detail=True)
    print("> tls_data_preprocess.shape =", tls_data_processed.shape, '\n')
    
    print("> input data dls_path:", dls_path)
    dls_data_processed, x_min_d, x_max_d, y_min_d, y_max_d, z_min_d, z_max_d = read_data(dls_path, detail=True)
    print("> dls_data_preprocess.shape =", dls_data_processed.shape, '\n')

    # find overlap region
    x_min_overlap = max(x_min_d, x_min_t)
    y_min_overlap = max(y_min_d, y_min_t)
    x_max_overlap = min(x_max_d, x_max_t)
    y_max_overlap = min(y_max_d, y_max_t)
    height_min = max(z_min_t,z_min_d) + 2.5

    print("x_min_overlap={}, y_min_overlap={}, x_max_overlap={}, y_max_overlap={}".format(x_min_overlap, y_min_overlap, x_max_overlap, y_max_overlap))

    dls_data_processed_selected = dls_data_processed[get_region_index(dls_data_processed, x_min_overlap, x_max_overlap, y_min_overlap, y_max_overlap)]
    print("dls_data_selected.shape={}".format(dls_data_processed_selected.shape))
    
    dls_data_processed_selected = dls_data_processed_selected[np.where(dls_data_processed_selected[:,2]>height_min)]
    dls_data_processed_selected[:,0] = dls_data_processed_selected[:,0] - x_min_overlap
    dls_data_processed_selected[:,1] = dls_data_processed_selected[:,1] - y_min_overlap
    #dls_data_processed_selected[:,2] = dls_data_processed_selected[:,2] - height_min

    # here we have our 
    _, _, dls_voxelized = voxel_grid_sample(dls_data_processed_selected, voxel_size, 'mc')
    print("voxelized_dls.shape={}".format(dls_voxelized.shape))
    
    tls_data_processed = tls_data_processed[np.where(tls_data_processed[:,2]>height_min)]    
    tls_data_processed[:,0] = tls_data_processed[:,0] - x_min_overlap
    tls_data_processed[:,1] = tls_data_processed[:,1] - y_min_overlap
    #tls_data_processed[:,2] = tls_data_processed[:,2] - height_min
    
    # voxelize tls data
    _,_,_, voxel_grid_tls = voxel_grid_sample_tls(tls_data_processed, voxel_size)
    #print("len(voxel_grid_tls)={}".format(len(voxel_grid_tls)))

    #dls_point_nb_in_voxel = analyse_voxel_in_cuboid_subsample(dls_voxelized, h, side)
    
    res = np.zeros((nb_sample, len(dls_data_processed_selected), 13))
    nb_point_total = 0
    missed_point = 0
    for e in dls_voxelized:
        if tuple(e[:3]) in voxel_grid_tls:
            tls_point_nb = len(voxel_grid_tls[tuple(e[:3])])
            dls_point_nb = e[3]
            #print("tls point nb ={}, dls point nb={}".format(tls_point_nb, dls_point_nb))
            
            if tls_point_nb < dls_point_nb:
                print("tls point nb ={}, dls point nb={}, tls_point_nb < dls_point_nb, skip this step".format(tls_point_nb, dls_point_nb))
                nb_point_total = nb_point_total + dls_point_nb
                missed_point = missed_point + dls_point_nb
                continue
            
            index_list = [index for index in range(len(voxel_grid_tls[tuple(e[:3])]))]
            for i in range(nb_sample):
                index_list = random.sample(index_list, e[3])
                res[i][nb_point_total:nb_point_total + e[3]] = voxel_grid_tls[tuple(e[:3])][index_list,:]
            nb_point_total = nb_point_total + e[3]

    print("dls_like_data : nb_points_totale={}, nb_missed_point={} (percentage={})".format(nb_point_total, missed_point, missed_point/nb_point_total))

    for i in range(nb_sample):
        print("result{} is working!".format(i))
        '''
        header = laspy.header.LasHeader()
        # Dimension names: ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 
        # 'scan_angle_rank', 'point_source_id', 'gps_time', 'deviation', 'WL', 'TreeID']
        header.add_extra_dim(laspy.ExtraBytesParams(name="intensity", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="return_number", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="number_of_returns", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="scan_direction_flag", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="scan_angle_rank", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="point_source_id", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="gps_time", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="deviation", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="WL", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="TreeID", type=np.float64))
        '''

        las = laspy.create(point_format=3)
        #las.add_extra_dim(laspy.ExtraBytesParams(name="intensity", type=np.float64))
        las.add_extra_dim(laspy.ExtraBytesParams(name="return_number", type=np.float64))
        las.add_extra_dim(laspy.ExtraBytesParams(name="number_of_returns", type=np.float64))
        las.add_extra_dim(laspy.ExtraBytesParams(name="scan_direction_flag", type=np.float64))
        #las.add_extra_dim(laspy.ExtraBytesParams(name="scan_angle_rank", type=np.float64))
        #las.add_extra_dim(laspy.ExtraBytesParams(name="point_source_id", type=np.float64))
        #las.add_extra_dim(laspy.ExtraBytesParams(name="gps_time", type=np.float64))
        las.add_extra_dim(laspy.ExtraBytesParams(name="deviation", type=np.float64))
        las.add_extra_dim(laspy.ExtraBytesParams(name="WL", type=np.float64))
        las.add_extra_dim(laspy.ExtraBytesParams(name="TreeID", type=np.float64))

        las.x = res[i][:,0] + x_min_overlap
        las.y = res[i][:,1] + y_min_overlap
        las.z = res[i][:,2]
        las.intensity = res[i][:,3]
        las.return_number = res[i][:,4]
        las.number_of_returns = res[i][:,5]
        las.scan_direction_flag = res[i][:,6]
        las.scan_angle_rank = res[i][:,7]
        las.point_source_id = res[i][:,8]
        las.gps_time = res[i][:,9]
        las.deviation = res[i][:,10]
        las.WL = res[i][:,11]
        las.TreeID = res[i][:,12]

        path = os.getcwd()+"/dls_like_data_{}_".format(i) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".las"
        las.write(path)

        
