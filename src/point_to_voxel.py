from utility import *

if __name__ == "__main__":
    
    # build arguments
    parser = ap.ArgumentParser(description="Convert lidar to BEV, support only .las for the moment.")
    parser.add_argument("target", help="The path of the target file.", type=str)
    parser.add_argument("--v_resolution", help="The voxel resolution of the study field.", type=float, default=0.05)
    parser.add_argument("--d_name", help="Choose one dimension name to color the voxels.", type=str, default="intensity")
    args = parser.parse_args()

    # take parameters
    target_path = args.target
    v_resolution = args.v_resolution
    d_name = args.d_name
    print("> resolution set to :", v_resolution, "- study dimension name :", d_name)
    
    # read data
    print("> Reading data from :", target_path)
    las = read_header(target_path)

    #print_info(las)
    
    # transfert laspoint it to the open3d points
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.vstack((las.x, las.y, las.z)).transpose())
    """
    ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 'classification', 'synthetic', 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time']
    red, green, blue is useless for dls data
    """
    pcd.colors = voxel_coloring(las, d_name)

    # voxel size
    v_size = round(max(pcd.get_max_bound() - pcd.get_min_bound())*v_resolution, 4)
    v_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=v_size)
    open3d.visualization.draw_geometries([v_grid])
    
    # store the result as mesh/voxel
    sotre_as_mesh(v_grid, v_size)
    
    print("###### End! ######")