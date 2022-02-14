from utility import *

def check_classfication(las):
    '''
    Args:
        las : a LasData.
    Returns:
        None.
    '''
    # what classcification is
    print("lala >> :", las[las["classification"] !=0])
    print("lala >> :", np.array(las[las["classification"] != 0]))
    
    return None

def voxel_coloring(d_name):
    '''
    Args:
        d_name : a string. The dimension name of the las header.
    Returns:
        res : pcd.colors for coloring our voxels.   
    '''
    return open3d.utility.Vector3dVector(np.vstack((las[d_name], las[d_name], las[d_name])).transpose()/np.max(las[d_name])) 


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

    # read data
    print("> Reading data from :", target_path)
    las = read_header(target_path)
     
    # print info
    print_info(las)

    # transfert laspoint it to the open3d points
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.vstack((las.x, las.y, las.z)).transpose())
    """
    ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 'classification', 'synthetic', 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time']
    red, green, blue is useless for dls data
    """
    #pcd.colors = open3d.utility.Vector3dVector(np.vstack((las.red, las.green, las.blue)).transpose())
    pcd.colors = voxel_coloring(d_name)

    # voxel size
    v_size = round(max(pcd.get_max_bound() - pcd.get_min_bound())*v_resolution, 4)
    v_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=v_size)
    open3d.visualization.draw_geometries([v_grid])

    # store our voxel cubes
    voxels = v_grid.get_voxels()
    vox_mesh = open3d.geometry.TriangleMesh()

    for v in voxels:
        cube=open3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color(v.color)
        cube.translate(v.grid_index, relative=False)
        vox_mesh+=cube
    
    # storing
    vox_mesh.translate([0.5,0.5,0.5], relative=True)
    vox_mesh.scale(v_size, [0,0,0])
    vox_mesh.translate(v_grid.origin, relative=True)
    vox_mesh.merge_close_vertices(0.0000001)
    open3d.io.write_triangle_mesh(os.getcwd()+"/res_voxel_mesh_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".ply", vox_mesh)
    
    print("###### End! ######")
