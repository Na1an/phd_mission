import copy
from scipy.spatial.transform import Rotation
from utility import *

# This function works for the preprocessing the data
def read_data(path, detail=False):
    '''
    Args:
        path : a string. The path of the data file.
        feature : a string. Which feature we want to keep in the output.
        detail : a bool. False by default.
    Returns:
        res : a 4-D numpy array type tensor.
    '''
    data_las = laspy.read(path)
    #data_las.write("/home/yuchen/Documents/PhD/phd_mission/scripts/New_Subsample_Algo/test.las")
    x_min, x_max, y_min, y_max, z_min, z_max = get_info(data_las)
    data = np.vstack((data_las.x, data_las.y, data_las.z)).transpose()
    print(">>> data.points shape =", data.shape, " type =", type(data))

    return data, x_min, x_max, y_min, y_max, z_min, z_max

# This function works for the preprocessing the data
def read_data_tls(path, detail=False):
    '''
    Args:
        path : a string. The path of the data file.
        feature : a string. Which feature we want to keep in the output.
        detail : a bool. False by default.
    Returns:
        res : a 4-D numpy array type tensor.
    '''
    
    data_las = laspy.read(path)
    
    x_min, x_max, y_min, y_max, z_min, z_max = get_info(data_las)
    # Dimension names: ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'scan_angle_rank', 'point_source_id', 'gps_time', 'deviation', 'WL', 'TreeID']
    data = np.vstack((data_las.x, data_las.y, data_las.z, data_las.intensity, data_las.return_number, data_las.number_of_returns, data_las.scan_direction_flag, data_las.scan_angle_rank, data_las.point_source_id, data_las.gps_time, data_las.deviation, data_las.WL, data_las.TreeID)).transpose()
    print(">>> data.points shape =", data.shape, " type =", type(data))

    return data, x_min, x_max, y_min, y_max, z_min, z_max


# This function works for the preprocessing the data with intensity
def read_data_with_intensity(path, feature, feature2='intensity', detail=False):
    '''
    Args:
        path : a string. The path of the data file.
        feature : a string. Which feature we want to keep in the output.
        detail : a bool. False by default.
    Returns:
        res : a 4-D numpy array type tensor.
    '''
    data_las = laspy.read(path)
    x_min, x_max, y_min, y_max, z_min, z_max = get_info(data_las)
    
    data_las.z = data_las.z - int(z_min)
    print(">> data_las.z min={} max={}".format(np.min(data_las.z), np.max(data_las.z)))

    '''
    # intensity put it here
    intensity_max = np.log(np.max(data_las['intensity']))
    intensity_min = np.log(np.min(data_las['intensity']))
    data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, ((np.log(data_las['intensity'])-intensity_min)/(intensity_max-intensity_min)), data_las[feature])).transpose()
    
    mean_z = np.mean(data_las.z)
    std_z = np.std(data_las.z)
    data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, ((data_las.z - mean_z)/std_z), data_las[feature])).transpose()
    '''
    f2_max = np.log(np.max(data_las[feature2]))
    f2_min = np.log(np.min(data_las[feature2]))
    data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, ((np.log(data_las[feature2])-f2_min)/(f2_max-f2_min)), data_las[feature])).transpose()
    
    print(">>>[!data with intensity] data shape =", data.shape, " type =", type(data))

    return data, x_min, x_max, y_min, y_max, z_min, z_max

# read header
def read_header(path, detail=True):
    '''
    Args:
        path : a string. The path of file.
        detail : a boolean. If we want to see the details.
    Returns:
        The file's header and data. (and VLRS if it has)
    '''
    print("\n> Reading data from :", path)

    las = laspy.read(path)
    nb_points = las.header.point_count
    point_format = las.point_format

    if detail:
        print(">> Point of Data Format:", las)
        print(">> min: x,y,z", np.min(las.x), np.min(las.y), np.min(las.z))
        print(">> max: x,y,z", np.max(las.x), np.max(las.y), np.max(las.z))
        print(">> Number of points:", nb_points)
        print(">> LAS File Format:", point_format.id)
        print(">> Dimension names:", list(point_format.dimension_names))

    return las

# get indice from the study region
def get_region_indice(data, x_min, x_max, y_min, y_max, blank):
    '''
    Args:
        data : a 4-D numpy.darray (x,y,z,label). The data to process.
        *_min/*_max : a interger. The data range.
        blank : a float. The margin of the area.
    Returns:
        res : a list of 2-d coordinates (x,y). The set of grid coordinates.
    '''
    return np.where((((x_min)<data[:,0]) & (data[:,0]<(x_max))) & (((y_min)<data[:,1]) & (data[:,1]<(y_max))))

# to do : decide which kind of key point in each voxel we need. 
# [mean center, closest point to mean center, voxel center]
# voxelization
#def voxel_grid_sample(cuboid, grid_size, height, voxel_size, mode):
def voxel_grid_sample(cuboid, voxel_size, mode):
    '''
    Args:
        points : a (n,4) numpy.darray. The data to process.
        voxel_size : a float. The resolution of the voxel. 
        mode : a string. How to select points in voxel. ('mc': mean_center, 'cmc' : closest point to mean center)
    Returns:
        res : a voxelized data. key points in each voxel.
        nb_points_per_voxel : a list integer. The total voxel number.
        non_empty_voxel : a (n,3) np.darray. The index of occupied voxel.
    '''

    res = []
    points = cuboid[:,:3]

    # non_empy_voxel : no empty voxel :)
    # index : the positions of [new elements in old array]
    # index_inversed : the positions of [old elements in new array]
    # nb_pts_per_voxel : nb of points in each voxels
    no_empty_voxel, index, index_inversed, nb_points_per_voxel = np.unique((points//voxel_size).astype(int), axis=0, return_index=True, return_inverse=True, return_counts=True)
    index_points_on_voxel_sorted = np.argsort(index_inversed)
    # we can then access the points that are linked to each voxel through index_points_on_voxel_sorted and how many they are (nb_pts_per_voxel)

    voxel_grid = {}
    loc_select = 0

    # inner fucntion
    if mode == "mc":
        def key_point_in_voxel(v):
            return np.mean(voxel_grid[tuple(v)],axis=0)
    elif mode == "cmc":
        def key_point_in_voxel(v):
            return voxel_grid[tuple(v)][np.linalg.norm(voxel_grid[tuple(v)] - np.mean(voxel_grid[tuple(v)],axis=0),axis=1).argmin()]
    else:
        raise RuntimeError("Function : voxel_grid_sample, select point mode unknowm (neither mc nor cmc)")

    # i - index, v - coordinate of non empty voxel
    for i,v in enumerate(no_empty_voxel):
        nb_points = nb_points_per_voxel[i]
        voxel_grid[tuple(v)] = points[index_points_on_voxel_sorted[loc_select:loc_select+nb_points]]
        res.append(key_point_in_voxel(v))
        loc_select = loc_select + nb_points
    
    #nb_p_max = np.max(nb_points_per_voxel)
    #nb_p_min = np.min(nb_points_per_voxel)
    print("no_empty_voxel.shape={}, nb_points_per_voxel.shape={}".format(no_empty_voxel.shape, nb_points_per_voxel.shape))
    voxel_and_points = np.append(no_empty_voxel, nb_points_per_voxel.reshape(-1, 1), axis=1)

    return np.array(res), np.array(nb_points_per_voxel), voxel_and_points

def voxel_grid_sample_tls(cuboid, voxel_size):
    '''
    Args:
        points : a (n,4) numpy.darray. The data to process.
        voxel_size : a float. The resolution of the voxel. 
    Returns:
        res : a voxelized data. key points in each voxel.
        nb_points_per_voxel : a list integer. The total voxel number.
        non_empty_voxel : a (n,3) np.darray. The index of occupied voxel.
    '''

    res = []

    # non_empy_voxel : no empty voxel :)
    # index : the positions of [new elements in old array]
    # index_inversed : the positions of [old elements in new array]
    # nb_pts_per_voxel : nb of points in each voxels
    no_empty_voxel, index, index_inversed, nb_points_per_voxel = np.unique((cuboid[:,:3]//voxel_size).astype(int), axis=0, return_index=True, return_inverse=True, return_counts=True)
    index_points_on_voxel_sorted = np.argsort(index_inversed)
    # we can then access the points that are linked to each voxel through index_points_on_voxel_sorted and how many they are (nb_pts_per_voxel)

    voxel_grid = {}
    loc_select = 0

    # i - index, v - coordinate of non empty voxel
    for i,v in enumerate(no_empty_voxel):
        nb_points = nb_points_per_voxel[i]
        voxel_grid[tuple(v)] = cuboid[index_points_on_voxel_sorted[loc_select:loc_select+nb_points]]
        loc_select = loc_select + nb_points
    
    #nb_p_max = np.max(nb_points_per_voxel)
    #nb_p_min = np.min(nb_points_per_voxel)
    print("no_empty_voxel.shape={}, nb_points_per_voxel.shape={}".format(no_empty_voxel.shape, nb_points_per_voxel.shape))
    voxel_and_points = np.append(no_empty_voxel, nb_points_per_voxel.reshape(-1, 1), axis=1)

    return np.array(res), np.array(nb_points_per_voxel), voxel_and_points, voxel_grid

# analyse
def analyse_voxel_in_cuboid_subsample(voxel_skeleton_cuboid, h, side):
    '''
    Args:
        voxel_skeleton_cuboid : a dictionnary.
        h : a float. The height cuboid.
        side : a float. side length.
    Returns:
        res: a voxelized space, indicate each voxel is occupied or not.
    '''
    '''
    print("len(voxel_skeleton_cuboid) =", len(voxel_skeleton_cuboid), " ", type(voxel_skeleton_cuboid))
    print("v_k_c[0]=type",type(voxel_skeleton_cuboid[0]))
    print("v_k_c[0]=",voxel_skeleton_cuboid[0])
    '''
    print(">> voxel_net, h={} side={}".format(h,side))
    res = np.zeros([side, side, h])
    for c in voxel_skeleton_cuboid:
        v_x,v_y,v_z, v_p = c
        res[int(v_x),int(v_y),int(v_z)] = v_p

    return res
# analyse
def analyse_voxel_in_cuboid(voxel_skeleton_cuboid, h, side):
    '''
    Args:
        voxel_skeleton_cuboid : a dictionnary.
        h : a float. The height cuboid.
        side : a float. side length.
    Returns:
        res: a voxelized space, indicate each voxel is occupied or not.
    '''
    '''
    print("len(voxel_skeleton_cuboid) =", len(voxel_skeleton_cuboid), " ", type(voxel_skeleton_cuboid))
    print("v_k_c[0]=type",type(voxel_skeleton_cuboid[0]))
    print("v_k_c[0]=",voxel_skeleton_cuboid[0])
    '''
    print(">> voxel_net, h={} side={}".format(h,side))
    nb_cuboid = len(voxel_skeleton_cuboid)
    print(">> voxel_skeleton_cuboid[0].shape=", voxel_skeleton_cuboid[0].shape)
    res = np.zeros([nb_cuboid, side, side, h])
    for k,v in voxel_skeleton_cuboid.items():
        #print("k=",k, "v.shape=", v.shape)
        # 加个numpy 搜索key什么的,然后赋值给点的数量
        for c in v:
            v_x,v_y,v_z, v_p = c
            res[k,int(v_x),int(v_y),int(v_z)] = v_p

    return res