import copy
from scipy.spatial.transform import Rotation
from utility import *

# This function works for the preprocessing the data
def read_data(path, feature, detail=False):
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
    data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, data_las[feature])).transpose()
    print(">>> data shape =", data.shape, " type =", type(data))

    return data, x_min, x_max, y_min, y_max, z_min, z_max

# This function works for the preprocessing the data with intensity
def read_data_with_intensity(path, feature, detail=False):
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
    
    '''
    # intensity put it here
    intensity_max = np.log(np.max(data_las['intensity']))
    intensity_min = np.log(np.min(data_las['intensity']))
    data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, ((np.log(data_las['intensity'])-intensity_min)/(intensity_max-intensity_min)), data_las[feature])).transpose()
    '''
    mean_z = np.mean(data_las.z)
    std_z = np.std(data_las.z)
    data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, ((data_las.z - mean_z/std_z)), data_las[feature])).transpose()

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

    return laspy.read(path)

# return the set of sliding window coordinates
def sliding_window(x_min, x_max, y_min, y_max, grid_size):
    '''
    Args:
        *_min/*_max : a interger. The data range.
        grid_size : a interger/float. The side length of the grid.
    Returns:
        res : a list of 2-d coordinates (x,y). The set of grid coordinates.
    '''

    divide_x = int(np.ceil((x_max - x_min)/grid_size))
    divide_y = int(np.ceil((y_max - y_min)/grid_size))
    
    coor_x = np.zeros(divide_x)
    overlap_x = (divide_x * grid_size - (x_max - x_min))/(divide_x-1)
    for i in range(divide_x):
        coor_x[i] = i*grid_size - overlap_x*i
    
    coor_y = np.zeros(divide_y)
    overlap_y = (divide_y * grid_size - (y_max - y_min))/(divide_y-1)
    for j in range(divide_y):
        coor_y[j] = j*grid_size - overlap_y*j

    mesh_x, mesh_y = np.meshgrid(coor_x, coor_y)

    return np.stack((mesh_x,mesh_y), 2)

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
        #grid_size : a interger/float. The side length of a grid.
        #height : a float. The max height of the raw data. Not local height!
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
    #tmp = copy.deepcopy(points//voxel_size).astype(int)
    non_empty_voxel, index, index_inversed, nb_points_per_voxel = np.unique((points//voxel_size).astype(int), axis=0, return_index=True, return_inverse=True, return_counts=True)
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
    for i,v in enumerate(non_empty_voxel):
        nb_points = nb_points_per_voxel[i]
        voxel_grid[tuple(v)] = points[index_points_on_voxel_sorted[loc_select:loc_select+nb_points]]
        res.append(key_point_in_voxel(v))
        loc_select = loc_select + nb_points
        
    return np.array(res), np.array(nb_points_per_voxel), non_empty_voxel

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
    print(voxel_skeleton_cuboid[0].shape)
    res = np.zeros([nb_cuboid, side, side, h])
    for k,v in voxel_skeleton_cuboid.items():
        #print("k=",k, "y.shape=", v.shape)
        # 加个numpy 搜索key什么的,然后赋值给点的数量
        for c in v:
            x,y,z = c
            res[k,x,y,z] = 1

    return res

# for prepare dataset
def prepare_dataset(data, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=False):
    '''
    Args:
        data: a numpy.ndarray (x,y,z,label). 
        coords_sw: (cuboid_x, cuboid_y, 2). The coordinates of sliding window.
        grid_size: The cuboid length and width.
        voxel_size: The voxel size.
        global_height: a float. The global height, our cuboid height.
        voxel_sample_mode: a string. mc or cmc.
        sample_size: how many points in a sample.
    Returns:
        samples: (nb_sample, 5000, 4 :x + y + z + label).
        sample_cuboid_index: (nb_sample, index of nb_cuboid).
        voxel_skeleton_cuboid: (nb_voxel, 4:x+y+z+[1 or 0]).
    '''

    # get voxel_skeleton_cuboid
    (coord_x,coord_y,_) = coords_sw.shape
    nb_cuboid = coord_x * coord_y
    cub_s_nb = int(grid_size/voxel_size)
    cub_h_nb = int(global_height/voxel_size)
    
    # returns
    samples = []
    sample_cuboid_index = {}
    voxel_skeleton_cuboid = {}

    w_nb = 0
    nb_sample = 0
    count_voxel_skeleton = 0
    # inner class
    def voxelization_and_centralization(local_points_inner, window_nb):
        # (4) voxelization
        key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points_inner, voxel_size, voxel_sample_mode)
        voxel_skeleton_cuboid[window_nb] = voxel
        #visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

        # (5) centralization in (x y z) thress axis by the center of voxels
        local_points_inner[:,:3] = local_points_inner[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
        local_points_inner[:,:2] = local_points_inner[:,:2]/grid_size
        local_points_inner[:,2] = local_points_inner[:,2]/global_height
        return None
    
    for i in range(coord_x):
        for j in range(coord_y):   
            # (1) global coordinates of each cuboids
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')

            # (2) find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data, local_x, local_x+grid_size, local_y, local_y+grid_size)
            
            # (3) shift points to local origin (0, 0, 0) and zero-centered
            #local_points = copy.deepcopy(data[local_index])
            local_points = data[local_index]
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_abs_height = np.max(local_points[:,2]) - np.min(local_points[:,2])
            # local_abs_height
            local_points[:,2] = local_points[:,2] - np.min(local_points[:,2])

            if detail:
                print(">>> local abs height :", local_abs_height)
                print(">>> local data.shape :", local_points.shape)
                print(">>> local_data (points in cuboid) zero-centered and standardization/normalization")
                #print(">>> local_points=", local_points[:,:3][0:100])

            # the number of local_points
            tmp_nb_sample = int(len(local_points)/sample_size)

            # (4) voxelization
            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            voxel_skeleton_cuboid[count_voxel_skeleton] = voxel
            #visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

            # (5) centralization in (x y z) thress axis by the center of voxels
            #local_points[:,:3] = local_points[:,:3] - np.mean(local_points[:, :3], axis=0)
            local_points_tmp = local_points.copy()
            local_points_tmp[:,:3] = local_points_tmp[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
            local_points_tmp[:,:2] = local_points_tmp[:,:2]/grid_size
            local_points_tmp[:,2] = local_points_tmp[:,2]/global_height
            
            #print(">>> nb_sample={}".format(nb_sample))
            # set sample_cuboid_index
            for i_s in range(nb_sample, nb_sample+tmp_nb_sample):
                sample_cuboid_index[i_s] = count_voxel_skeleton
            count_voxel_skeleton = count_voxel_skeleton + 1
            
            nb_sample = nb_sample + tmp_nb_sample
            if detail:
                print(">>> tmp_nb_sample={}, nb_sample+tmp={}".format(tmp_nb_sample, nb_sample))

            # set samples
            np.random.shuffle(local_points_tmp)
            tmp_samples = [local_points_tmp[sample_size*i_t:sample_size*(i_t+1)] for i_t in range(tmp_nb_sample)]
            [samples.append(item) for item in tmp_samples]

            
            # data augmentation
            rotation = Rotation.from_euler('z', [90, 180, 270], degrees=True)
            for angle in range(3):
                print(">>> [data augmentation - sample rotated:{}°]".format((angle+1)*90))
                #rotation[angle]
                local_points_tmp = local_points.copy()
                local_points_tmp[:,:3] = local_points_tmp[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
                local_points_tmp[:,:3] = rotation[angle].apply(local_points_tmp[:,:3])
                local_points_tmp[:,:3] = local_points_tmp[:,:3] + np.array([grid_size/2, grid_size/2, global_height/2])

                # (4) voxelization
                key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points_tmp, voxel_size, voxel_sample_mode)
                voxel_skeleton_cuboid[count_voxel_skeleton] = voxel
                #visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

                local_points_tmp[:,:3] = local_points_tmp[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
                local_points_tmp[:,:2] = local_points_tmp[:,:2]/grid_size
                local_points_tmp[:,2] = local_points_tmp[:,2]/global_height
                
                # set sample_cuboid_index
                for i_s in range(nb_sample, nb_sample+tmp_nb_sample):
                    sample_cuboid_index[i_s] = count_voxel_skeleton
                count_voxel_skeleton = count_voxel_skeleton + 1
                nb_sample = nb_sample + tmp_nb_sample

                if detail:
                    print(">>> tmp_nb_sample={}, nb_sample+tmp={}".format(tmp_nb_sample, nb_sample))
                
                # set samples
                np.random.shuffle(local_points_tmp)
                tmp_samples = [local_points_tmp[sample_size*i_t:sample_size*(i_t+1)] for i_t in range(tmp_nb_sample)]
                [samples.append(item) for item in tmp_samples]
                
            # sliding_window added 1
            w_nb = w_nb + 1
                
    return np.array(samples), sample_cuboid_index, voxel_skeleton_cuboid

def prepare_procedure(path, grid_size, voxel_size, voxel_sample_mode, sample_size, label_name="llabel", detail=False, naif_sliding=False):
    '''
    Args:
        path : raw_data_path. The path of training/validation/test file.
        detail : a bool. If we want to show the details below.
    Returns:
    '''
    # (1) preprocess data and get set of sliding window coordinates
    print("> input data:", path)
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data_with_intensity(path, label_name, detail=True)
    print("\n> data_preprocess.shape =", data_preprocessed.shape)
    
    # sliding window
    if naif_sliding:
        print(">> ok, we do naif sliding")
        coords_sw = sliding_window_naif(0, x_max - x_min, 0, y_max - y_min, grid_size)
    else:
        coords_sw = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size)

    (d1,d2,_) = coords_sw.shape
    nb_cuboid = d1 * d2
    #print("> coords.shape={}, size={}".format(coords_sw.shape, coords_sw.size))
    
    #global_height = z_max - z_min
    global_height = 50
    samples, sample_cuboid_index, voxel_skeleton_cuboid = prepare_dataset(data_preprocessed, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=detail)
    print(">>> samples.shape={}, sample_cuboid_index.shape={}, voxel_skele.len={}".format(samples.shape, len(sample_cuboid_index), len(voxel_skeleton_cuboid)))
    
    voxel_nets = analyse_voxel_in_cuboid(voxel_skeleton_cuboid, int(global_height/voxel_size), int(grid_size/voxel_size))
    
    unique,count = np.unique(voxel_nets, return_counts=True)
    data_count = dict(zip(unique, count))
    
    if detail:
        print("> grid_size:", grid_size)
        print("> voxel_size:", voxel_size)
        print("> voxel sample mode is:", voxel_sample_mode)
        print("len(voxel_skeleton_cuboid) =", len(voxel_skeleton_cuboid), " ", type(voxel_skeleton_cuboid))
        print("v_k_c[0]=type",type(voxel_skeleton_cuboid[0]))
        print("v_k_c[0]=",voxel_skeleton_cuboid[0])
        print("voxel_nets.shape=", voxel_nets.shape)
        print("> data_count", data_count)

    return samples, sample_cuboid_index, voxel_nets

############################## for prediction ###############################
# return the set of sliding window coordinates
def sliding_window_naif(x_min, x_max, y_min, y_max, grid_size):
    '''
    Args:
        *_min/*_max : a interger. The raw data range.
        grid_size : a interger/float. The side length of the grid.
    Returns:
        res : a list of 2-d coordinates (x,y). The set of grid coordinates.
    '''

    divide_x = np.floor((x_max - x_min)/grid_size)
    divide_y = np.floor((y_max - y_min)/grid_size)
    coor_x = np.arange(0,divide_x) * grid_size
    coor_y = np.arange(0,divide_y) * grid_size
    mesh_x, mesh_y = np.meshgrid(coor_x, coor_y)
    
    return np.stack((mesh_x,mesh_y), 2)

# predict version
def prepare_dataset_predict(data, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=False):
    '''
    Args:
        data: a numpy.ndarray (x,y,z,label). 
        coords_sw: (cuboid_x, cuboid_y, 2). The coordinates of sliding window.
        grid_size: The cuboid length and width.
        voxel_size: The voxel size.
        global_height: a float. The global height, our cuboid height.
        voxel_sample_mode: a string. mc or cmc.
        sample_size: how many points in a sample.
    Returns:
        samples: (nb_sample, 5000, 4 :x + y + z + label).
        sample_cuboid_index: (nb_sample, index of nb_cuboid).
        voxel_skeleton_cuboid: (nb_voxel, 4:x+y+z+[1 or 0]).
    '''

    # get voxel_skeleton_cuboid
    (coord_x,coord_y,_) = coords_sw.shape
    nb_cuboid = coord_x * coord_y
    cub_s_nb = int(grid_size/voxel_size)
    cub_h_nb = int(global_height/voxel_size)
    
    # returns
    samples = []
    sample_cuboid_index = {}
    voxel_skeleton_cuboid = {}

    w_nb = 0
    nb_sample = 0
    sw = []
    index_sw = 0
    for i in range(coord_x):
        for j in range(coord_y):
            
            # (1) cut data to cubes
            # local origin
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')

            # find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data, local_x, local_x+grid_size, local_y, local_y+grid_size)
            print(">> reigon ({},{}) - ({},{})".format(local_x,local_y, local_x+grid_size, local_y+grid_size))
            # shift points to local origin (0, 0, 0)
            # zero-centered
            '''
            #local_points = copy.deepcopy(data[local_index])
            local_points = data[local_index]
            local_z = np.min(local_points[:,2])
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_abs_height = np.max(local_points[:,2]) - local_z
            local_points[:,2] = local_points[:,2] - local_z
            adjust = np.mean(local_points[:, :3], axis=0) 
            local_points[:,:3] = local_points[:,:3] - adjust
            '''
            # shift points to local origin (0, 0, 0)
            # zero-centered
            #local_points = copy.deepcopy(data[local_index])
            local_points = data[local_index]
            local_z = np.min(local_points[:,2])
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_abs_height = np.max(local_points[:,2]) - local_z
            # local_abs_height
            local_points[:,2] = local_points[:,2] - local_z

            # voxelization
            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            voxel_skeleton_cuboid[w_nb] = voxel
            #visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

            # centralization in (x y z) thress axis by the center of voxels
            #local_points[:,:3] = local_points[:,:3] - np.mean(local_points[:, :3], axis=0)
            local_points[:,:3] = local_points[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
            local_points[:,:2] = local_points[:,:2]/grid_size
            local_points[:,2] = local_points[:,2]/global_height
            
            #sw.append((local_x, local_y, local_z, adjust[0], adjust[1], adjust[2]))
            sw.append((local_x, local_y, local_z, grid_size/2, grid_size/2, global_height/2))
            local_points[:,4] = index_sw
            index_sw = index_sw + 1

            if detail:
                print(">>> local abs height :", local_abs_height)
                print(">>> local data.shape :", local_points.shape)
                print(">>> local_data (points in cuboid) zero-centered but no standardization/normalization")
            

            # the number of local_points
            tmp_nb_sample = int(len(local_points)/sample_size)
            
            #print(">>> nb_sample={}".format(nb_sample))
            # set sample_cuboid_index
            #[sample_cuboid_index.append([i_s, w_nb]) for i_s in range(nb_sample, nb_sample+tmp_nb_sample)]
            
            for i_s in range(nb_sample, nb_sample+tmp_nb_sample):
                sample_cuboid_index[i_s] = w_nb
            
            nb_sample = nb_sample + tmp_nb_sample
            if detail:
                print(">>> tmp_nb_sample={}, nb_sample+tmp={}".format(tmp_nb_sample, nb_sample))

            # set samples
            np.random.shuffle(local_points)
            tmp_samples = [local_points[sample_size*i_t:sample_size*(i_t+1)] for i_t in range(tmp_nb_sample)]
            [samples.append(item) for item in tmp_samples]

            #print(">> nb_points_per_voxel.shape :",nb_points_per_voxel.shape)
            w_nb = w_nb + 1
            
    return np.array(samples), sample_cuboid_index, voxel_skeleton_cuboid, sw

def prepare_procedure_predict(path, grid_size, voxel_size, voxel_sample_mode, sample_size, label_name="llabel", global_height=50, detail=False, naif_sliding=False):
    '''
    Args:
        path : raw_data_path. The path of training/validation/test file.
        detail : a bool. If we want to show the details below.
    Returns:
        sw : now, the label dim (3) of samples is the index of sliding window information, will be changed later.
    '''
    # (1) preprocess data and get set of sliding window coordinates
    print("> input data:", path)
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data_with_intensity(path, label_name, detail=True)
    print("\n> data_preprocess.shape =", data_preprocessed.shape)
    
    # sliding window
    if naif_sliding:
        print(">> ok, we do naif sliding")
        coords_sw = sliding_window_naif(x_min, x_max, y_min, y_max, grid_size)
    else:
        coords_sw = sliding_window(x_min, x_max, y_min, y_max, grid_size)

    (d1,d2,_) = coords_sw.shape
    nb_cuboid = d1 * d2
    print("> coords.shape={}, size={}, sw={}".format(coords_sw.shape, coords_sw.size, coords_sw))
    
    samples, sample_cuboid_index, voxel_skeleton_cuboid, sw = prepare_dataset_predict(data_preprocessed, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=detail)
    print(">>> samples.shape={}, sample_cuboid_index.shape={}, voxel_skele.len={}".format(samples.shape, len(sample_cuboid_index), len(voxel_skeleton_cuboid)))
    
    voxel_nets = analyse_voxel_in_cuboid(voxel_skeleton_cuboid, int(global_height/voxel_size), int(grid_size/voxel_size))
    
    unique,count = np.unique(voxel_nets, return_counts=True)
    data_count = dict(zip(unique, count))
    
    if detail:
        print("> grid_size:", grid_size)
        print("> voxel_size:", voxel_size)
        print("> voxel sample mode is:", voxel_sample_mode)
        print("len(voxel_skeleton_cuboid) =", len(voxel_skeleton_cuboid), " ", type(voxel_skeleton_cuboid))
        print("v_k_c[0]=type",type(voxel_skeleton_cuboid[0]))
        print("v_k_c[0]=",voxel_skeleton_cuboid[0])
        print("voxel_nets.shape=", voxel_nets.shape)
        print("> data_count", data_count)

    return samples, sample_cuboid_index, voxel_nets, sw
    
############################## make a copy ###############################
# for prepare dataset
def prepare_dataset_copy(data, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=False):
    '''
    Args:
        data: a numpy.ndarray (x,y,z,label). 
        coords_sw: (cuboid_x, cuboid_y, 2). The coordinates of sliding window.
        grid_size: The cuboid length and width.
        voxel_size: The voxel size.
        global_height: a float. The global height, our cuboid height.
        voxel_sample_mode: a string. mc or cmc.
        sample_size: how many points in a sample.
    Returns:
        samples: (nb_sample, 5000, 4 :x + y + z + label).
        sample_cuboid_index: (nb_sample, index of nb_cuboid).
        voxel_skeleton_cuboid: (nb_voxel, 4:x+y+z+[1 or 0]).
    '''

    # get voxel_skeleton_cuboid
    (coord_x,coord_y,_) = coords_sw.shape
    nb_cuboid = coord_x * coord_y
    cub_s_nb = int(grid_size/voxel_size)
    cub_h_nb = int(global_height/voxel_size)
    
    # returns
    samples = []
    sample_cuboid_index = {}
    voxel_skeleton_cuboid = {}

    w_nb = 0
    nb_sample = 0
    
    for i in range(coord_x):
        for j in range(coord_y):   
            # (1) global coordinates of each cuboids
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')

            # (2) find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data, local_x, local_x+grid_size, local_y, local_y+grid_size)
            
            # (3) shift points to local origin (0, 0, 0) and zero-centered
            #local_points = copy.deepcopy(data[local_index])
            local_points = data[local_index]
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_abs_height = np.max(local_points[:,2]) - np.min(local_points[:,2])
            # local_abs_height
            local_points[:,2] = local_points[:,2] - np.min(local_points[:,2])

            if detail:
                print(">>> local abs height :", local_abs_height)
                print(">>> local data.shape :", local_points.shape)
                print(">>> local_data (points in cuboid) zero-centered and standardization/normalization")
                #print(">>> local_points=", local_points[:,:3][0:100])

            # (4) voxelization
            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            voxel_skeleton_cuboid[w_nb] = voxel
            #visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

            # (5) centralization in (x y z) thress axis by the center of voxels
            #local_points[:,:3] = local_points[:,:3] - np.mean(local_points[:, :3], axis=0)
            local_points[:,:3] = local_points[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
            local_points[:,:2] = local_points[:,:2]/grid_size
            local_points[:,2] = local_points[:,2]/global_height
            
            # the number of local_points
            tmp_nb_sample = int(len(local_points)/sample_size)
            
            #print(">>> nb_sample={}".format(nb_sample))
            # set sample_cuboid_index
            #[sample_cuboid_index.append([i_s, w_nb]) for i_s in range(nb_sample, nb_sample+tmp_nb_sample)]
            
            for i_s in range(nb_sample, nb_sample+tmp_nb_sample):
                sample_cuboid_index[i_s] = w_nb
            
            nb_sample = nb_sample + tmp_nb_sample
            if detail:
                print(">>> tmp_nb_sample={}, nb_sample+tmp={}".format(tmp_nb_sample, nb_sample))

            # set samples
            np.random.shuffle(local_points)
            tmp_samples = [local_points[sample_size*i_t:sample_size*(i_t+1)] for i_t in range(tmp_nb_sample)]
            [samples.append(item) for item in tmp_samples]

            data_augmentation()

            w_nb = w_nb + 1
            
    return np.array(samples), sample_cuboid_index, voxel_skeleton_cuboid