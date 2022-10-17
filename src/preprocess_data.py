import copy
from scipy.spatial.transform import Rotation
from utility import *
from datetime import datetime

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

# normalize feature
def normalize_feature(f):
    f_min = np.min(f)
    f_max = np.max(f)
    print("f_min={}, f_max={}".format(f_min, f_max))
    return (f-f_min)/(f_max-f_min)

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
    
    print(">> data_las.z min={} max={} diff={}".format(z_min, z_max, z_max - z_min))

    # intensity
    #f2_max = np.log(np.max(data_las[feature2]))
    #f2_min = np.log(np.min(data_las[feature2]))
    #f_intensity = ((np.log(data_las[feature2])-f2_min)/(f2_max-f2_min))
    
    #(data_target['intensity']/65535)*35 - 30 for TLS
    f_intensity = (data_las[feature2]/65535)*40 - 40
    
    #print(">> f_intensity.shape={}, nan size={}, non nan={}".format(f_intensity.shape, f_intensity[np.isnan(f_intensity)].shape, f_intensity[~np.isnan(f_intensity)].shape))

    f_roughness = data_las["Roughness (0.7)"]
    f_roughness[np.isnan(f_roughness)] = -0.1
    f_roughness = f_roughness + 0.1
    
    f_ncr = data_las["Normal change rate (0.7)"]
    f_ncr[np.isnan(f_ncr)] = -0.1
    f_ncr = f_ncr + 0.1

    max_nb_of_returns = 5
    # order
    f_return_nb = data_las["return_number"]
    f_return_nb[np.isnan(f_return_nb)] = 1
    f_return_nb = f_return_nb/max_nb_of_returns
    
    # total number
    f_nb_of_returns = data_las["number_of_returns"]
    f_nb_of_returns[np.isnan(f_nb_of_returns)] = 1
    f_nb_of_returns = f_nb_of_returns/max_nb_of_returns
    
    f_rest_return = (f_nb_of_returns - f_return_nb)/max_nb_of_returns
    f_ratio_return = f_return_nb/(f_nb_of_returns*max_nb_of_returns)
    f_ratio_return[np.isnan(f_ratio_return)] = 0
    
    '''
    print("nan shape = {} {} {} {}".format(
        f_return_nb[np.isnan(f_return_nb)].shape, 
        f_nb_of_returns[np.isnan(f_nb_of_returns)].shape,
        f_rest_return[np.isnan(f_rest_return)].shape,
        f_ratio_return[np.isnan(f_ratio_return)].shape
        ))
    exit()
    '''
    data = np.vstack((
        data_las.x - x_min, 
        data_las.y - y_min, 
        data_las.z - z_min, 
        data_las[feature],
        normalize_feature(f_intensity),
        normalize_feature(f_roughness), 
        normalize_feature(f_ncr)
        #f_return_nb,
        #f_nb_of_returns,
        #f_rest_return,
        #f_ratio_return
        ))

    print(">>>[!data with intensity] data shape =", data.shape, " type =", type(data))

    return data.transpose(), x_min, x_max, y_min, y_max, z_min, z_max

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

# return the set of sliding window coordinates
def sliding_window(x_min, x_max, y_min, y_max, grid_size, nb_window=5):
    '''
    Args:
        *_min/*_max : a interger. The data range.
        grid_size : a interger/float. The side length of the grid.
    Returns:
        res : a list of 2-d coordinates (x,y). The set of grid coordinates.
    '''
    coord_x = np.random.uniform(x_min, x_max-grid_size, nb_window)
    coord_y = np.random.uniform(y_min, y_max-grid_size, nb_window)
    mesh_x, mesh_y = np.meshgrid(coord_x, coord_y)

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
    #points = cuboid[:,:3]
    
    # non_empy_voxel : no empty voxel :)
    # index : the positions of [new elements in old array]
    # index_inversed : the positions of [old elements in new array]
    # nb_pts_per_voxel : nb of points in each voxels
    no_empty_voxel, index, index_inversed, nb_points_per_voxel = np.unique((cuboid[:,:3]//voxel_size).astype(int), axis=0, return_index=True, return_inverse=True, return_counts=True)
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

    intensity_std = []
    # i - index, v - coordinate of non empty voxel
    for i,v in enumerate(no_empty_voxel):
        nb_points = nb_points_per_voxel[i]
        voxel_grid[tuple(v)] = cuboid[index_points_on_voxel_sorted[loc_select:loc_select+nb_points]]
        #intensity_std.append(np.mean(cuboid[index_points_on_voxel_sorted[loc_select:loc_select+nb_points]][:,3]))
        #res.append(key_point_in_voxel(v))
        loc_select = loc_select + nb_points
    
    nb_p_max = np.max(nb_points_per_voxel)
    nb_p_min = np.min(nb_points_per_voxel)
    voxel_and_points = np.concatenate((no_empty_voxel, np.array([(nb_points_per_voxel - nb_p_min)/(nb_p_max - nb_p_min)]).T), axis=1)
    #voxel_and_points = np.append(no_empty_voxel, np.array(intensity_std).reshape(-1, 1), axis=1)

    #return np.array(res), np.array(nb_points_per_voxel), voxel_and_points
    return voxel_grid, np.array(nb_points_per_voxel), voxel_and_points

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

# for prepare dataset
def prepare_dataset(data, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=False, data_augmentation=False):
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
    (nb_cuboid,_) = coords_sw.shape
    cub_s_nb = int(grid_size/voxel_size)
    cub_h_nb = int(global_height/voxel_size)
    
    # returns
    samples = []
    sample_cuboid_index = {}
    voxel_skeleton_cuboid = {}

    w_nb = 0
    nb_sample = 0
    count_voxel_skeleton = 0
    
    for coord in coords_sw:
        
        # (1) global coordinates of each cuboids
        local_x, local_y = coord
        print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')

        # (2) find index of the data_preprocessed in this sliding window
        local_index = get_region_index(data, local_x, local_x+grid_size, local_y, local_y+grid_size)
        #print(">> there are {} points in this cuboid".format(len(local_index[0])))
        
        if len(local_index[0]) < 100:
            print(">> point number not enough, cuboid-{} skiped".format(w_nb))
            #voxel_skeleton_cuboid[w_nb] = []
            w_nb = w_nb + 1
            continue

        # (3) shift points to local origin (0, 0, 0) and zero-centered
        local_points = data[local_index]
        local_points[:,0] = local_points[:,0] - local_x
        local_points[:,1] = local_points[:,1] - local_y
        local_z_min = np.min(local_points[:,2])
        local_abs_height = np.max(local_points[:,2]) - local_z_min
        # local_abs_height
        local_points[:,2] = local_points[:,2] - local_z_min
        print("local points size={} shape={} type={}".format(local_points.size, local_points.shape, type(local_points)))
        
        if len(local_index[0]) < sample_size:
            print(">> local points shape={}".format(local_points.shape))
            local_points = np.repeat(local_points, (sample_size//len(local_index[0]))+1, axis=0)
            print(">> [duplicate] local points shape={}".format(local_points.shape))

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

        if data_augmentation:            
            # data augmentation
            rotation = Rotation.from_euler('z', [90, 180, 270], degrees=True)
            for angle in range(3):
                #rotation[angle]
                local_points_tmp = local_points.copy()
                local_points_tmp[:,:2] = local_points_tmp[:,:2] - np.array([grid_size/2, grid_size/2])
                local_points_tmp[:,:3] = rotation[angle].apply(local_points_tmp[:,:3])
                local_points_tmp[:,:2] = local_points_tmp[:,:2] + np.array([grid_size/2, grid_size/2])

                '''
                # $$$$$$$$$$$$$$see what we used for training$$$$$$$$$$$$$$$
                new_file = laspy.create(point_format=3)
                new_file.x = local_points_tmp[:,0]
                new_file.y = local_points_tmp[:,1]
                new_file.z = local_points_tmp[:,2]
                new_file.write(os.getcwd()+"/test_"+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + str(angle)+".las")
                '''

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
                    print(">>> [data augmentation - sample rotated:{}°]".format((angle+1)*90))
                    print(">>> tmp_nb_sample={}, nb_sample+tmp={}".format(tmp_nb_sample, nb_sample))
                
                # set samples
                np.random.shuffle(local_points_tmp)
                tmp_samples = [local_points_tmp[sample_size*i_t:sample_size*(i_t+1)] for i_t in range(tmp_nb_sample)]
                [samples.append(item) for item in tmp_samples]
            
        # sliding_window added 1
        w_nb = w_nb + 1
        
    return np.array(samples), sample_cuboid_index, voxel_skeleton_cuboid

def prepare_procedure(path, grid_size, voxel_size, voxel_sample_mode, sample_size, global_height=50, label_name="llabel", detail=False, naif_sliding=False, nb_window=10):
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
        t1 = sliding_window_naif(0, x_max - x_min, 0, y_max - y_min, grid_size)
        d1,d2,d3 = t1.shape
        coords_sw = t1.reshape((d1*d2, 2))
    else:
        t1 = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size, nb_window)
        d1,d2,d3 = t1.shape
        t1 = t1.reshape((d1*d2, 2))
        t2 = sliding_window_old(0, x_max - x_min, 0, y_max - y_min, grid_size)
        d1,d2,d3 = t2.shape
        t2 = t2.reshape((d1*d2, 2))
        #print("t1.shape={} t1={}".format(t1.shape, t1))
        #print("t2.shape={} t2={}".format(t2.shape, t2))
        coords_sw = np.concatenate((t1,t2), axis=0)

    d1,d2 = coords_sw.shape
    nb_cuboid = d1
    print("> sliding window : coords.shape={}".format(coords_sw.shape))
    #exit()

    #global_height = z_max - z_min
    #global_height = 50
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
        #print("> data_count", data_count)

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
            print(">> there are {} points in this cuboid".format(len(local_index[0])))
            
            if len(local_index[0]) < 2:
                print(">> point number not enough, cuboid-{} skiped".format(w_nb))
                voxel_skeleton_cuboid[w_nb] = []
                #index_sw = index_sw + 1
                w_nb = w_nb + 1
                continue
            
            '''
            # shift points to local origin (0, 0, 0)
            # zero-centered
            
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

            if len(local_index[0]) < sample_size:
                print(">> local points shape={}".format(local_points.shape))
                local_points = np.repeat(local_points, (sample_size//len(local_index[0]))+1, axis=0)
                print(">> [duplicate] local points shape={}".format(local_points.shape))

            # voxelization
            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            voxel_skeleton_cuboid[w_nb] = voxel
            #visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

            # centralization in (x y z) thress axis by the center of voxels
            adjust = np.mean(local_points[:, :3], axis=0) 
            #local_points[:,:3] = local_points[:,:3] - adjust
            local_points[:,:3] = local_points[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
            local_points[:,:2] = local_points[:,:2]/grid_size
            local_points[:,2] = local_points[:,2]/global_height
            
            #sw.append((local_x, local_y, local_z, adjust[0], adjust[1], adjust[2]))
            sw.append((local_x, local_y, local_z, grid_size/2, grid_size/2, global_height/2))
            local_points[:,3] = index_sw
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

# return the set of sliding window coordinates
def sliding_window_old(x_min, x_max, y_min, y_max, grid_size):
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

# laspy read and write incorrectly
# this explain why the scale is different
# point format error, change [point_format=las.point_format, file_version="1.2"] to [point_format=3], all is ok
def error_not_urgent():
    las = read_header(data_path)
    get_info(las)
    data,_,_,_,_,_,_ = read_data(data_path, "llabel")
    local_index = get_region_index(data, 286624.0, 286699, 583755, 583799)
    # see here [!!!] new_file = laspy.create(point_format=las.point_format, file_version="1.2")
    new_file = laspy.create(point_format=3)
    new_file.x = data[local_index][:,0]
    new_file.y = data[local_index][:,1]
    new_file.z = data[local_index][:,2]
    las.points = new_file.points
    #las['llabel'] = predict.cpu().detach().numpy()
    las.write(os.getcwd()+"/predict_res/res_{:04}.las".format(110))
    exit()

def voxel_grid_sample_copy(cuboid, voxel_size, mode):
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
    #points = cuboid[:,:3]
    
    # non_empy_voxel : no empty voxel :)
    # index : the positions of [new elements in old array]
    # index_inversed : the positions of [old elements in new array]
    # nb_pts_per_voxel : nb of points in each voxels
    no_empty_voxel, index, index_inversed, nb_points_per_voxel = np.unique((cuboid[:,:3]//voxel_size).astype(int), axis=0, return_index=True, return_inverse=True, return_counts=True)
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
        voxel_grid[tuple(v)] = cuboid[index_points_on_voxel_sorted[loc_select:loc_select+nb_points]]
        res.append(key_point_in_voxel(v))
        loc_select = loc_select + nb_points
    
    nb_p_max = np.max(nb_points_per_voxel)
    nb_p_min = np.min(nb_points_per_voxel)
    voxel_and_points = np.concatenate((no_empty_voxel, np.array([(nb_points_per_voxel - nb_p_min)/(nb_p_max - nb_p_min)]).T), axis=1)

    #return np.array(res), np.array(nb_points_per_voxel), voxel_and_points
    return voxel_grid, np.array(nb_points_per_voxel), voxel_and_points
