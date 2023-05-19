import copy
from scipy.spatial.transform import Rotation
from utility import *
from datetime import datetime, timedelta
from jakteristics import compute_features

# This function works for the preprocessing the data
def read_data(path, label_name, detail=False):
    '''
    Args:
        path : a string. The path of the data file.
        label_name : a string. Label_name.
        detail : a bool. False by default.
    Returns:
        res : a 4-D numpy array type tensor.
    '''
    data_las = laspy.read(path)
    x_min, x_max, y_min, y_max, z_min, z_max = get_info(data_las)
    data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, data_las[label_name])).transpose()
    print(">>> data shape =", data.shape, " type =", type(data))

    return data, x_min, x_max, y_min, y_max, z_min, z_max

# read tran and val dataset from directory
def read_data_from_directory(path_files, resolution, voxel_sample_mode, label_name, sample_size, augmentation):
    res = []
    files = os.listdir(path_files)
    for f in files:
        path_tmp = path_files + '/' + f
        print(">> preprocessing data:", path_tmp)
        samples_tmp, _, _ = prepare_procedure_ier(
                                                path_tmp, 
                                                resolution=resolution,
                                                voxel_sample_mode=voxel_sample_mode,
                                                label_name=label_name, 
                                                sample_size=sample_size,
                                                augmentation=augmentation)
        res.append(samples_tmp)
    
    res = np.concatenate(res)
    print("\n>> samples.shape = {} \n".format(res.shape))
    return res

# This function works for the preprocessing the data with intensity
def read_data_with_intensity(path, label_name, feature='intensity', detail=False):
    '''
    Args:
        path : a string. The path of the data file.
        feature : a string. Which feature we want to keep in the output.
        detail : a bool. False by default.
    Returns:
        res : a 4-D numpy array type tensor.
    '''
    data_las = laspy.read(path)
    data_las = data_las[data_las[label_name]>0]
    x_min, x_max, y_min, y_max, z_min, z_max = get_info(data_las)
    print(">> bincount label_name={} : {}".format(label_name, np.bincount(data_las[label_name].astype(int))))
    print(">> data_las.z min={} max={} diff={}".format(z_min, z_max, z_max - z_min))

    data = np.vstack((
        data_las.x - x_min, 
        data_las.y - y_min, 
        data_las.z - z_min,
        data_las[label_name],
        data_las[label_name] # here is placeholder here
        ))

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

# [mean center, closest point to mean center, voxel center]
# voxelization
#def voxel_grid_sample(cuboid, grid_size, height, voxel_size, mode):
def voxel_grid_sample(cuboid, voxel_size, mode):
    '''
    Args:
        cuboid : a (n,4) numpy.darray. The data to process.
        voxel_size : a float. The resolution of the voxel. 
        mode : a string. How to select points in voxel. ('mc': mean_center, 'cmc' : closest point to mean center)
    Returns:
        voxel_grid : a dict. key is (x,y,z) coordinate of voxel, value is a list of points in the voxel
        nb_points_per_voxel : a list integer. The total voxel number.
        voxel_and_points : 
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
        loc_select = loc_select + nb_points

    nb_p_max = np.max(nb_points_per_voxel)
    nb_p_min = np.min(nb_points_per_voxel)
    
    voxel_and_points = np.concatenate((no_empty_voxel, np.array([(nb_points_per_voxel/nb_p_max)]).T), axis=1)
    return voxel_grid, np.array(nb_points_per_voxel), voxel_and_points

def analyse_voxel_in_cuboid_bak(voxel_skeleton_cuboid, h, side):
    '''
    Args:
        voxel_skeleton_cuboid : a dictionnary.
        h : a float. The height cuboid.
        side : a float. side length.
    Returns:
        res: a voxelized space, indicate each voxel is occupied or not.
    '''
    print(">> voxel_net, h={} side={}".format(h,side))
    nb_cuboid = len(voxel_skeleton_cuboid)
    print(">> voxel_skeleton_cuboid[0].shape=", voxel_skeleton_cuboid[0].shape)
    res = np.zeros([nb_cuboid, side, side, h])
    for k,v in voxel_skeleton_cuboid.items():
        #print("k=",k, "v.shape=", v.shape)
        for c in v:
            v_x,v_y,v_z, v_p = c
            res[k,int(v_x),int(v_y),int(v_z)] = v_p

    return res

##########################
# training - ier version #
##########################
def prepare_dataset_ier(data, voxel_size_ier, voxel_sample_mode, augmentation, limit_comp, limit_p_in_comp, tls_mode, resolution=25, for_test=False):
    '''
    Args:
        data: a np.ndarray. (x,y,z,label, placeholder)
    Returns:
        ..
    '''
    # tls_mode is a temporary parameter
    show_sample = False
    sample_position = []
    # (1) calculate gd and ier. group trees is also splited in the same time.

    features_30 = compute_features(data[:,:3], search_radius=0.3, feature_names=["PCA1","linearity","sphericity", "verticality"])
    features_60 = compute_features(data[:,:3], search_radius=0.6, feature_names=["PCA1","linearity","sphericity", "verticality"])
    features_90 = compute_features(data[:,:3], search_radius=0.9, feature_names=["PCA1","linearity","sphericity", "verticality"])
    data = np.concatenate((data, features_30, features_60, features_90), axis=1)
    
    nb_p, len_f = data.shape
    print(">> before remove nan, data.shape={}".format(data.shape))
    data = data[np.all(~np.isnan(data[:,-12:]), axis=1)]
    print(">> after remove nan, data.shape={}, {}% point removed".format(data.shape, 100 - 100*(data.shape[0]/nb_p)))
    data[:,-12:] = standardization(data[:,-12:])
    print(">> norlization - down")

    dict_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(data, voxel_size_ier, voxel_sample_mode)
    # dict_points_in_voxel is a dict, key is voxel coord, value is a list of (points, label, ..) 
    initialize_voxels(dict_points_in_voxel)
    _, max_comp_id = geodesic_distance(dict_points_in_voxel, voxel_size_ier, tree_radius=7, limit_comp=limit_comp, limit_p_in_comp=limit_p_in_comp, tls_mode=tls_mode)
    # dict_points_in_voxel: k is coord of voxel, value is a list of points
    # dict_points_in_voxel[k=(0,0,0)] = [point1, ...], point1 = [x,y,z,label,_,gd,ier,nb_comp is id of comp]

    # dict_voxels to samples
    sample_tmp = [[] for i in range(max_comp_id)]

    if augmentation:
        # -90, -45, 45, 90
        sample_tmp_aug = [[[] for i in range(max_comp_id)] for j in range(4)]
        print(">>> Augmentation is true, sample_tmp_aug created. (rotation: -90, -45, 45, 90)")

    for _,v in dict_points_in_voxel.items():
        points, _ = v
        id_comp = int(points[0][-1])
        [sample_tmp[id_comp].append(ps) for ps in points[:,:-1]]

    ic_empty = []
    sample_res = [[] for i in range(max_comp_id)]
    for ic in range(len(sample_tmp)):
        # sample_tmp[ic] : [[x,y,z,label,_, 12*features, (17) gd, (18) ier], ...]
        try:
            sample_tmp[ic] = np.vstack(sample_tmp[ic])
        except ValueError:
            sample_tmp[ic] = []
            sample_position.append([(0, 0, 0, 0, 0, 0, 0)])
            ic_empty.append(ic)
            print(">> Value Error, sample_tmp[ic].shape = {}, ic={}".format(len(sample_tmp[ic]), ic))
            continue

        #sample_tmp[ic][np.isnan(sample_tmp[ic])] = 1
        if tls_mode:
            np.random.shuffle(sample_tmp[ic])
        else:
            sample_tmp[ic] = sample_tmp[ic][sample_tmp[ic][:, -2].argsort()]
        sample_tmp[ic][:,-1] = ic
        # normalize ier
        x_min, y_min, z_min = np.min(sample_tmp[ic][:,0]), np.min(sample_tmp[ic][:,1]), np.min(sample_tmp[ic][:,2])
        sample_tmp[ic][:,0] = sample_tmp[ic][:,0] - x_min
        sample_tmp[ic][:,1] = sample_tmp[ic][:,1] - y_min
        sample_tmp[ic][:,2] = sample_tmp[ic][:,2] - z_min

        pos_raw = np.copy(sample_tmp[ic][:,:3])

        try:
            sample_tmp[ic][:,:3], max_axe, max_x_axe, max_y_axe, max_z_axe = normalize_long_axe(sample_tmp[ic][:,:3])
        except ValueError as e:
            print(">>>> [ERROR] we have a error:", e)
            print(">>>> [ERROR] this sample will be skipped.")
            sample_skipped = True
            continue

        new_voxel_size = 1/resolution
        sample_position.append([(x_min, y_min, z_min, max_axe, max_x_axe, max_y_axe, max_z_axe)])
        
        #sample_tmp_bis[:,:3] = sample_tmp_bis[:,:3] - 0.5
        sample_res[ic] = np.concatenate((sample_tmp[ic], pos_raw), axis=1)

        #sample_res_rest[ic] = sample_tmp_bis_rest
        if show_sample:
            plot_pc(sample_tmp_bis[:,:3])
            plot_pc(voxel)

        #print("voxel.shape={}".format(voxel.shape))
        if augmentation:
            # data augmentation
            rotation = Rotation.from_euler('z', [-90, -45, 45, 90], degrees=True)
            for angle in range(4):
                s_tmp = copy.deepcopy(sample_tmp_bis)
                s_tmp[:,:3] = rotation[angle].apply(s_tmp[:,:3])
                s_tmp[:,:3] = s_tmp[:,:3] + 0.5
                _, _, voxel = voxel_grid_sample(s_tmp, (new_voxel_size), voxel_sample_mode)
                s_tmp[:,:3] = s_tmp[:,:3] - 0.5
                
                # plot
                if show_sample:
                    plot_pc(s_tmp)
                    plot_pc(voxel)
                sample_tmp_aug[angle][ic] = s_tmp
        
    if augmentation:
        sample_tmp = sample_tmp + sample_tmp_aug[0] + sample_tmp_aug[1] + sample_tmp_aug[2] + sample_tmp_aug[3]

    for ic_del in ic_empty:
        del sample_res[ic_del]
        del sample_position[ic_del]
    samples = np.array(sample_res, dtype='object')
    print(">> prepare_dataset_ier finesehd samples.shape={} len(sample_position)={}".format(samples.shape, len(sample_position)))
    print(">> ic_empy=", ic_empty)
    return samples, 0, sample_position

def prepare_procedure_ier(path, resolution, voxel_sample_mode, label_name, augmentation, sample_size=3000, for_test=False, voxel_size_ier=0.6, limit_comp=10, limit_p_in_comp=100, tls_mode=False):
    '''
    Args:
    Returns:
    '''
    print("augmen is",augmentation)
    # (1) load data
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data_with_intensity(path, label_name=label_name, detail=True)
    print("> input data: {} \n> data_preprocess.shape = {}".format(path, data_preprocessed.shape))

    # (2) build samples
    # data_preprocessed : (x,y,z,label,intensity)
    samples, _, sample_position = prepare_dataset_ier(
                                                    data_preprocessed, 
                                                    voxel_size_ier, 
                                                    voxel_sample_mode, 
                                                    resolution=resolution, 
                                                    augmentation=augmentation, 
                                                    for_test=for_test, 
                                                    limit_comp=limit_comp, 
                                                    limit_p_in_comp=limit_p_in_comp,
                                                    tls_mode=tls_mode)

    samples_res = []
    sample_voxel_net_index = []
    len_samples = len(samples)
    for i in range(len_samples):
        new_sample_tmp = split_reminder(samples[i], sample_size, axis=0)
        # p_size, the last one 
        p_size, f_size = new_sample_tmp[-1].shape

        if len(new_sample_tmp) == 1:
            new_sample_tmp[-1] = np.concatenate((new_sample_tmp[-1], samples[i][np.random.choice([c for c in range(p_size)], sample_size - p_size)]))
        else:
            new_sample_tmp[-1] = np.concatenate((new_sample_tmp[-1], samples[i][np.random.choice([c for c in range(sample_size*(len(new_sample_tmp)-1))], sample_size - p_size)]))
        #print("new_sample_tmp[-1].shape=",new_sample_tmp[-1].shape)
        samples_res = samples_res + new_sample_tmp
        
        for c in range(len(new_sample_tmp)):
            sample_voxel_net_index.append(i)
        
        print(">>> processing samples {}/{} - ok \t".format(i+1, len_samples) , end="\r")

    samples_res = np.array(samples_res)
    
    if for_test:
        return samples_res, sample_voxel_net_index, 0, sample_position, x_min, y_min, z_min
    else:
        return samples_res, sample_voxel_net_index, 0

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



