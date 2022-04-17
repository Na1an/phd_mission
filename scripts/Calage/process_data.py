from utility import *
from torch.utils.data import Dataset, DataLoader

# This function works for the preprocessing the (TLS) data
def read_data(path, feature=None, detail=False):
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
    #data = np.vstack((data_las.x - x_min, data_las.y - y_min, data_las.z, data_las[feature])).transpose()
    data = np.vstack((data_las.x, data_las.y, data_las.z)).transpose()
    #print(">>> data token :", data[0:3], " shape =", data.shape, " type =", type(data))
    #print(">>> data border: x_min={}, y_min={}, x_max={}, y_max={}".format(x_min, y_min, x_max, y_max))
    return data, x_min, x_max, y_min, y_max, z_min, z_max

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
    print(">>> inside get region indiece data_pricessed[0:10] =", data[0:10])
    return np.where((((x_min)<data[:,0]) & (data[:,0]<(x_max))) & (((y_min)<data[:,1]) & (data[:,1]<(y_max))))

# to do : decide which kind of key point in each voxel we need. 
# [mean center, closest point to mean center, voxel center]
# voxelization
#def voxel_grid_sample(cuboid, grid_size, height, voxel_size, mode):
def voxel_grid_sample(cuboid, voxel_size, mode, get_voxel_grid=False):
    '''
    Args:
        points : a (n,4) numpy.darray. The data to process.
        voxel_size : a float. The resolution of the voxel. 
        mode : a string. How to select points in voxel. ('mc': mean_center, 'cmc' : closest point to mean center)
        get_voxel_grid : a boolean. If true, return plus the voxel grid.
        #grid_size : a interger/float. The side length of a grid.
        #height : a float. The max height of the raw data. Not local height!
    Returns:
        res : a voxelized data. key points in each voxel.
        nb_points_per_voxel : a list integer. The total point number in each voxel.
        non_empty_voxel : a (n,3) np.darray. The index of occupied voxel.
    '''

    res = []
    points = cuboid[:,:3]
    #nb_voxel = int((((grid_size+0.000001)//voxel_size)**2) * ((height+0.000001)//voxel_size))
    #print(">>> grid_size//voxel_size =", grid_size//voxel_size)
    #print(">>> height//voxel_size =", height//voxel_size)
    #print(">>> voxel_nb =", nb_voxel)
    
    # non_empy_voxel : no empty voxel :)
    # index : the positions of [new elements in old array]
    # index_inversed : the positions of [old elements in new array]
    # nb_pts_per_voxel : nb of points in each voxels
    non_empty_voxel, index, index_inversed, nb_points_per_voxel = np.unique((points//voxel_size).astype(int), axis=0, return_index=True, return_inverse=True, return_counts=True)
    index_points_on_voxel_sorted = np.argsort(index_inversed)
    # we can then access the points that are linked to each voxel through index_points_on_voxel_sorted and how many they are (nb_pts_per_voxel)

    # voxel grid save the points' index for each voxel
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
    
    if get_voxel_grid:
        return np.array(res), np.array(nb_points_per_voxel), non_empty_voxel, voxel_grid

    return np.array(res), np.array(nb_points_per_voxel), non_empty_voxel

def bottom_voxel(data):
    '''
    Args:
        data : a (n,3) np.ndarray. The voxelized data.
    Returns:
        index : a list of integer. The index of bottom voxel.
    '''
    _,index = np.unique(data[:,0:2], axis=0, return_index=True)
    
    return index

def slice_voxel_data(bottom, layer_bot, layer_top, voxel_size, voxel_grid):
    '''
    Args:
        bottom : (n,3) np.ndarray. The index, output of the bottom_voxel.
        layer_bot : a float. The bottom of layer.
        layer_top : a float. The top of layer.
    Return:
        res : (layer_height, n, 3). The index of a voxel layer.
    '''

    layer_bot_voxel = int((layer_bot+0.000001)//voxel_size)
    layer_top_voxel = int(((layer_top+0.000001)//voxel_size)) + 1
    x_s, y_s = bottom.shape
    
    res2 = []
    i = 0
    for h in range(layer_bot_voxel, layer_top_voxel):
        #res[i] = bottom[:,2] + h
        #i = i+1
        #tmp = bottom[:,2] + h
        
        for x in bottom:
            x[2] = x[2] + h
            # flatten/reshape maybe work?
            if tuple(x) in voxel_grid:
                [res2.append(a) for a in voxel_grid[tuple(x)]]
    
    return np.array(res2)

def slice_voxel_data_and_find_coincidence(bottom, layer_bot, layer_top, voxel_size, voxel_grid_tls, voxel_grid_dls):
    '''
    Args:
        bottom : (n,3) np.ndarray. The index, output of the bottom_voxel.
        layer_bot : a float. The bottom of layer.
        layer_top : a float. The top of layer.
    Return:
        tls_data (points) : (layer_height, n, 3). The points data of tls.
        dls_data (points) : (layer_height, n, 3). The points data of dls.
        nb_voxel_tls : a integer. The number of TLS voxel exists in the layer.
        nb_voxel_dls : a integer. The number of DLS voxel exists in the layer.
        nb_voxel_coi : a integer. The number of voxel exists in the layer for both TLS and DLS.
        coi_voxel : a (n,3) np.darray. The index of the coincidence voxel.
    '''

    layer_bot_voxel = int((layer_bot+0.000001)//voxel_size)
    layer_top_voxel = int(((layer_top+0.000001)//voxel_size)) + 1
    x_s, y_s = bottom.shape
    
    #res 
    tls_data = []
    dls_data = []
    nb_voxel_tls = 0
    nb_voxel_dls = 0
    nb_voxel_coi = 0
    coi_voxel = []

    for h in range(layer_bot_voxel, layer_top_voxel):
        for x in bottom:
            x[2] = x[2] + h
            # flatten/reshape maybe work?
            if tuple(x) in voxel_grid_tls:
                nb_voxel_tls = nb_voxel_tls + 1
                [tls_data.append(a) for a in voxel_grid_tls[tuple(x)]]
            if tuple(x) in voxel_grid_dls:
                nb_voxel_dls = nb_voxel_dls + 1
                [dls_data.append(a) for a in voxel_grid_dls[tuple(x)]]
            if (tuple(x) in voxel_grid_dls) and (tuple(x) in voxel_grid_tls):
                nb_voxel_coi = nb_voxel_coi + 1
                coi_voxel.append(tuple(x))

    return np.array(tls_data), np.array(dls_data), nb_voxel_tls, nb_voxel_dls, nb_voxel_coi, np.array(coi_voxel)
    
############################## abandoned ###############################
# return the set of sliding window coordinates
def sliding_window_naif(x_min, x_max, y_min, y_max, grid_size):
    '''
    Args:
        *_min/*_max : a interger. The raw data range.
        grid_size : a interger/float. The side length of the grid.
    Returns:
        res : a list of 2-d coordinates (x,y). The set of grid coordinates.
    '''

    divide_x = np.ceil((x_max - x_min)/grid_size)
    divide_y = np.ceil((y_max - y_min)/grid_size)
    coor_x = np.linspace(x_min, x_max, int(divide_x), endpoint=False)
    coor_y = np.linspace(y_min, y_max, int(divide_y), endpoint=False)
    mesh_x, mesh_y = np.meshgrid(coor_x, coor_y)
    
    return np.stack((mesh_x,mesh_y), 2)


