from utility import *

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

# write data
def write_data(data, filename, x_min_overlap, y_min_overlap):
    '''
    Args:
        data : a (n,3) np.ndarray. The points coordiantes!
        filename : a string.
    Returns:
        None.
    '''
    new_file = laspy.create(point_format=2, file_version="1.2")
    new_file.x = data[:,0] + x_min_overlap
    new_file.y = data[:,1] + y_min_overlap
    new_file.z = data[:,2]
    path = os.getcwd()+"/"+ filename + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".las"
    new_file.write(path)
    print(">> data writed to:", path)
    
    return None

# write data
def write_data_bis(data, filename, x_min_overlap, y_min_overlap):
    '''
    Args:
        data : a (n,3) np.ndarray. The points coordiantes!
        filename : a string.
    Returns:
        None.
    '''
    new_file = laspy.create(point_format=2, file_version="1.2")
    new_file.x = data[:,0] + x_min_overlap
    new_file.y = data[:,1] + y_min_overlap
    new_file.z = data[:,2]
    new_file.classification = data[:,3]
    path = os.getcwd()+"/"+ filename + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".las"
    new_file.write(path)
    print(">> data writed to:", path)
    
    return None

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
    print("layer_bot={}, layer_top={}, layer_bot_voxel={}, layer_top_voxel={}, voxel_size={}".format(layer_bot, layer_top, layer_bot_voxel, layer_top_voxel, voxel_size))
    
    #x_s, y_s = bottom.shape
    #print("bottom.shape", bottom[0:100])
    #print("bottom[0:100]", bottom[0:100])
    
    res2 = []
    i = 0
    tmp = bottom.copy()
    tmp[:,2] = tmp[:,2] + layer_bot_voxel
    for h in range(0,layer_top_voxel- layer_bot_voxel):
        for x in tmp:
            # flatten/reshape maybe work?
            if tuple(x) in voxel_grid:
                [res2.append(a) for a in voxel_grid[tuple(x)]]
        tmp[:,2] = tmp[:,2] + 1
    
    return np.array(res2)

def slice_voxel_data_and_find_coincidence(bottom, layer_bot, layer_top, voxel_size, voxel_grid_tls, voxel_grid_dls, show_coi_rate=True):
    '''
    Args:
        bottom : (n,3) np.ndarray. The index, output of the bottom_voxel.
        layer_bot : a float. The bottom of layer.
        layer_top : a float. The top of layer.
        voxel_size: a float. The voxel size.
        voxel_grid_tls : a dict. Key is (x,y,z) voxel index. Value is the points in this voxel for TLS data.
        voxel_grid_dls : a dict. Key is (x,y,z) voxel index. Value is the points in this voxel for DLS data.
        show_coi_rate : a boolean.
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
    check = lambda x,y: 0 if y==0 else x/y
    print("layer_bot={}, layer_top={}, layer_bot_voxel={}, layer_top_voxel={}, voxel_size={}".format(layer_bot, layer_top, layer_bot_voxel, layer_top_voxel, voxel_size))
    
    #res 
    coi_voxel, tls_data, dls_data = [], [], []
    nb_voxel_coi, nb_voxel_tls, nb_voxel_dls = 0, 0, 0

    tmp = bottom.copy()
    tmp[:,2] = tmp[:,2] + layer_bot_voxel
    for h in range(0, layer_top_voxel-layer_bot_voxel): 
        for x in tmp:
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
        tmp[:,2] = tmp[:,2] + 1
    
    if show_coi_rate:
        print("\n> nb_voxel_tls={}, nb_voxel_dls={}, nb_voxel_coi={}".format(nb_voxel_tls, nb_voxel_dls, nb_voxel_coi))
        print("> coi_rate={}".format(check(nb_voxel_coi, (nb_voxel_tls+nb_voxel_dls-nb_voxel_coi))))
        print("> nb_voxel_coi/nb_voxel_tls ={}".format(check(nb_voxel_coi,nb_voxel_tls)))
        print("> nb_voxel_coi/nb_voxel_dls ={}".format(check(nb_voxel_coi,nb_voxel_dls)))

    return np.array(tls_data), np.array(dls_data), nb_voxel_tls, nb_voxel_dls, nb_voxel_coi, np.array(coi_voxel)

# draw coi_rate
def plot_coi_rate(bottom, layer_bot, layer_top, voxel_size, tls_voxel_grid, dls_voxel_grid, slice_height, grid_size):
    '''
    Draw the plot.
    Args:
        bottom : (n,3) np.ndarray. The index, output of the bottom_voxel.
        layer_bot : a float. The bottom of layer.
        layer_top : a float. The top of layer.
        voxel_size: a float. The voxel size.
        voxel_grid_tls : a dict. Key is (x,y,z) voxel index. Value is the points in this voxel for TLS data.
        voxel_grid_dls : a dict. Key is (x,y,z) voxel index. Value is the points in this voxel for DLS data.
        slice_height : a float. The height of each slice.
    Return:
        None. 
    '''
    x_axis = []
    nb_coi, nb_tls, nb_dls = [], [], []
    coi_rate, coi_tls_rate, coi_dls_rate = [], [], []
    check = lambda x,y: 0 if y==0 else x/y

    for h in range(0, int(layer_top - layer_bot)):
        #print(">> h={}".format(h))
        print(">> layer_bot={}".format(layer_bot))
        layer_tls, layer_dls, nb_voxel_tls, nb_voxel_dls, nb_voxel_coi, coi_voxel = slice_voxel_data_and_find_coincidence(bottom, layer_bot, layer_bot+slice_height, voxel_size, tls_voxel_grid, dls_voxel_grid)
        nb_coi.append(nb_voxel_coi)
        nb_tls.append(nb_voxel_tls)
        nb_dls.append(nb_voxel_dls)
        coi_rate.append(check(nb_voxel_coi, (nb_voxel_tls+nb_voxel_dls-nb_voxel_coi)))
        coi_tls_rate.append(check(nb_voxel_coi,nb_voxel_tls))
        coi_dls_rate.append(check(nb_voxel_coi,nb_voxel_dls))
        x_axis.append(h)
        layer_bot = layer_bot + 1
    
    sns.set(style = "darkgrid")
    
    plt.title("Coi rate, slice height={}m, voxel_size={}m, grid_size={}m".format(slice_height, voxel_size, grid_size), fontsize=22)
    plt.plot(x_axis, coi_rate, color="red", label="coi_voxel/(tls_voxel+dls_voxel-coi_voxel)")
    plt.plot(x_axis, coi_tls_rate, color="green", label="coi_voxel/tls_voxel")
    plt.plot(x_axis, coi_dls_rate, color="blue", label="coi_voxel/dls_voxel")
    plt.legend(fontsize=14)
    plt.xlabel("different layer_bot height", fontsize=20)
    plt.ylabel("rate", fontsize=20)
    plt.show()

    plt.title("Nb voxel, slice height={}m, voxel_size={}m, grid_size={}m".format(slice_height, voxel_size, grid_size), fontsize=22)
    plt.plot(x_axis, nb_coi, color="red", label="nb_coi")
    plt.plot(x_axis, nb_tls, color="green", label="nb_voxel_tls")
    plt.plot(x_axis, nb_dls, color="blue", label="nb_voxel_dls")
    plt.legend(fontsize=18)
    plt.xlabel("different layer_bot height", fontsize=20)
    plt.ylabel("nb_voxel", fontsize=20)
    plt.show()

# get slice data
def get_slice_data(bottom, layer_bot, layer_top, voxel_size, tls_voxel_grid, dls_voxel_grid, slice_height, x_min_overlap, y_min_overlap):
    '''
    Args:
    Returns:
    '''
    print("\n>> save slice date")
    
    for h in range(1, int(np.floor((layer_top-layer_bot)/slice_height))):
        l_bot = layer_bot+(h-1)*slice_height
        l_top = layer_bot+h*slice_height
        print(">> layer_bot={}, layer_top={}".format(l_bot, l_top))
        layer_tls, layer_dls, nb_voxel_tls, nb_voxel_dls, nb_voxel_coi, coi_voxel = slice_voxel_data_and_find_coincidence(bottom, l_bot, l_top, voxel_size, tls_voxel_grid, dls_voxel_grid)
        print("> voxel_layer_tls.shape =", layer_tls.shape, "voxel_layer_dls.shape =", layer_dls.shape)
        #visualize_voxel_key_points(layer_tls, layer_tls, "voxel_layer_tls", only_points=True)
        #visualize_voxel_key_points(layer_dls, layer_dls, "voxel_layer_dls", only_points=True)
        #visualize_voxel_key_points(coi_voxel, coi_voxel, "both", only_points=True)
        visualize_voxel_key_points(layer_tls, layer_tls, "voxel_layer_tls", only_points=True)
        visualize_voxel_key_points(layer_dls, layer_dls, "voxel_layer_dls", only_points=True)
        
        write_data(layer_tls, "tls_layer_bot="+str(l_bot)+"_slice_h="+str(slice_height), x_min_overlap, y_min_overlap)
        write_data(layer_dls, "dls_layer_bot="+str(l_bot)+"_slice_h="+str(slice_height), x_min_overlap, y_min_overlap)
        write_data(coi_voxel*voxel_size, "coi_voxel_layer_bot="+str(l_bot)+"_slice_h="+str(slice_height), x_min_overlap, y_min_overlap)

    return None

# get slice data
def get_slice_data_bis(bottom, layer_bot, layer_top, voxel_size, tls_voxel_grid, dls_voxel_grid, slice_height, x_min_overlap, y_min_overlap):
    '''
    Args:
    Returns:
    '''
    nb_slice = np.ceil((layer_top - layer_bot)/slice_height)
    heights = np.linspace(layer_bot, layer_top, int(nb_slice), endpoint=False)
    print("\n>> save slice date, nb_slice={}".format(nb_slice))
    for h in heights:
        l_bot = h
        l_top = h + slice_height
        print(">> layer_bot={}, layer_top={}".format(l_bot, l_top))
        layer_tls, layer_dls, nb_voxel_tls, nb_voxel_dls, nb_voxel_coi, coi_voxel = slice_voxel_data_and_find_coincidence(bottom, l_bot, l_top, voxel_size, tls_voxel_grid, dls_voxel_grid)
        print("> voxel_layer_tls.shape =", layer_tls.shape, "voxel_layer_dls.shape =", layer_dls.shape)
        #visualize_voxel_key_points(layer_tls, layer_tls, "voxel_layer_tls", only_points=True)
        #visualize_voxel_key_points(layer_dls, layer_dls, "voxel_layer_dls", only_points=True)
        #visualize_voxel_key_points(coi_voxel, coi_voxel, "both", only_points=True)

        write_data(layer_tls, "tls_layer_bot="+str(l_bot)+"_slice_h="+str(slice_height), x_min_overlap, y_min_overlap)
        write_data(layer_dls, "dls_layer_bot="+str(l_bot)+"_slice_h="+str(slice_height), x_min_overlap, y_min_overlap)
        write_data(coi_voxel*voxel_size, "coi_voxel_layer_bot="+str(l_bot)+"_slice_h="+str(slice_height), x_min_overlap, y_min_overlap)

    return None
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


