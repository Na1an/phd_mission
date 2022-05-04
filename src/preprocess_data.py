from utility import *

# This function works for the preprocessing the (TLS) data
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
    print(">>> data token :", data[0:10], " shape =", data.shape, " type =", type(data))

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

# for prepare dataset
def prepare_dataset(data, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size):
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
    sample_cuboid_index = []
    voxel_skeleton_cuboid = {}

    w_nb = 0
    nb_sample = 0
    
    for i in range(coord_x):
        for j in range(coord_y):
            
            # (1) cut data to cubes
            # local origin
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')

            # find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data, local_x, local_x+grid_size, local_y, local_y+grid_size)
           
            # shift points to local origin (0, 0, 0)
            local_points = data[local_index]
            local_z_min = np.min(local_points[:,2])
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_points[:,2] = local_points[:,2] - np.min(local_points[:,2])
            local_abs_height = np.max(local_points[:,2])
            print(">>> local abs height :", local_abs_height)
            print(">>> local data.shape :", local_points.shape)
            print(">>> local data shifted")
            
            # voxelization
            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            #print(">> voxel.shape :",voxel.shape, "should be equal to: ({},{},{})".format(cub_s_nb, cub_s_nb, cub_h_nb))
            #print(">>>>>>>>>>>>>>>>> voxel[0:10]", voxel[0:10])
            voxel_skeleton_cuboid[w_nb] = voxel

            # the number of local_points
            nb_local_points = len(local_points)
            tmp_nb_sample = int(nb_local_points/sample_size)
            
            print(">>> nb_sample={}".format(nb_sample))
            # set sample_cuboid_index
            [sample_cuboid_index.append([i_s, w_nb]) for i_s in range(nb_sample, nb_sample+tmp_nb_sample)]
            nb_sample = nb_sample + tmp_nb_sample
            print(">>> tmp_nb_sample={}, nb_sample+tmp={}".format(tmp_nb_sample, nb_sample))

            # set samples
            np.random.shuffle(local_points)
            tmp_samples = [local_points[sample_size*i_t:sample_size*(i_t+1)] for i_t in range(tmp_nb_sample)]
            [samples.append(item) for item in tmp_samples]

            #print(">> nb_points_per_voxel.shape :",nb_points_per_voxel.shape)
            w_nb = w_nb + 1

    return np.array(samples), np.array(sample_cuboid_index), voxel_skeleton_cuboid


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

def old_code_befor_0405():
        # to do : add more overlap between the cubes
    # beta version
    # record of point numbers in each cube
    w_nb = 0
    tmp = []
    global_height = z_max - z_min

    for i in range(d1):
        for j in range(d2):
            w_nb = w_nb + 1
            # (1) cut data to cubes
            # local origin
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')
            
            # find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data_preprocessed, local_x, local_x+grid_size, local_y, local_y+grid_size)

            # shift points to local origin (0, 0, 0)
            local_points = data_preprocessed[local_index]
            #if local_points.size < 1550000:
            if local_points.size < 10:
                print(">> Local_points is empty, no points founds here!")
                continue
            
            local_z_min = np.min(local_points[:,2])
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_points[:,2] = local_points[:,2] - np.min(local_points[:,2])
            local_abs_height = np.max(local_points[:,2])
            print(">> local abs height : ", local_abs_height)
            print(">> local data.shape :", local_points.shape)
            print(">> local data shifted")
            #print(">> local_points:", local_points[0:10])
            tmp.append(local_points.size)

            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            print(">> voxel.shape :",voxel.shape)
            print(">> nb_points_per_voxel.shape :",nb_points_per_voxel.shape)
            #print(voxel[0:10])
            #visualize_voxel_key_points(voxel, nb_points_per_voxel, "voxel - cuboid "+str(w_nb))
            # a remove
            visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

            print(key_points_in_voxel[0:10])
            print(key_points_in_voxel.shape)
            visualize_voxel_key_points(key_points_in_voxel, nb_points_per_voxel, voxel_sample_mode + " key points in voxel - cuboid "+str(w_nb) + "ratio nb_point/max(nb_point)")
            
            # (2) put cube data to device : (cpu or gpu)

            # prepar voxel
            voxel_skeleton = np.zeros([my_model.x_v, my_model.y_v, my_model.z_v], dtype=int)
            voxel_skeleton[voxel[:,0], voxel[:,1], voxel[:,2]] = 1
            
            # (3) feed it to the model
            #my_trainer.train_model(nb_epoch, local_points, voxel_skeleton)



