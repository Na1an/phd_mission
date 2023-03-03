import laspy
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import argparse as ap
import matplotlib.pyplot as plt
from collections import deque 

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

def get_info(las):
    '''
    Args:
        las : a LasData.
    Returns:
        x_min, x_max, y_min, y_max : a region.
    '''
    x_min, y_min, z_min = np.min(las.x), np.min(las.y), np.min(las.z)
    x_max, y_max, z_max = np.max(las.x), np.max(las.y), np.max(las.z)

    print("\n>> Point of Data Format:", las)
    #print(np.unique(las.classification))
    print(">> min: x,y,z", x_min, y_min, z_min)
    print(">> max: x,y,z", x_max, y_max, z_max)
    print(">> the data present a cube:", x_max-x_min, '*', y_max-y_min, '*', z_max-z_min)

    nb_points = las.header.point_count
    print(">> Number of points:", nb_points)

    point_format = las.point_format
    print(">> Point format (las point format):", point_format.id)
    print(">> Dimension names:", list(point_format.dimension_names), '\n')

    return x_min, x_max, y_min, y_max, z_min, z_max

def get_region_index(data, x_min, x_max, y_min, y_max):
    '''
    Args:
        data : a 4-D np.darray. (x,y,z,label)
    Returns:
        numpyp index, which are in this region.
    '''
    return np.where((((x_min)<data[:,0]) & (data[:,0]<(x_max))) & (((y_min)<data[:,1]) & (data[:,1]<(y_max))))

def normalize_feature(f):
    f_min = np.min(f)
    f_max = np.max(f)
    print("f_min={}, f_max={}".format(f_min, f_max))
    return (f-f_min)/(f_max-f_min)

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def voxel_grid_sample(cuboid, voxel_size, mode, ier=False):
    '''
    Args:
        cuboid : a (n,4) numpy.darray. The data to process.
        voxel_size : a float. The resolution of the voxel. 
        mode : a string. How to select points in voxel. ('mc': mean_center, 'cmc' : closest point to mean center)
        #grid_size : a interger/float. The side length of a grid.
        #height : a float. The max height of the raw data. Not local height!
    Returns:
        res : a voxelized data. key points in each voxel.
        nb_points_per_voxel : a list integer. The total voxel number.
        non_empty_voxel : a (n,3) np.darray. The index of occupied voxel.
        *ier : return intrinsic-extrinsic ratio if ier is True.
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

def voxels_dict_to_numpy(d):
    x_max, y_max, z_max = -1, -1, -1
    for k in d:
        x,y,z = k
        x_max = x if x>x_max else x_max
        y_max = y if y>y_max else y_max
        z_max = z if z>z_max else z_max
    
    res = np.zeros([x_max+1, y_max+1, z_max+1])
    for k,v in d.items():
        x,y,z = k
        points, gd = v
        res[x,y,z] = gd
        if gd==0:
            res[x,y,z] = -1
    return res

def plot_voxels(voxels, grid_size=0, voxel_size=0.5):
    '''
    Args:
        voxels: a nupmy.ndarray. shape (x,y,z), the value is the feature.
    Return:
        None
    '''
    mycolormap = plt.get_cmap('coolwarm')
    maxvalue = voxels.max()
    sizeall = voxels.size
    relativevalue=np.round(voxels/maxvalue,1)
    zt = np.reshape(relativevalue, (sizeall,))
    colorsvalues = mycolormap(relativevalue)
    alpha=0.5
    d,w,h=voxels.shape
    colorsvalue=np.array([(mycolormap(i)[0],mycolormap(i)[1],mycolormap(i)[2],alpha) for i in zt])
    colorsvalues=np.reshape(colorsvalue,(d,w,h,4))
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim((((10+grid_size)//2)-25)/voxel_size, (((10+grid_size)//2)+25)/voxel_size)
    ax.set_ylim((((10+grid_size)//2)-25)/voxel_size, (((10+grid_size)//2)+25)/voxel_size)
    ax.set_zlim(0, 50/voxel_size)
    p = ax.voxels(voxels, facecolors=colorsvalues, edgecolor=('k'), shade=False)
    #plt.colorbar(p, ax=ax)
    norm = matplotlib.colors.Normalize(vmin=np.min(voxels), vmax=np.max(voxels))
    m = matplotlib.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    m.set_array([])
    plt.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

def dist(a,b):
    x_a, y_a = a
    x_b, y_b = b
    return (((x_a-x_b)**2 + (y_a-y_b)**2)**0.5)

def dist_3d(a,b):
    x_a, y_a, z_a = a
    x_b, y_b, z_b = b
    return (((x_a-x_b)**2 + (y_a-y_b)**2 + (z_a - z_b**2))**0.5)