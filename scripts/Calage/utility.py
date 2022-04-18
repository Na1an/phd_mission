import os
import torch
import laspy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits import mplot3d

# print info of the laspy data
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

# visualize key point of voxels
def visualize_voxel_key_points(points, points_per_voxel, title, only_points=False):
    '''
    Args:
        points: a (x,y,z) np.darray.
        points_per_voxel: a (m,n) np.darray. m is the points number in each voxel, n is index of voxel.
        title: a string.
    Returns:
        None.
    '''
    sns.set(style = "darkgrid")
    ax = plt.axes(projection='3d')
    ax.set_title(title)

    if only_points:
        sc = ax.scatter(points[:,0], points[:,1], points[:,2], s=10, cmap="gist_rainbow")
    else:
        sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=points_per_voxel, s=20, cmap="gist_rainbow")
    #plt.gca().invert_xaxis()
    #plt.legend("nb points")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.colorbar(sc, fraction=0.020, pad=0.04) 
    plt.show()
    
    return None

# old code
def old_code():
    '''
    # sliding window
    coords_sw = sliding_window(0, x_max - x_min, 0, y_max - y_min, grid_size)
    (d1,d2,_) = coords_sw.shape
    #print("> coords :", coords_sw)

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
            
            # find index of the tls_data_processed in this sliding window
            local_index = get_region_index(tls_data_processed, local_x, local_x+grid_size, local_y, local_y+grid_size)

            # shift points to local origin (0, 0, 0)
            local_points = tls_data_processed[local_index]
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
            visualize_voxel_key_points(voxel, nb_points_per_voxel, "voxel - cuboid "+str(w_nb))

            #print(key_points_in_voxel[0:10])
            #print(key_points_in_voxel.shape)
            #visualize_voxel_key_points(key_points_in_voxel, nb_points_per_voxel, voxel_sample_mode + " key points in voxel - cuboid "+str(w_nb) + "ratio nb_point/max(nb_point)")
        
    #print(tmp)
    print("\n###### End ######")
    '''
    return None