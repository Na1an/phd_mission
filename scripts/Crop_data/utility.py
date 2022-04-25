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
    print(">> Dimension names:", list(point_format.dimension_names))

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