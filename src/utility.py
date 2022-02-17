import os
import laspy
import open3d 
import numpy as np
import argparse as ap
from datetime import datetime
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from collections import Counter

def read_header(path):
    '''
    Args:
        path : a string. The path of file.
    Returns:
        The file's header and data. (and VLRS if it has)
    '''
    return laspy.read(path)

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

    return x_min, x_max, y_min, y_max

def check_classfication(las):
    '''
    Args:
        las : a laspy.lasdata.LasData.
    Returns:
        None.
    '''
    # what classcification is
    print("lala >> :", las[las["classification"] !=0])
    print("lala >> :", np.array(las[las["classification"] != 0]))
    
    return None

# coloring the voxels
def voxel_coloring(las, d_name):
    '''
    Args:
        las : a laspy.lasdata.LasData. The input data.
        d_name : a string. The dimension name of the las header.
    Returns:
        res : pcd.colors for coloring our voxels.   
    '''
    #open3d.utility.Vector3dVector(np.vstack((las.red, las.green, las.blue)).transpose())
    d_max = np.max(las[d_name])
    return open3d.utility.Vector3dVector(np.vstack((las[d_name]/(2*d_max), las[d_name]/(1.8*d_max), las[d_name]/d_max)).transpose())  

# store the processing data as the voxels
def sotre_as_mesh(v_grid, v_size):
    '''
    Args:
        v_grid : a open3d.cpu.pybind.geometry.VoxelGrid. voxel grid.
        v_size : a numpy.float64. voxel size.
    Returns:
        None.   
    '''
    # store our voxel cubes
    voxels = v_grid.get_voxels()
    vox_mesh = open3d.geometry.TriangleMesh()

    for v in voxels:
        cube=open3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color(v.color)
        cube.translate(v.grid_index, relative=False)
        vox_mesh+=cube
    
    vox_mesh.translate([0.5,0.5,0.5], relative=True)
    vox_mesh.scale(v_size, [0,0,0])
    vox_mesh.translate(v_grid.origin, relative=True)
    vox_mesh.merge_close_vertices(0.0000001)
    open3d.io.write_triangle_mesh(os.getcwd()+"/res_voxel_mesh_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".ply", vox_mesh)

    return None

# reformulate the data
def reformulate_data(las, label):
    '''
    Args:
        las : a laspy.lasdata.LasData. The data with label.
    Returns:
        None.
    '''

    return np.vstack(las.x, las.y, las.z) 

# get the study region
def get_region_indice(data, x_min, x_max, y_min, y_max, blank):
    return np.where(((x_min+blank<data[:,0]) & (data[:,0]<x_max-blank)) & ((y_min+blank<data[:,1]) & (data[:,1]<y_max-blank)))

# transpose the labels from one point cloud to another
# 问题在这！
def transpose(target, ref, indice_region):
    '''
    Args:
        origin : a point cloud, 3d data. The data with label.
        traget : a point cloud, 3d data. The target to transpose the label.
        k : a integer. The number the neighbours to consider.
    Returns:
        None.
    '''
    print(">> [Time consuming part] ok, let's wait")
    #kdt = KDTree(ref[:, 0:3], leaf_size=(len(ref)-10), metric="euclidean")
    #btree = BallTree
    neigh = NearestNeighbors(n_neighbors=3, radius=0.5)
    neigh.fit(ref[:, 0:3])
    # for each point in the target, we search k cloest points in the ref
    #dist, ind = kdt.query(target[indice_region], k=3, return_distance=True)
    dist, ind = neigh.kneighbors(target[indice_region][:, 0:3], return_distance=True)

    # Data cannot be modified directly on np.ndarray[indice], that's why we need temporary res
    res = np.empty(len(indice_region[0]), dtype=float)
    
    for i in range(len(indice_region[0])):
        # the most frequent label
        t = Counter(ref[ind[i]][:,3]).most_common(1)[0][0]
        res[i] = t
    
    return res

