import os
import copy
import math
import torch
import laspy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque 
from mpl_toolkits import mplot3d
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

''' 
for maket model structure plot 
from torchsummary import summary
summary = summary(model, (1, 512, 512))
'''

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

# get current directory path
def get_current_direct_path():
    return os.path.dirname(os.path.abspath(__file__))

def dist(a,b):
    x_a, y_a = a
    x_b, y_b = b
    return (((x_a-x_b)**2 + (y_a-y_b)**2)**0.5)

# from pointnet++
# centerization by long axe
def normalize_long_axe(pc):
    #print("pc.shape = {}".format(pc.shape))
    max_x_axe = np.max(pc[:,0])
    max_y_axe = np.max(pc[:,1])
    max_z_axe = np.max(pc[:,2])

    max_axe = max(max(max_x_axe, max_y_axe), max_z_axe)
    #print("max_x_axe={}, max_y_axe={}, max_z_axe={}, max_axe = {}".format(max_x_axe, max_y_axe, max_z_axe, max_axe))
    pc[:,:3] = pc[:,:3]/max_axe
    pc[:,0] = pc[:,0] + ((1 - np.max(pc[:,0]))/2)
    pc[:,1] = pc[:,1] + ((1 - np.max(pc[:,1]))/2)
    pc[:,2] = pc[:,2] + ((1 - np.max(pc[:,2]))/2)
    #print("after pc.shape = {}".format(pc.shape))
    return pc, max_axe, max_x_axe, max_y_axe, max_z_axe

def check_nan_in_array(feature_name,a):
    print(feature_name + "shape={} nan size={}".format(a.shape, a[np.isnan(a)].shape))
    return None
    
# stat numpy
def show_numpy_stat(data):
    print(pd.DataFrame(data).describe())
    print("\n")
    return None

# normalize feature
def normalize_feature(f):
    f_min = np.min(f)
    f_max = np.max(f)
    print("f_min={}, f_max={}".format(f_min, f_max))
    return (2*(f-f_min)/(f_max-f_min)) - 1

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# we then calculate recall, precision
def calculate_recall_precision(tn, fp, fn, tp):
    #confusion_matrix(y_true, y_pred)
    # Sensitivity, hit rate, recall, or true positive rate
    recall = tp/(tp+fn)
    # Specificity or true negative rate
    if tn==0:
        specificity = 0
    else:
        specificity = tn/(tn+fp)
    # Precision or positive predictive value
    precision = tp/(tp+fp)

    # Overall accuracy
    acc = (tp+tn)/(tp+fp+fn+tn)
    return recall, specificity, precision, acc

def split_reminder(x, chunk_size, axis=0):
    indices = np.arange(chunk_size, x.shape[axis], chunk_size)
    return np.array_split(x, indices, axis)

#######
# ier #
#######
# find the lowest voxel
def lowest_voxel(voxels):
    (x_min, y_min, z_min) = (10000, 10000, 10000)
    for k in voxels:
        _,_,z = k
        if z<z_min:
            (x_min, y_min, z_min) = k
    return (x_min, y_min, z_min)

# find neighbour voxels
def find_neighbours_and_assign_gd(v_act, voxels):
    father = []
    child = []
    x,y,z = v_act
    adjacent = [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]
    
    gd_fa_min = 10000000
    for e in adjacent:
        if e in voxels:
            points,gd = voxels[e]
            if gd<0:
                child.append(e)
            if gd>=0:
                father.append(e)
                gd_fa_min = gd if gd<gd_fa_min else gd_fa_min
            
    return father, child, gd_fa_min

# check complete
def assignment_incomplete(voxels):
    for k,v in voxels.items():
        points, gd = v
        if gd<0:
            return True
    return False

def initialize_voxels(voxels):
    # key_points_in_voxel is a dict
    # key: (x,y,z) coords of voxelized space
    # values: (a list of points in the voxel, geodesic distance)
    for k,v in voxels.items():
        # initialize geodesic distance to -1
        voxels[k] = (v,-1)

def cauculate_ier(voxels, voxel_low, seen, voxel_size, nb_component, limit_comp=10):
    # mean coordinate of lowest voxel
    points, gd = voxels[voxel_low] 
    '''
    if gd != 0:
        print("problem here, def cauculate_ier")
    '''

    coord_low = np.mean(points[:,:3], axis=0)
    len_f = len(points[0])

    for k in seen:
        points, gd = voxels[k]
        len_points = len(points)
    
        feature_add = np.zeros((len_points, 3), dtype=points.dtype)
        points = np.concatenate((points,feature_add), axis=1)
        points[:,len_f] = gd

        for i in range(len(points)):
            ed = np.linalg.norm(points[i][:3] - coord_low)
            points[i][len_f+1] = (gd * voxel_size)/ed # ier
            points[i][len_f+2] = nb_component
        
        voxels[k] = points, gd

def dist_3d(a,b):
    x_a, y_a, z_a = a
    x_b, y_b, z_b = b
    res = (((x_a-x_b)**2 + (y_a-y_b)**2 + (z_a - z_b**2))**0.5)

    return res

# calculate the geodesic diatance of a voxelized space (cuboid)
def geodesic_distance(voxels, voxel_size, tree_radius=7.0, limit_comp=10, limit_p_in_comp=100, tls_mode=False):
    '''
    Args:
        voxles: a dict. Key is the coordinates of the occupied voxel and value is the points inside the voxel and geodesic distance initialised to 0.
    Return:
        None.
    '''
    #remaining_voxel = len(voxels)
    nb_component = 0
    nb_v_keep, nb_v_abandon, nb_p_keep, nb_p_abandon = 0, 0, 0, 0
    rest = {}
    occupied_voxel = set(list(voxels.keys()))
    while(len(occupied_voxel)>0):
        #print("occupied_voxel len=", len(occupied_voxel))
        voxel_low = lowest_voxel(occupied_voxel)
        (x_low, y_low, z_low) = voxel_low
        occupied_voxel.remove(voxel_low)

        q_v = deque([(x_low, y_low, z_low)])
        seen = set()
        seen.add(voxel_low)
        # nb_in_comp -> voxels number inside the comp
        nb_in_comp = 0
        # nb_p_in_comp -> points number inside the comp
        nb_p_in_comp = 0
        while(len(q_v)>0):
            #print("len(q_v)={}".format(len(q_v)))
            v_act = q_v.popleft() # coordooné d'un voxel
            nb_in_comp = nb_in_comp + 1
            father, child, gd_fa_min = find_neighbours_and_assign_gd(v_act, voxels)
            
            points, gd_act = voxels[v_act]
            voxels[v_act] = points, gd_fa_min+1
            nb_p_in_comp = nb_p_in_comp + len(points)
            x_a,y_a,z_a = v_act
            
            if len(father)==0 and gd_act==-1:
                points,_ = voxels[v_act]
                voxels[v_act] = points, 0
                seen.add(v_act) # seen add the coordinate of the voxels
            else:
                # here, setting the limits about IER or geodesic distance
                # extend limit
                if (gd_fa_min+1) >= 10:
                    continue

                if ((gd_fa_min+1)/dist_3d(v_act, voxel_low)) >= 1.5:
                    continue

                '''
                # limit on ratio with height in z-axis and gd
                if (gd_fa_min+1) > 2.5*(z_a-z_low):
                    continue
                
                #if (((x_a-x_low)**2 + (y_a-y_low)**2)**0.5) > tree_radius:
                    #continue
                
                # radius limit
                if dist((x_a,y_a), (x_low, y_low))* voxel_size > tree_radius:
                    continue
                '''

            for c in child:
                if c not in seen:
                    q_v.append(c)
                    seen.add(c)
                    occupied_voxel.remove(c)
        
        # when a set of component is processed
        if nb_in_comp < limit_comp or nb_p_in_comp < limit_p_in_comp:
            nb_v_abandon = nb_v_abandon + nb_in_comp
            nb_p_abandon = nb_p_abandon + nb_p_in_comp
            for i in seen:
                p_del, gd_del = voxels[i]
                len_f = len(p_del[0])
                len_points = len(p_del)
                feature_add = np.zeros((len_points, 3), dtype=points.dtype)
                p_del = np.concatenate((p_del,feature_add), axis=1)
                p_del[:,len_f] = gd_del
                p_del[:,len_f+1] = ((gd_fa_min+1)/dist_3d(v_act, voxel_low))
                p_del[:,len_f+2] = nb_component
                #rest[i] = p_del, gd_del
                rest[i] = p_del, -1
                #print("voxel corrd = {} is removed, gd_del={}".format(i, gd_del))
                del voxels[i]
        else:
            nb_v_keep = nb_v_keep + nb_in_comp
            nb_p_keep = nb_p_keep + nb_p_in_comp
            cauculate_ier(voxels, (x_low, y_low, z_low), seen, voxel_size, nb_component)
            print(">> {} voxels in component n°{} : ier calculated \t".format(nb_in_comp, nb_component), end="\r")
            nb_component = nb_component + 1
    
    # this will not give us the mosaic holes
    # the numpy shuffler will resolve the mosaic color problem
    if tls_mode:
        for k,v in rest.items():
            voxels[k] = v
        nb_component = nb_component+1
        
    print("\n>> All voxels are processed, we have {} component in this zone".format(nb_component))

    return voxels, nb_component

#####################################
# stable version for gd calculation #
#####################################
# find the lowest voxel
def lowest_voxel_stable(voxels):
    (x_min, y_min, z_min) = (10000, 10000, 10000)
    for k,v in voxels.items():
        _, gd = v
        if gd>=0:
            continue
        _,_,z = k
        if z<z_min:
            (x_min, y_min, z_min) = k
    return (x_min, y_min, z_min)

# calculate the geodesic diatance of a voxelized space (cuboid)
def geodesic_distance_stable(voxels, voxel_size, tree_radius=7.0, limit_comp=10, limit_p_in_comp=100, tls_mode=False):
    '''
    Args:
        voxles: a dict. Key is the coordinates of the occupied voxel and value is the points inside the voxel and geodesic distance initialised to 0.
    Return:
        None.
    '''
    #remaining_voxel = len(voxels)
    nb_component = 0
    nb_v_keep, nb_v_abandon, nb_p_keep, nb_p_abandon = 0, 0, 0, 0
    rest = {}
    while(assignment_incomplete(voxels)):
        #print("voxel remaining={}".format(remaining_voxel))
        voxel_low = lowest_voxel(voxels)
        (x_low, y_low, z_low) = voxel_low

        q_v = deque([(x_low, y_low, z_low)])
        seen = set()
        seen.add(voxel_low)
        # nb_in_comp -> voxels number inside the comp
        nb_in_comp = 0
        # nb_p_in_comp -> points number inside the comp
        nb_p_in_comp = 0
        while(len(q_v)>0):
            #print("len(q_v)={}".format(len(q_v)))
            v_act = q_v.popleft() # coordooné d'un voxel
            
            nb_in_comp = nb_in_comp + 1
            #print("v_act={}".format(v_act))
            father, child, gd_fa_min = find_neighbours_and_assign_gd(v_act, voxels)
            
            points,gd_act = voxels[v_act]
            voxels[v_act] = points, gd_fa_min+1
            nb_p_in_comp = nb_p_in_comp + len(points)
            x_a,y_a,z_a = v_act
            
            if len(father)==0 and gd_act==-1:
                points,_ = voxels[v_act]
                voxels[v_act] = points, 0
                seen.add(v_act) # seen add the coordinate of the voxels
            else:
                # here, setting the limits about IER or geodesic distance
                # extend limit

                if ((gd_fa_min+1)/dist_3d(v_act, voxel_low)) > 1.5:
                    continue

                if (gd_fa_min+1) > 10:
                    continue
                '''
                # limit on ratio with height in z-axis and gd
                if (gd_fa_min+1) > 2.5*(z_a-z_low):
                    continue
                
                #if (((x_a-x_low)**2 + (y_a-y_low)**2)**0.5) > tree_radius:
                    #continue
                
                # radius limit
                if dist((x_a,y_a), (x_low, y_low))* voxel_size > tree_radius:
                    continue
                '''

            for c in child:
                if c not in seen:
                    q_v.append(c)
                    seen.add(c)
        
        # when a set of component is processed
        if nb_in_comp < limit_comp or nb_p_in_comp < limit_p_in_comp:
            nb_v_abandon = nb_v_abandon + nb_in_comp
            nb_p_abandon = nb_p_abandon + nb_p_in_comp
            for i in seen:
                p_del, gd_del = voxels[i]
                len_f = len(p_del[0])
                len_points = len(p_del)
                feature_add = np.zeros((len_points, 3), dtype=points.dtype)
                p_del = np.concatenate((p_del,feature_add), axis=1)
                p_del[:,len_f] = gd_del
                p_del[:,len_f+1] = ((gd_fa_min+1)/dist_3d(v_act, voxel_low))
                p_del[:,len_f+2] = nb_component
                rest[i] = p_del, gd_del
                #print("voxel corrd = {} is removed, gd_del={}".format(i, gd_del))
                del voxels[i]
        else:
            nb_v_keep = nb_v_keep + nb_in_comp
            nb_p_keep = nb_p_keep + nb_p_in_comp
            cauculate_ier(voxels, (x_low, y_low, z_low), seen, voxel_size, nb_component)
            print(">> {} voxels in component n°{} : ier calculated \t".format(nb_in_comp, nb_component), end="\r")
            nb_component = nb_component + 1
    
    # this will not give us the mosaic holes
    # the numpy shuffler will resolve the mosaic color problem
    if tls_mode:
        for k,v in rest.items():
            voxels[k] = v
        nb_component = nb_component+1
        
    print("\n>> All voxels are processed, we have {} component in this zone".format(nb_component))

    return voxels, nb_component

##########
# Assets #
##########
def get_region_index(data, x_min, x_max, y_min, y_max):
    '''
    Args:
        data : a 4-D np.darray. (x,y,z,label)
    Returns:
        numpyp index, which are in this region.
    '''
    return np.where((((x_min)<data[:,0]) & (data[:,0]<(x_max))) & (((y_min)<data[:,1]) & (data[:,1]<(y_max))))

# plot pc
def plot_pc(data, c=1):
    
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, cmap=plt.hot())
    plt.show()

# plot voxel
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

# visualize key point of voxels
def visualize_voxel_key_points(points, points_per_voxel, title):
    '''
    Args:
        points: a (x,y,z) np.darray.
        points_per_voxel: a (m,n) np.darray. m is the points number in each voxel, n is index of voxel.
        title: a string.
    Returns:
        None.
    '''
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=(points_per_voxel/np.max(points_per_voxel)), s=20)
    #plt.gca().invert_xaxis()
    #plt.legend("nb points")
    plt.colorbar(sc, fraction=0.020, pad=0.04) 
    plt.show()
    return None