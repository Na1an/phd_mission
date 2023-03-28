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

'''
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

def get_region_index(data, x_min, x_max, y_min, y_max):
    '''
    Args:
        data : a 4-D np.darray. (x,y,z,label)
    Returns:
        numpyp index, which are in this region.
    '''
    return np.where((((x_min)<data[:,0]) & (data[:,0]<(x_max))) & (((y_min)<data[:,1]) & (data[:,1]<(y_max))))

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

# get current directory path
def get_current_direct_path():
    return os.path.dirname(os.path.abspath(__file__))

# to do
# setting device
def setting_device():
    return None

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

def normalize_point_cloud_remake(pc, mode="gm"):
    if mode == "cm":
        centroid = np.mean(pc, axis=0) # 求取点云的中心
    elif mode == "gm":
        centroid = np.mean(pc, axis=0) # geometric center
    pc = pc - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized = pc / m # 依据长轴将点云归一化到 (-1, 1)
    return pc_normalized, centroid, m  # centroid: 点云中心, m: 长轴长度, centroid和m可用于keypoints的计算

# plot pc
def plot_pc(data, c=1):
    
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c, cmap=plt.hot())
    plt.show()

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

def check_nan_in_array(feature_name,a):
    print(feature_name + "shape={} nan size={}".format(a.shape, a[np.isnan(a)].shape))
    return None

# for prepare dataset
def old_prepare_dataset_copy(data, coords_sw, grid_size, voxel_size, global_height, voxel_sample_mode, sample_size, detail=False):
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
    
    for i in range(coord_x):
        for j in range(coord_y):   
            # (1) global coordinates of each cuboids
            local_x, local_y = coords_sw[i, j]
            print("\n>> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')

            # (2) find index of the data_preprocessed in this sliding window
            local_index = get_region_index(data, local_x, local_x+grid_size, local_y, local_y+grid_size)
           
            if len(local_index[0]) < 1:
                print(">> point number not enough, cuboid-{} skiped".format(w_nb))
                voxel_skeleton_cuboid[w_nb] = []
                w_nb = w_nb + 1
                continue
        
            # (3) shift points to local origin (0, 0, 0) and zero-centered
            #local_points = copy.deepcopy(data[local_index])
            local_points = data[local_index]
            local_points[:,0] = local_points[:,0] - local_x
            local_points[:,1] = local_points[:,1] - local_y
            local_abs_height = np.max(local_points[:,2]) - np.min(local_points[:,2])
            # local_abs_height
            local_points[:,2] = local_points[:,2] - np.min(local_points[:,2])

            if detail:
                print(">>> local abs height :", local_abs_height)
                print(">>> local data.shape :", local_points.shape)
                print(">>> local_data (points in cuboid) zero-centered and standardization/normalization")
                #print(">>> local_points=", local_points[:,:3][0:100])

            # (4) voxelization
            key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, voxel_sample_mode)
            voxel_skeleton_cuboid[w_nb] = voxel
            #visualize_voxel_key_points(voxel, nb_points_per_voxel, "TLS voxelized data")

            # (5) centralization in (x y z) thress axis by the center of voxels
            #local_points[:,:3] = local_points[:,:3] - np.mean(local_points[:, :3], axis=0)
            local_points[:,:3] = local_points[:,:3] - np.array([grid_size/2, grid_size/2, global_height/2])
            local_points[:,:2] = local_points[:,:2]/grid_size
            local_points[:,2] = local_points[:,2]/global_height
            
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

            data_augmentation()

            w_nb = w_nb + 1
            
    return np.array(samples), sample_cuboid_index, 
    
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

# L1/L2
#!!!!!!!!!!!!!!!!!!!!!! à faire
# calculate auroc score
def calculate_auroc(y_score, y_true):
    '''
    Args:
        y_score: a numpy.ndarray. (n, 2)
        y_true: a numpy.ndarray. (n, 2)
    Returns:
        res: a float. The auroc score of logits and label. We should note here: leaf label=0, wood label=1
    '''

    roc_auc_score(y_score=logits[:,1], y_true=label[:,1])
    return 0

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
    for k,v in voxels.items():
        _, gd = v
        if gd>=0:
            continue
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
    return (((x_a-x_b)**2 + (y_a-y_b)**2 + (z_a - z_b**2))**0.5)

# calculate the geodesic diatance of a voxelized space (cuboid)
def geodesic_distance(voxels, voxel_size, tree_radius=7.0, limit_comp=10, limit_p_in_comp=100):
    '''
    Args:
        voxles: a dict. Key is the coordinates of the occupied voxel and value is the points inside the voxel and geodesic distance initialised to 0.
    Return:
        None.
    '''
    #remaining_voxel = len(voxels)
    nb_component = 0
    nb_v_keep, nb_v_abandon, nb_p_keep, nb_p_abandon = 0, 0, 0, 0

    while(assignment_incomplete(voxels)):
        #print("voxel remaining={}".format(remaining_voxel))
        #(x_low, y_low, z_low) = lowest_voxel(voxels)
        voxel_low = lowest_voxel(voxels)
        (x_low, y_low, z_low) = voxel_low
        q_v = deque([(x_low, y_low, z_low)])
        seen = set()
        seen.add(voxel_low)
        nb_in_comp = 0
        nb_p_in_comp = 0
        while(len(q_v)>0):
            #print("len(q_v)={}".format(len(q_v)))
            v_act = q_v.popleft() # coordooné d'un voxel
            nb_in_comp = nb_in_comp + 1
            #print("v_act={}".format(v_act))
            father, child, gd_fa_min = find_neighbours_and_assign_gd(v_act, voxels)
            
            points,_ = voxels[v_act]
            voxels[v_act] = points, gd_fa_min+1
            nb_p_in_comp = nb_p_in_comp + len(points)
            x_a,y_a,z_a = v_act
            
            if len(father)==0:
                points,_ = voxels[v_act]
                voxels[v_act] = points, 0
            else:
                # here, setting the limits about IER or geodesic distance
                # extend limit

                if ((gd_fa_min+1)/dist_3d(v_act, voxel_low)) > 1.5:
                    continue

                if (gd_fa_min+1) > 10:
                    continue
                '''
                # 关于高度与gd的限制
                if (gd_fa_min+1) > 2.5*(z_a-z_low):
                    continue
                
                # 关于树木直径的限制
                #if (((x_a-x_low)**2 + (y_a-y_low)**2)**0.5) > tree_radius:
                    #continue
                
                # 树木直径的限制
                # radius limit
                if dist((x_a,y_a), (x_low, y_low))* voxel_size > tree_radius:
                    continue
                '''

            for c in child:
                if c not in seen:
                    q_v.append(c)
                    seen.add(c) # seen add the coordinate of the voxels
            #print("queue =", q_v)
            #print("child={}, father={}".format(child,father))
        
        # when a set of component is processed
        if nb_in_comp < limit_comp or nb_p_in_comp < limit_p_in_comp:
            nb_v_abandon = nb_v_abandon + nb_in_comp
            nb_p_abandon = nb_p_abandon + nb_p_in_comp
            for i in seen:
                del voxels[i]
        else:
            nb_v_keep = nb_v_keep + nb_in_comp
            nb_p_keep = nb_p_keep + nb_p_in_comp
            cauculate_ier(voxels, (x_low, y_low, z_low), seen, voxel_size, nb_component)
            print(">> {} voxels in component n°{} : ier calculated".format(nb_in_comp, nb_component))
            nb_component = nb_component + 1
        
    print(">> All voxels are processed, we have {} component in this zone".format(nb_component))
    print(">> {} voxels keeped, {} voxels abondaned because of small component, remove {}% voxels.".format(nb_v_keep, nb_v_abandon, round((100*nb_v_abandon)/(nb_v_abandon+nb_v_keep),2)))
    print(">> {} points keeped, {} points abondaned because of small component, remove {}% points.".format(nb_p_keep, nb_p_abandon, round((100*nb_p_abandon)/(nb_p_abandon+nb_p_keep),2)))

    return voxels, nb_component

