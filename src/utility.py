import os
import torch
import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score
import seaborn as sns
import pandas as pd

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

# create confusion matrix
def createConfusionMatrix(loader):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()

# to do
# setting device
def setting_device():
    return None

# once we have tn, fp, fn, tp = cf_matrix.ravel()
# we then calculate recall, precision
def calculate_recall_precision(tn, fp, fn, tp):
    # Sensitivity, hit rate, recall, or true positive rate
    recall = tp/(tp+fn)
    # Specificity or true negative rate
    specificity = tn/(tn+fp)
    # Precision or positive predictive value
    precision = tp/(tp+fp)
    # Negative predictive value
    npv = tn/(tn+fn)
    # Fall out or false positive rate
    fpr = fp/(fp+tn)
    # False negative rate
    fnr = fn/(tp+fn)
    # False discovery rate
    fdr = fp/(tp+fp)

    # Overall accuracy
    acc = (tp+tn)/(tp+fp+fn+tn)
    return recall, specificity, precision, npv, fpr, fnr, fdr, acc

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
            
    return np.array(samples), sample_cuboid_index, voxel_skeleton_cuboid