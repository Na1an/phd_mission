import os
from datetime import datetime
from utily import *

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
    
    gd_fa_min = 100000
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

def cauculate_ier(voxels, voxel_low, voxels_in_comp, voxel_size, nb_component):
    # mean coordinate of lowest voxel
    points, gd = voxels[voxel_low]
    if gd != 0:
        print("problem here, def cauculate_ier")
    coord_low = np.mean(points[:,:3], axis=0)
    len_points = len(points)
    feature_add = np.zeros((len_points, 3), dtype=points.dtype)
    len_f = len(points[0])
    points = np.concatenate((points,feature_add), axis=1)
    #points[:,len_f] = gd
    #points[:,len_f+1] = 0 # IER
    points[:,len_f+2] = nb_component
    voxels[voxel_low] = points, gd
    
    print("points[0]=",points[0])
    for k in voxels_in_comp:
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

# calculate the geodesic diatance of a voxelized space (cuboid)
def geodesic_distance(voxels, voxel_size):
    '''
    Args:
        voxles: a dict. Key is the coordinates of the occupied voxel and value is the points inside the voxel and geodesic distance initialised to 0.
    Return:
        None.
    '''
    #remaining_voxel = len(voxels)
    nb_component = 0
    while(assignment_incomplete(voxels)):
        #print("voxel remaining={}".format(remaining_voxel))
        (x_low, y_low, z_low) = lowest_voxel(voxels)
        #voxel_low = lowest_voxel(voxels)
        q_v = deque([(x_low, y_low, z_low)])
        seen = set()
        while(len(q_v)>0):
            print("len(q_v)={}".format(len(q_v)))
            v_act = q_v.popleft() # coordooné d'un voxel
            #print("v_act={}".format(v_act))
            father, child, gd_fa_min = find_neighbours_and_assign_gd(v_act, voxels)
            
            points,_ = voxels[v_act]
            voxels[v_act] = points, gd_fa_min+1
            x_a,y_a,z_a = v_act
            
            if len(father)==0:
                points,_ = voxels[v_act]
                voxels[v_act] = points, 0
            else:
                if (gd_fa_min+1) > 3*(z_a-z_low):
                    continue
                
            for c in child:
                if c not in seen:
                    q_v.append(c)
                    seen.add(c) # seen add the coordinate of the voxels
            #print("queue =", q_v)
            #print("child={}, father={}".format(child,father))
        
        # when a set of component is processed
        cauculate_ier(voxels, (x_low, y_low, z_low), seen, voxel_size, nb_component)
        print(">> component n°{} : ier calculated".format(nb_component))
        nb_component = nb_component + 1
        
    print(">> All voxels are processed, we have {} component in this zone".format(nb_component))
    return voxels

def write_dict_data(voxels, path):
    
    start = True
    for _,v in voxels.items():
        points,_ = v
        if start:
            local_points = np.array(points)
            start = False
        else:
            local_points = np.concatenate((local_points, points), axis=0)
        #print(">>> points.shade={}".format(points.shape))
    local_points = np.array(local_points)
    print(">>> local_points.shape={}".format(local_points.shape))

    new_file = laspy.create(point_format=3)
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="ier", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="gd", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.float64))
    new_file.x = local_points[:,0]
    new_file.y = local_points[:,1]
    new_file.z = local_points[:,2]
    new_file.ier = local_points[:,-2] #ier
    new_file.gd = local_points[:,-3] #gd
    new_file.tree_id = local_points[:,-1] #tree_id/nb_component
    new_file.write(path)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (train data with labels).", type=str)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--grid_size", help="The grid_size.", type=float, default=10.0)
    args = parser.parse_args()
    data_path = args.data_path
    voxel_size = args.voxel_size
    grid_size = args.grid_size
    
    x,y = (10,10)

    data, x_min, x_max, y_min, y_max, z_min, z_max = read_data_with_intensity(data_path, "intensity", detail=True)
    local_index = get_region_index(data, x, x+grid_size, y, y+grid_size)
    local_points = data[local_index]
    local_z = np.min(local_points[:,2])
    local_points[:,0] = local_points[:,0] -int(np.min(local_points[:,0]))
    local_points[:,1] = local_points[:,1] - int(np.min(local_points[:,1]))
    local_abs_height = np.max(local_points[:,2]) - local_z
    # local_abs_height
    local_points[:,2] = local_points[:,2] - local_z

    key_points_in_voxel, nb_points_per_voxel, voxel = voxel_grid_sample(local_points, voxel_size, 'mc')

    initialize_voxels(key_points_in_voxel)
    geodesic_distance(key_points_in_voxel, voxel_size)
    #plot_voxels(voxels_dict_to_numpy(key_points_in_voxel), grid_size, voxel_size)

    output = os.getcwd()+"/test_ier_"+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +"_" + str(voxel_size) + ".las"
    write_dict_data(key_points_in_voxel, output)