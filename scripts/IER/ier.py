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

# calculate the geodesic diatance of a voxelized space (cuboid)
def geodesic_distance(voxels, voxel_size, tree_radius=7.0, limit_comp=10):
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
        if nb_in_comp < limit_comp:
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

    new_file = laspy.create(point_format=3, file_version="1.2")
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="ier", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="gd", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="WL", type=np.float64))
    new_file.x = local_points[:,0]
    new_file.y = local_points[:,1]
    new_file.z = local_points[:,2]
    new_file.ier = local_points[:,-2] #ier
    new_file.gd = local_points[:,-3] #gd
    new_file.tree_id = local_points[:,-1] #tree_id/nb_component
    new_file.WL = local_points[:,3]
    new_file.write(path)

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (train data with labels).", type=str)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--grid_size", help="The grid_size.", type=float, default=10.0)
    parser.add_argument("--limit_comp", help="Nb of voxels below limit_comp will be removed.", type=float, default=20.0)
    parser.add_argument("--tree_radius", help="tree_radius.", type=float, default=7.0)
    args = parser.parse_args()
    data_path = args.data_path
    voxel_size = args.voxel_size
    grid_size = args.grid_size
    limit_comp = args.limit_comp
    tree_radius = args.tree_radius
    
    x,y = (0,0)

    data, x_min, x_max, y_min, y_max, z_min, z_max = read_data_with_intensity(data_path, "WL", detail=True)
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
    geodesic_distance(key_points_in_voxel, voxel_size, tree_radius=tree_radius, limit_comp=limit_comp)
    #plot_voxels(voxels_dict_to_numpy(key_points_in_voxel), grid_size, voxel_size)

    output = os.getcwd()+"/test_ier_"+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +"_" + str(voxel_size) + ".las"
    write_dict_data(key_points_in_voxel, output)