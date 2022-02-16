from utility import *

def dist_to_center(a, p, r):
    print(a.shape)
    d = np.linalg.norm(a[0:2]-p)

    a = np.array([[5.0, 6.0, 7.0, 1.0], [5.0, 12.0, 2.0, -1.0], [2.2, 3.5, 4.1, 2.0]])
    p = np.array([4.0, 5.0])
    
    ind = np.where((a[:,0] > 1.0) & (a[:,0] < 6.0) & (a[:,1] > 1.0) & (a[:,1] < 8.0))
    print(a[ind])

    return  None

def get_region_indice(data, x_min, x_max, y_min, y_max, blank):
    return np.where(((data[:,0]>x_min+blank) & (data[:,0]<x_max-blank)) & ((data[:,1]>y_min+blank) & (data[:,1]<y_max-blank)))

if __name__ == "__main__":
    
    # build arguments
    parser = ap.ArgumentParser(description="Convert lidar to BEV, support only .data for the moment.")
    parser.add_argument("target", help="The path of the target file (no label).", type=str)
    parser.add_argument("ref", help="The path of the ref file (with label).", type=str)
    args = parser.parse_args()

    # take parameters
    target_path = args.target
    ref_path = args.ref
    
    # read data
    print("> Reading data from target:", target_path)
    data_target = read_header(target_path)
    x_min_target, x_max_target, y_min_target, y_max_target = get_info(data_target)

    print("> Reading data from ref:", ref_path)
    data_ref = read_header(ref_path)
    x_min_ref, x_max_ref, y_min_ref, y_max_ref = get_info(data_ref)

    # reformulate data
    data_target_rf = np.vstack((data_target.x, data_target.y, data_target.z, -np.ones(len(data_target)))).transpose()
    print(">>> data target reformulated :", data_target_rf[0:10], " shape =", data_target_rf.shape)

    # ref label: semented -> 0/dtm, 1/leaf, 2/cwd, 3/wood segmented_clean -> 1/dtm, 2/leaf, 3/cwd, 4/wood
    data_ref_rf = np.vstack((data_ref.x, data_ref.y, data_ref.z, data_ref.label)).transpose()
    print(">>> data ref reformulated :", data_ref_rf[0:10], " shape =", data_ref_rf.shape)
    
    indice_region_t = get_region_indice(data_target_rf[:,0:2], x_min_ref, x_max_ref, y_min_ref, y_max_ref, 0.2)
    print(">> we have picked :", indice_region_t[0].shape, "points to receive the transpose")
    #data_target_rf[indice_region_t]

    print("###### Transpose End! ######")