from utility import *
from sklearn.neighbors import KDTree
from collections import Counter

def get_region_indice(data, x_min, x_max, y_min, y_max, blank):
    return np.where(((data[:,0]>x_min+blank) & (data[:,0]<x_max-blank)) & ((data[:,1]>y_min+blank) & (data[:,1]<y_max-blank)))

def transpose(target, ref, indice_region):
    print(">> [Time consuming part] ok, let's wait")
    #kdt = KDTree(ref[0:3], leaf_size=(0.5*len(ref)+3), metric="euclidean")
    kdt = KDTree(ref, leaf_size=(0.5*len(ref)+3), metric="euclidean")
    # for each point in the target, we search k cloest points in the ref
    dist, ind = kdt.query(target[indice_region], k=3, return_distance=True)
    print("ind =",ind)
    print("target[indice_region] shape", target[indice_region].shape, "è--", len(indice_region_t[0]))

    res = np.empty(len(indice_region_t[0]), dtype=float)

    for i in range(len(indice_region_t[0])):
        # the most frequent label
        '''
        print(">>>>< lala >>>", ref[ind[i]][:,3])
        print(">>>>< lala >>>", int(np.bincount(ref[ind[i]][:,3])))
        print(">>>>< jiayou =", np.argmax(np.bincount(ref[ind[i]][:,3])))
        target[indice_region_t][i][3] = np.argmax(np.bincount(ref[ind[i]][:,3]))
        '''
        #print("gogo =", Counter(ref[ind[i]][:,3]).most_common(1)[0][0])
        t = Counter(ref[ind[i]][:,3]).most_common(1)[0][0]
        res[i] = t
        if t>1 :
            print("~~~~~ t=", t)
        #print("sssss =", res[i])
        #print("sssss ind = ", ind[i])

    return res

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
    data_target_rf = np.vstack((data_target.x, data_target.y, data_target.z, -1.0*np.ones(len(data_target)))).transpose()
    print(">>> data target reformulated :", data_target_rf[0:10], " shape =", data_target_rf.shape)

    # ref label: semented -> 0/dtm, 1/leaf, 2/cwd, 3/wood segmented_clean -> 1/dtm, 2/leaf, 3/cwd, 4/wood
    data_ref_rf = np.vstack((data_ref.x, data_ref.y, data_ref.z, data_ref.label)).transpose()
    print(">>> data ref reformulated :", data_ref_rf[0:10], " shape =", data_ref_rf.shape)
    
    indice_region_t = get_region_indice(data_target_rf[:,0:2], x_min_ref, x_max_ref, y_min_ref, y_max_ref, 0.2)
    print(">> we have picked :", indice_region_t[0].shape, "points to receive the transpose")
    data_tmp = data_target_rf[indice_region_t]
    data_tmp[:,3] = transpose(data_target_rf, data_ref_rf, indice_region_t)
    
    print(">> traget transposed:", data_tmp[0:10])
    #print(">> traget transposed:", data_target_rf[indice_region_t][:,3])
    #print(">> traget transposed:", transpose(data_target_rf, data_ref_rf, indice_region_t)[0:10])
    data_target_rf[indice_region_t] = data_tmp
    print("final =", data_target_rf[np.where(data_target_rf[:,3] >0)][0:13])
    data_target.add_extra_dim(laspy.ExtraBytesParams(name="llabel", type=np.float64))
    data_target.llabel = data_target_rf[:,3]
    print("final2 =", data_target.llabel[0:20])
    data_target.write("new_file_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".las")
    print("###### Transpose End! ######")