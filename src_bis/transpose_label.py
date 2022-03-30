from utility import *

# main
if __name__ == "__main__":

    # build arguments
    parser = ap.ArgumentParser(description="Convert lidar to BEV, support only .data for the moment.")
    parser.add_argument("target", help="The path of the target file (no label).", type=str)
    parser.add_argument("ref", help="The path of the ref file (with label).", type=str)
    parser.add_argument("--ref_bis", help="The path of the ref bis file (with label), only for shift.", type=str)
    parser.add_argument("--shift_ref", help="If we need shift our reference data.", type=bool, default=False)
    args = parser.parse_args()

    # take parameters
    target_path = args.target
    ref_path = args.ref
    ref_path_bis = args.ref_bis
    shift_ref = args.shift_ref
    
    # read data
    print("> Reading data from target:", target_path)
    data_target = read_header(target_path)
    x_min_target, x_max_target, y_min_target, y_max_target = get_info(data_target)

    print("> Reading data from ref:", ref_path)
    data_ref = read_header(ref_path)
    x_min_ref, x_max_ref, y_min_ref, y_max_ref = get_info(data_ref)
    print("> shift reference data is", shift_ref)

    # reformulate data
    data_target_rf = np.vstack((data_target.x, data_target.y, data_target.z, -1.0*np.ones(len(data_target)))).transpose()
    print(">>> data target reformulated :", data_target_rf[0:10], " shape =", data_target_rf.shape)

    # ref label: segmented -> 0/dtm, 1/leaf, 2/cwd, 3/wood segmented_clean -> 1/dtm, 2/leaf, 3/cwd, 4/wood
    # here we use segmented.las file for testing
    if shift_ref is True:
        print("> ok, wee need shift ref data")
        x_min_ref_bis, x_max_ref_bis, y_min_ref_bis, y_max_ref_bis = get_info(read_header(ref_path_bis))
        data_ref_rf = np.vstack((data_ref.x, data_ref.y, data_ref.z, data_ref["Scalar field"])).transpose()
        data_ref_rf[:,0] = data_ref_rf[:,0] + x_min_ref_bis
        data_ref_rf[:,1] = data_ref_rf[:,1] + y_min_ref_bis
        indice_region_t = get_region_indice(data_target_rf[:,0:2], x_min_ref_bis, x_max_ref_bis, y_min_ref_bis, y_max_ref_bis, 0.3)

    else:
        data_ref_rf = np.vstack((data_ref.x, data_ref.y, data_ref.z, data_ref.WL+1.0)).transpose()
        indice_region_t = get_region_indice(data_target_rf[:,0:2], x_min_ref, x_max_ref, y_min_ref, y_max_ref, 0.3)
    
    print(">>> data ref reformulated :", data_ref_rf[0:10], " shape =", data_ref_rf.shape)
    print(">> we have picked :", indice_region_t[0].shape, "points to receive the transpose")
    
    data_tmp = data_target_rf[indice_region_t]
    data_tmp[:,3] = transpose(data_target_rf, data_ref_rf, indice_region_t, k=5)

    # assign the label to dls data
    data_target_rf[indice_region_t] = data_tmp
    data_target.add_extra_dim(laspy.ExtraBytesParams(name="llabel", type=np.float64))
    data_target.llabel = -1.0*np.ones(len(data_target))
    data_target.llabel[indice_region_t] = data_tmp[:,3]

    # store the result
    data_target.write(os.getcwd()+"/new_file_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".las")
    print("###### Transpose End! ######")