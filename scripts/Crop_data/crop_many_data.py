import argparse as ap
from process_data import *

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("tls_dir", help="The path of TLS data.", type=str)
    parser.add_argument("dls_path", help="The path of DLS data.", type=str)
    args = parser.parse_args()

    # take arguments
    tls_dir = args.tls_dir
    dls_path = args.dls_path
    #tls_paths = [ tls_dir+'/'+ str(i) +'.laz' for i in range(1,25)]
    #print(tls_paths)

    
    # (1) preprocess data and get set of sliding window coordinates
    print("> input data dls_path:", dls_path)
    dls_data_processed, x_min_d, x_max_d, y_min_d, y_max_d, z_min_d, z_max_d = read_data(dls_path, detail=True)
    print("> dls_data_preprocess.shape =", dls_data_processed.shape, '\n')

    for i in range(1,25):
        tls_path = tls_dir+'/'+ str(i) +'.laz'
        print("> input data tls_path:", tls_path)
        tls_data_processed, x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t = read_data(tls_path, detail=True)
        print("> tls_data_preprocess.shape =", tls_data_processed.shape, '\n')
    
        # find overlap region
        x_min_overlap = max(x_min_d, x_min_t)
        y_min_overlap = max(y_min_d, y_min_t)
        x_max_overlap = min(x_max_d, x_max_t)
        y_max_overlap = min(y_max_d, y_max_t)

        index_dls = get_region_index(dls_data_processed, x_min_overlap, x_max_overlap, y_min_overlap, y_max_overlap)
        write_data(dls_data_processed[index_dls], "dls_data_croped_" + str(i)+'_')