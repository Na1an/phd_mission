import argparse as ap
from preprocess_data import *

''''
path = "/home/yuchen/Documents/PhD/data_for_project/22-03-23_tls_labelled_data/res_bis_crop.las"
data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = preprocess_data(path, "WL", detail=True)

res = sliding_window(x_min, x_max, y_min, y_max, 5)
print(res)
'''

if __name__ == "__main__":
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (TLS data with labels).", type=str)
    args = parser.parse_args()

    # take parameters
    raw_data_path = args.data_path

    # (1) preprocess data and get set of sliding window coordinates
    path = "/home/yuchen/Documents/PhD/data_for_project/22-03-23_tls_labelled_data/res_bis_crop.las"
    data_preprocessed, x_min, x_max, y_min, y_max, z_min, z_max = read_data(path, "WL", detail=True)

    coords_sw = sliding_window(x_min, x_max, y_min, y_max, 5)
    (d1,d2,_) = coords_sw.shape
    
    w_nb = 1
    for i in range(d1):
        for j in range(d2):
            # local origin
            local_x, local_y = coords_sw[i, j]
            print("> sliding window n°", w_nb, "bottom left coordinate :(",local_x, ',',local_y,')')
            