from utility import *
import numpy as np
# main
if __name__ == "__main__":

    # build arguments
    parser = ap.ArgumentParser(description="Convert lidar to BEV, support only .data for the moment.")
    parser.add_argument("target", help="The path of the target file (no label).", type=str)

    args = parser.parse_args()

    # take parameters
    target_path = args.target

    # read data
    print("> Reading data from target:", target_path)
    data_target = read_header(target_path)
    x_min_target, x_max_target, y_min_target, y_max_target = get_info(data_target)
    data_target.WL[np.where(data_target.WL==1)]+=3

    #
    #data_target.add_extra_dim(laspy.ExtraBytesParams(name="return_ratio", type=np.float64))
    #data_target.return_ratio = (data_target['return_number'])/(data_target['number_of_returns'])

    # store the result
    
    path = os.getcwd()+"/new_label_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".las"
    data_target.write(path)
    print("> new file writed to {}".format(path))

    print("###### Transpose End! ######")