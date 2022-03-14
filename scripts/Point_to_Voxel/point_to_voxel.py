import os
import laspy as lp
import numpy as np
import open3d as o3d
import argparse as ap

if __name__ == "__main__":
    
    # build arguments
    parser = ap.ArgumentParser(description="Convert lidar to BEV, support only .las for the moment.")
    parser.add_argument("target", help="The path of the target file.", type=str)
    args = parser.parse_args()
    
    # take parameters
    target_path = args.target
    
    # read data
    print("> Reading data from :", target_path)
    las = read_header(target_path)
    print(">> Point of Data Format:", las)
    #print(np.unique(las.classification))
    print(">> min: x,y,z", np.min(las.x), np.min(las.y), np.min(las.z))
    print(">> max: x,y,z", np.max(las.x), np.max(las.y), np.max(las.z))

    print(">> min: x,y,z", np.min(las.x), np.min(las.y), np.min(las.z))
    print(">> max: x,y,z", np.max(las.x), np.max(las.y), np.max(las.z))

    nb_points = las.header.point_count
    print(">> Number of points:", nb_points)

    point_format = las.point_format
    print(">> Point format (las point format):", point_format.id)
    print(">> Dimension names:", list(point_format.dimension_names))

    # 
