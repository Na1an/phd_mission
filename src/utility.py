import os
import laspy
import open3d 
import numpy as np
import argparse as ap

def read_header(path):
    '''
    Args:
        path : a string. The path of file.
    Returns:
        The file's header and data. (and VLRS if it has)
    '''
    return laspy.read(path)

def print_info(las):
    '''
    Args:
        las : a LasData.
    Returns:
        None.
    '''
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

    return None
