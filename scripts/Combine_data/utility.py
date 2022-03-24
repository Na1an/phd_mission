import os
import laspy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_header(path, detail=True):
    '''
    Args:
        path : a string. The path of file.
        detail : a boolean. If we want to see the details.
    Returns:
        The file's header and data. (and VLRS if it has)
    '''
    print("\n> Reading data from :", path)

    las = laspy.read(path)
    nb_points = las.header.point_count
    point_format = las.point_format

    if detail:
        print(">> Point of Data Format:", las)
        print(">> min: x,y,z", np.min(las.x), np.min(las.y), np.min(las.z))
        print(">> max: x,y,z", np.max(las.x), np.max(las.y), np.max(las.z))
        print(">> Number of points:", nb_points)
        print(">> LAS File Format:", point_format.id)
        print(">> Dimension names:", list(point_format.dimension_names))

    return laspy.read(path)