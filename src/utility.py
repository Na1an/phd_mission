import torch
import laspy
import numpy as np

# print info of the laspy data
def get_info(las):
    '''
    Args:
        las : a LasData.
    Returns:
        x_min, x_max, y_min, y_max : a region.
    '''
    x_min, y_min, z_min = np.min(las.x), np.min(las.y), np.min(las.z)
    x_max, y_max, z_max = np.max(las.x), np.max(las.y), np.max(las.z)

    print("\n>> Point of Data Format:", las)
    #print(np.unique(las.classification))
    print(">> min: x,y,z", x_min, y_min, z_min)
    print(">> max: x,y,z", x_max, y_max, z_max)
    print(">> the data present a cube:", x_max-x_min, '*', y_max-y_min, '*', z_max-z_min)

    nb_points = las.header.point_count
    print(">> Number of points:", nb_points)

    point_format = las.point_format
    print(">> Point format (las point format):", point_format.id)
    print(">> Dimension names:", list(point_format.dimension_names), '\n')

    return x_min, x_max, y_min, y_max, z_min, z_max
