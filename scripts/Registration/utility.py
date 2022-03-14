import laspy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_header(path):
    '''
    Args:
        path : a string. The path of file.
    Returns:
        The file's header and data. (and VLRS if it has)
    '''
    print("\n> Reading data from :", path)

    las = laspy.read(path)
    print(">> Point of Data Format:", las)
    print(">> min: x,y,z", np.min(las.x), np.min(las.y), np.min(las.z))
    print(">> max: x,y,z", np.max(las.x), np.max(las.y), np.max(las.z))

    nb_points = las.header.point_count
    print(">> Number of points:", nb_points)
    
    point_format = las.point_format
    print(">> LAS File Format:", point_format.id)
    print(">> Dimension names:", list(point_format.dimension_names))

    return laspy.read(path)

def scale_to_255(a, min, max, dtype=np.int64):
    """ 
    Scales an array of values from specified min, max range to 0-255
    Optionally specify the data type of the output (default is uint8)
    Args:
        a : a float. The value
        min : a float. Minimum of the range.
        max : a float. Maximum of the range.
        dtype : numpy's type.
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def point_cloud_2_birdseye(points,
                           res=0.1,
                           y_range=(0., 10.),  
                           x_range = (0., 10.), 
                           z_range=(0., 2.),
                           dtype=np.int64
                           ):
    """
    Creates an 2D birds eye view representation of the point cloud data.
    
    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        y_range: (tuple of two floats), y-axis range.
        x_range: (tuple of two floats), x-axis range.
        z_range: (tuple of two floats), z-axis range.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # Extract points from each axis
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # Filter - find desired indices
    f_filt = np.logical_and((x_points > x_range[0]), (x_points < x_range[1]))
    s_filt = np.logical_and((y_points > y_range[0]), (y_points < y_range[1]))
    indices = np.where(f_filt & s_filt)

    # Only take useful data/points
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    print(">>> NB of points in the range :", x_points.shape)
    print("x_points =", x_points[0:10])
    print("y_points =", y_points[0:10])
    print("z_points =", z_points[0:10])

    # Convert to the pixels positions, based on resolution
    x_img = (x_points / res).astype(dtype)
    y_img = (y_points / res).astype(dtype)
    print(y_img)
    if np.max(y_img) >= y_range[1]:
        Exception("Erreur, filter doesn't work")
    
    print(">> before rescale")
    print("x_img min:", np.min(x_img))
    print("x_img max:", np.max(x_img))
    print("y_img min:", np.min(y_img))
    print("y_img max:", np.max(y_img))

   
    # Shift pixels to lower-base, (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(x_range[0] / res))
    y_img -= int(np.floor(y_range[0] / res))

    print(">> after rescale")
    print("x_img min:", np.min(x_img))
    print("x_img max:", np.max(x_img))
    print("y_img min:", np.min(y_img))
    print("y_img max:", np.max(y_img))

    # Clip number of points 
    pixel_values = np.clip(a=z_points, a_min=z_range[0], a_max=z_range[1])

    # Rescale number of points between the range 0-255
    pixel_values = scale_to_255(pixel_values, min=0, max=np.max(pixel_values))

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int(np.ceil((x_range[1] - x_range[0]) / res)) + 1
    y_max = int(np.ceil((y_range[1] - y_range[0]) / res)) + 1
    im = np.zeros([y_max, x_max], dtype=dtype)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im
