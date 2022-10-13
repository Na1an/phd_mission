import numpy as np

# return the set of sliding window coordinates
def sliding_window(x_min, x_max, y_min, y_max, grid_size, nb_window=500):
    '''
    Args:
        *_min/*_max : a interger. The data range.
        grid_size : a interger/float. The side length of the grid.
    Returns:
        res : a list of 2-d coordinates (x,y). The set of grid coordinates.
    '''

    coord_x = np.random.uniform(x_min, x_max-grid_size, nb_window)
    coord_y = np.random.uniform(y_min, y_max-grid_size, nb_window)
    mesh_x, mesh_y = np.meshgrid(coord_x, coord_y)

    return np.stack((mesh_x,mesh_y), 2)

res = sliding_window(0,10,0,10,5,2)

print("shape ={}, res ={}".format(res.shape, res))
