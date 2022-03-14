import argparse
from utility import *

if __name__ == "__main__":
    
    # start the programme
    print("\n############## Convert .las/.laz to BEV ##############\n")

    # process the arguments
    parser = argparse.ArgumentParser(description="Convert lidar to BEV, support only .las for the moment.")
    parser.add_argument("target", help="The path of the target file.", type=str)
    parser.add_argument("--resolution", help="This is the resolution we want", type=float, default=1.0) 
    args = parser.parse_args()
    target_path = args.target
    resolution = args.resolution

    # read data
    print("> Reading data from :", target_path)
    las = read_header(target_path)
    print(">> Point of Data Format:", las)
    #print(np.unique(las.classification))
    print(">> min: x,y,z", np.min(las.x), np.min(las.y), np.min(las.z))
    print(">> max: x,y,z", np.max(las.x), np.max(las.y), np.max(las.z))

    nb_points = las.header.point_count
    print(">> Number of points:", nb_points)
    
    point_format = las.point_format
    print(">> Point format (las point format):", point_format.id)
    print(">> Dimension names:", list(point_format.dimension_names))
    
    points = np.matrix([las.x,las.y,las.z])
    print(">> points.shape:", points.shape)
    
    # convert lidar to BEV
    im = point_cloud_2_birdseye(points.T, res=resolution, y_range=(np.floor(np.min(las.y)), np.ceil(np.max(las.y))), x_range=(np.floor(np.min(las.x)), np.ceil(np.max(las.x))), z_range=(np.floor(np.min(las.z)), np.ceil(np.max(las.z))))
    print(">> image.shape:", im.shape)
    
    # plot the image
    plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255, origin='lower')
    plt.show()
    
    # end the programme
    print("\n########### End : Convert .las to BEV ###########\n")    

