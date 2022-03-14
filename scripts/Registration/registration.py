import argparse
from utility import *

if __name__ == "__main__":
    
    # start the programme
    print("\n############## compare DLS & TLS ##############\n")

    # process the arguments
    parser = argparse.ArgumentParser(description="Convert lidar to BEV, support only .las for the moment.")
    parser.add_argument("target", help="DLS", type=str)
    parser.add_argument("tls", help="TLS", type=str)
    parser.add_argument("--resolution", help="This is the resolution we want", type=float, default=1.0) 
    args = parser.parse_args()
    target_path = args.target
    tls_path = args.tls
    resolution = args.resolution

    # read data
    las = read_header(target_path)
    points_dls = np.matrix([las.x,las.y,las.z])
    d_range_x = (np.floor(np.min(las.x)), np.ceil(np.max(las.x)))
    d_range_y = (np.floor(np.min(las.y)), np.ceil(np.max(las.y)))
    d_range_z = (np.floor(np.min(las.z)), np.ceil(np.max(las.z)))

    tls = read_header(tls_path)
    points_tls = np.matrix([tls.x,tls.y,tls.z])
    t_range_x = (np.floor(np.min(tls.x)), np.ceil(np.max(tls.x)))
    t_range_y = (np.floor(np.min(tls.y)), np.ceil(np.max(tls.y)))
    t_range_z = (np.floor(np.min(tls.z)), np.ceil(np.max(tls.z)))

    # common range z-axis
    com_range_x = (min(np.floor(np.min(tls.x)), np.floor(np.min(las.x))), max(np.ceil(np.max(tls.x)), np.ceil(np.max(las.x))))
    com_range_y = (min(np.floor(np.min(tls.y)), np.floor(np.min(las.y))), max(np.ceil(np.max(tls.y)), np.ceil(np.max(las.y))))
    com_range_z = (min(np.floor(np.min(tls.z)), np.floor(np.min(las.z))), max(np.ceil(np.max(tls.z)), np.ceil(np.max(las.z))))
    
    # convert lidar to BEV
    im_dls = point_cloud_2_birdseye(points_dls.T, res=resolution, y_range=t_range_y, x_range=t_range_x, z_range=com_range_z)
    print(">> image.shape:", im_dls.shape)
    
    im_tls = point_cloud_2_birdseye(points_tls.T, res=resolution, y_range=t_range_y, x_range=t_range_x, z_range=com_range_z)
    print(">> image.shape:", im_tls.shape)

    fig, axs = plt.subplots(1,3)
    axs[0].set_title('target1')
    pos0 = axs[0].imshow(im_dls, cmap="nipy_spectral", vmin=0, vmax=255, origin='lower')
    fig.colorbar(pos0, ax=axs[0], fraction=0.046, pad=0.04)
    
    axs[1].set_title('target2')
    pos1 = axs[1].imshow(im_tls, cmap="nipy_spectral", vmin=0, vmax=255, origin='lower')
    fig.colorbar(pos1, ax=axs[1], fraction=0.046, pad=0.04)
    axs[2].set_title('target2-target1')

    dif = (im_tls-im_dls)
    #im_dif = dif + np.abs(np.min(dif))
    im_dif = (dif + np.abs(np.min(dif)))/(np.max(dif) - np.min(dif))
    print(">>> vmin=", np.min(im_dif), "vmax=", np.max(im_dif))
    pos2 = axs[2].imshow(im_dif, cmap="nipy_spectral", vmin=np.min(im_dif), vmax=np.max(im_dif), origin='lower')
    fig.colorbar(pos2, ax=axs[2], fraction=0.046, pad=0.04)

    #plt.imshow(im_dls, cmap="nipy_spectral", vmin=0, vmax=255, origin='lower')
    #plt.imshow(im_tls, cmap="nipy_spectral", vmin=0, vmax=255, origin='lower')
    plt.show()
    
    # end the programme
    print("\n########### End : Convert .las to BEV ###########\n")    

