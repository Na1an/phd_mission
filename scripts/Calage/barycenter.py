import argparse as ap
from sklearn.cluster import DBSCAN
from utility import *
from process_data import *

# calculate barycenter distance of two cluster
def barycenter_dist(points_t, points_d):
    center_t = np.mean(points_t[:,0:3], axis=0)
    center_d = np.mean(points_d[:,0:3], axis=0)
    dist = np.linalg.norm(center_t-center_d)
    print("> distance ={}".format(dist))
    return dist

if __name__ == "__main__":
    print("\n###### start to calculate barycenter distance ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("tls_path", help="The path of TLS data.", type=str)
    parser.add_argument("dls_path", help="The path of DLS data.", type=str)
    parser.add_argument("--eps", help="eps.", type=float)
    parser.add_argument("--w_data", help="write data or not.", type=bool, default=False)
    parser.add_argument("--d_barycenter", help="calculate distance of barycenter or not.", type=bool, default=False)
    args = parser.parse_args()
    tls_path = args.tls_path
    dls_path = args.dls_path
    eps = args.eps
    w_data = args.w_data
    d_barycenter = args.d_barycenter

    print("> input data tls_path:", tls_path)
    tls_data_processed, x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t = read_data(tls_path, detail=True)
    print("> tls_data_preprocess.shape =", tls_data_processed.shape, '\n')
    n_t, l_t = tls_data_processed.shape

    print("> input data dls_path:", dls_path)
    dls_data_processed, x_min_d, x_max_d, y_min_d, y_max_d, z_min_d, z_max_d = read_data(dls_path, detail=True)
    print("> dls_data_preprocess.shape =", dls_data_processed.shape, '\n')
    n_d, l_d = dls_data_processed.shape

    cluster_tls = DBSCAN(eps=eps, min_samples=10, n_jobs=8).fit(tls_data_processed).labels_
    #visualize_voxel_key_points(tls_data_processed, cluster_tls, "test_tls")
    print("> cluster tls shape=", cluster_tls.shape)

    cluster_dls = DBSCAN(eps=eps, min_samples=10, n_jobs=8).fit(dls_data_processed).labels_
    #visualize_voxel_key_points(dls_data_processed, cluster_dls, "test_dls")
    print("> cluster dls shape=", cluster_dls.shape)

    if w_data:
        x_min_overlap = np.min(tls_data_processed[:,0])
        y_min_overlap = np.min(tls_data_processed[:,1])
        write_data_bis(np.concatenate((tls_data_processed, np.expand_dims(cluster_tls, axis=0).T), axis=1), "tls_with_cluster", x_min_overlap, y_min_overlap)
        write_data_bis(np.concatenate((dls_data_processed, np.expand_dims(cluster_dls, axis=0).T), axis=1), "dls_with_cluster", x_min_overlap, y_min_overlap)
        
    if d_barycenter:
        tls_with_label = np.concatenate((tls_data_processed, np.expand_dims(cluster_tls, axis=0).T), axis=1)
        dls_with_label = np.concatenate((dls_data_processed, np.expand_dims(cluster_dls, axis=0).T), axis=1)
        # cluster : TLS-8 DLS-11
        barycenter_dist(tls_with_label[np.where(tls_with_label[:,3] == 5)], dls_with_label[np.where(dls_with_label[:,3] == 3)])
        # cluster : TLS-9 DLS-12
        barycenter_dist(tls_with_label[np.where(tls_with_label[:,3] == 8)], dls_with_label[np.where(dls_with_label[:,3] == 6)])
        # cluster : TLS-10 DLS-14
        barycenter_dist(tls_with_label[np.where(tls_with_label[:,3] == 11)], dls_with_label[np.where(dls_with_label[:,3] == 10)])
        