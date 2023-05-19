import argparse as ap
import torch.nn.functional as F
from model import *
from trainer import *
from build_dataset import *
from preprocess_data import *

if __name__ == "__main__":
    
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (train data with labels).", type=str)
    parser.add_argument("checkpoint_path", help="The path of checkpoint (train data with labels).", type=str)
    parser.add_argument("--grid_size", help="The sliding window size.", type=float, default=5.0)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--sample_size", help="The sample size : number of points in one-time training.", type=int, default=3000)
    parser.add_argument("--global_height", help="The global height.", type=int, default=50)
    parser.add_argument("--resolution", help="resolution of data", type=int, default=25)
    parser.add_argument("--label_name", help="WL if the test dataset has label, intensity or something else if not", type=str, default="intensity")

    args = parser.parse_args()

    # take arguments
    data_path = args.data_path
    cp_path = args.checkpoint_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    sample_size = args.sample_size
    global_height = args.global_height
    resolution = args.resolution
    label_name = args.label_name

    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))
    
    # (2) prepare train dataset and validation dataset
    
    test_dataset, sample_voxel_net_index_test, test_voxel_nets, sample_position, x_min_all, y_min_all, z_min_all = prepare_procedure_ier(
                                                    data_path, 
                                                    resolution,
                                                    voxel_sample_mode, 
                                                    label_name=label_name,
                                                    sample_size=sample_size,
                                                    augmentation=False,
                                                    for_test=True,
                                                    voxel_size_ier=0.6,
                                                    limit_comp=1, 
                                                    limit_p_in_comp=5,
                                                    tls_mode=False)
    # samples, sample_voxel_net_index, device, sample_position,
    test_dataset = TestDataSet(test_dataset, sample_voxel_net_index_test, my_device, sample_position, num_classes=2)
    test_dataset.show_info()

    # (3) predict
    my_model = PointWiseModel(device=my_device)
    checkpoint = torch.load(cp_path, map_location=torch.device('cpu'))
    my_model.load_state_dict(checkpoint['model_state_dict'])
    my_model.to(my_device)
    my_model.eval()

    test_dataset_len = test_dataset.__len__()
    print("dataset len =", test_dataset_len)
    # batch_size = 1 for prediction
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    #
    data_test_all = []
    i = 0
    #predict label possibilty    
    for points, pointwise_features, labels, sp, points_raw, gd, id_comp in test_loader:
        points_for_pointnet = torch.cat([points.transpose(2,1), pointwise_features.transpose(2,1)], dim=1)
        points_for_pointnet = points_for_pointnet.float()
        
        points, pointwise_features, points_for_pointnet = points.to(my_device), pointwise_features.to(my_device), points_for_pointnet.to(my_device)
        logits = my_model(points_for_pointnet)
        logits = F.softmax(logits, dim=1)
        predict = logits.squeeze(0).float()
        predict_label = logits.argmax(dim=1).float()
        
        # sp: is the sample position
        [x_min, y_min, z_min, max_axe, max_x_axe, max_y_axe, max_z_axe] = sp[0]
        x_min, y_min, z_min, max_axe, max_x_axe, max_y_axe, max_z_axe = x_min.numpy(), y_min.numpy(), z_min.numpy(), max_axe.numpy(), max_x_axe.numpy(), max_y_axe.numpy(), max_z_axe.numpy()
        #print("x_min, y_min, z_min, max_axe, max_x_axe, max_y_axe, max_z_axe".format(local_x, local_y, local_z, adjust_x, adjust_y, adjust_z))
        
        labels = labels.squeeze(0).cpu().detach().numpy()
        points_raw = points_raw.squeeze(0).cpu().detach().numpy()
        gd = gd.squeeze(0).cpu().detach().numpy()
        id_comp = id_comp.squeeze(0).cpu().detach().numpy()
        pointwise_features = pointwise_features.squeeze(0).cpu().detach().numpy()
        predict = predict.cpu().detach().numpy()
        predict = np.transpose(predict, (1,0)) 
        predict_label = logits.argmax(dim=1).float().cpu().detach().numpy()
        predict_label = np.transpose(predict_label, (1,0)) 

        points_raw[:,0] = points_raw[:,0] + x_min + x_min_all
        points_raw[:,1] = points_raw[:,1] + y_min + y_min_all
        points_raw[:,2] = points_raw[:,2] + z_min + z_min_all

        data_test = np.concatenate((points_raw, labels.reshape(-1,1), pointwise_features, id_comp.reshape(-1,1), gd.reshape(-1,1), predict, predict_label), axis=1)
        data_test_all.append(data_test)
        
        i = i+1
        print(">>> cube - N°{}/{} predicted".format(i+1, test_dataset_len), end="\t\r")

    data_test_all = np.array(data_test_all, dtype='object')
    nb_sample, sample_size, nb_f = data_test_all.shape
    data_test_all = data_test_all.reshape(nb_sample*sample_size, nb_f)
    print("data_test_all.shape={}".format(data_test_all.shape))
    data_test_all = np.array(data_test_all, dtype=np.float32)

    new_file = laspy.create(point_format=3)
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="wood_proba", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="leave_proba", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="predict", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="true", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="gd", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="id_comp", type=np.float64))
    
    # you can access computed features here
    '''
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="PCA1_30", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="linearity_30", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="sphericity_30", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="verticality_30", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="PCA1_60", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="linearity_60", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="sphericity_60", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="verticality_60", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="PCA1_90", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="linearity_90", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="sphericity_90", type=np.float64))
    new_file.add_extra_dim(laspy.ExtraBytesParams(name="verticality_90", type=np.float64))
    '''

    new_file.x = data_test_all[:,0]
    new_file.y = data_test_all[:,1]
    new_file.z = data_test_all[:,2]
    new_file.true = data_test_all[:,3]

    new_file.id_comp = data_test_all[:,-5]
    new_file.gd = data_test_all[:,-4]
    new_file.wood_proba = data_test_all[:,-3]
    new_file.leave_proba = data_test_all[:,-2]
    new_file.predict = data_test_all[:,-1]
    '''
    new_file.PCA1_30 = data_test_all[:,4]
    new_file.linearity_30 = data_test_all[:,5]
    new_file.sphericity_30 = data_test_all[:,6]
    new_file.verticality_30 = data_test_all[:,7]
    new_file.PCA1_60 = data_test_all[:,8]
    new_file.linearity_60 = data_test_all[:,9]
    new_file.sphericity_60 = data_test_all[:,10]
    new_file.verticality_60 = data_test_all[:,11]
    new_file.PCA1_90 = data_test_all[:,12]
    new_file.linearity_90 = data_test_all[:,13]
    new_file.sphericity_90 = data_test_all[:,14]
    new_file.verticality_90 = data_test_all[:,15]
    '''

    time_end = datetime.now()
    new_file.write(os.getcwd()+"/predict_res/res_{}.las".format(time_end.strftime("%Y-%m-%d %H:%M:%S")))
    
    print("\n###### End ######")
        
