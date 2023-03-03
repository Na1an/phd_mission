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
    parser.add_argument("--sample_size", help="The sample size : number of points in one-time training.", type=int, default=5000)
    parser.add_argument("--global_height", help="The global height.", type=int, default=50)
    parser.add_argument("--resolution", help="resolution of data", type=int, default=20)

    args = parser.parse_args()

    # take arguments
    data_path = args.data_path
    cp_path = args.checkpoint_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    sample_size = args.sample_size
    global_height = args.global_height
    resolution = args.resolution
    
    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))
    
    # (2) prepare train dataset and validation dataset
    
    samples_test, sample_voxel_net_index_test, test_voxel_nets, sample_position, x_min_all, y_min_all, z_min_all, samples_rest = prepare_procedure_ier(
                                                    data_path, 
                                                    resolution,
                                                    voxel_sample_mode, 
                                                    label_name="WL",
                                                    sample_size=sample_size,
                                                    augmentation=False,
                                                    for_test=True)
    test_dataset = TestDataSet(samples_test, sample_voxel_net_index_test, test_voxel_nets, my_device, sample_position, samples_rest)
    #test_dataset.show_info()

    # (3) predict
    my_model = PointWiseModel(device=my_device)
    checkpoint = torch.load(cp_path, map_location=torch.device('cpu'))
    my_model.load_state_dict(checkpoint['model_state_dict'])
    my_model.eval()

    print("dataset len =", test_dataset.__len__())
    # batch_size must be 1!!!
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_voxel_nets = torch.from_numpy(test_voxel_nets.copy()).type(torch.float).to(my_device)
    
    i = 0
    #predict label possibilty
    
    for points, pointwise_features, labels, voxel_net, sp, samples_rest_single, points_raw in test_loader:
        points_for_pointnet = torch.cat([points.transpose(2,1), pointwise_features.transpose(2,1), points.transpose(2,1)], dim=1)
        #points_for_pointnet[:,0:2,:] = points_for_pointnet[:,0:2,:] * self.grid_size
        #points_for_pointnet[:,2,:] = points_for_pointnet[:,2,:] * self.global_height + (self.global_height/2)
        points_for_pointnet[:,6:,:] = points_for_pointnet[:,6:,:] + 0.5
        #print("pointsfor_pointnet.shape={}".format(points_for_pointnet.shape))
        logits = my_model(points, pointwise_features, voxel_net, points_for_pointnet)
        logits = F.softmax(logits, dim=1)
        predict = logits.squeeze(0).float()
        predict_label = logits.argmax(dim=1).float()
        #print("predict.shape", predict.shape)
        #print("predict_label.shape", predict_label.shape)
        [x_min, y_min, z_min, max_axe, max_x_axe, max_y_axe, max_z_axe] = sp[0]
        x_min, y_min, z_min, max_axe, max_x_axe, max_y_axe, max_z_axe = x_min.numpy(), y_min.numpy(), z_min.numpy(), max_axe.numpy(), max_x_axe.numpy(), max_y_axe.numpy(), max_z_axe.numpy()
        #print("x_min, y_min, z_min, max_axe, max_x_axe, max_y_axe, max_z_axe".format(local_x, local_y, local_z, adjust_x, adjust_y, adjust_z))
        new_file = laspy.create(point_format=3)
        new_file.add_extra_dim(laspy.ExtraBytesParams(name="wood_proba", type=np.float64))
        new_file.add_extra_dim(laspy.ExtraBytesParams(name="leave_proba", type=np.float64))
        new_file.add_extra_dim(laspy.ExtraBytesParams(name="predict", type=np.float64))
        new_file.add_extra_dim(laspy.ExtraBytesParams(name="true", type=np.float64))
        points = points.squeeze(0).cpu().detach().numpy()
        labels = labels.squeeze(0).cpu().detach().numpy()
        points_raw = points_raw.squeeze(0).cpu().detach().numpy()
        
        # + 0.5 : react to centered to (0,0,0)
        # - (1 - np.max(points[:,0])) : react to cube centering
        # * max_axe : react to scaling
        # + x_min : find global position
        new_file.x = points_raw[:,0] + x_min + x_min_all
        new_file.y = points_raw[:,1] + y_min + y_min_all
        new_file.z = points_raw[:,2] + z_min + z_min_all

        new_file.wood_proba = predict[0,:].cpu().detach().numpy()
        new_file.leave_proba = predict[1,:].cpu().detach().numpy()
        new_file.predict = predict_label.cpu().detach().numpy()
        new_file.true = labels
        new_file.write(os.getcwd()+"/predict_res/res_{:04}.las".format(i))

        # sample rest
        new_file_bis = laspy.create(point_format=3)
        new_file_bis.add_extra_dim(laspy.ExtraBytesParams(name="wood_proba", type=np.float64))
        new_file_bis.add_extra_dim(laspy.ExtraBytesParams(name="leave_proba", type=np.float64))
        new_file_bis.add_extra_dim(laspy.ExtraBytesParams(name="predict", type=np.float64))
        new_file_bis.add_extra_dim(laspy.ExtraBytesParams(name="true", type=np.float64))
        if len(samples_rest_single) ==0:
            continue
        samples_rest_single = samples_rest_single.squeeze(0).cpu().detach().numpy()
        samples_rest_single = samples_rest_single[:,:4]
        
        # + 0.5 : react to centered to (0,0,0)
        # - (1 - np.max(points[:,0])) : react to cube centering
        # * max_axe : react to scaling
        # + x_min : find global position
        new_file_bis.x = samples_rest_single[:,0] + x_min + x_min_all
        new_file_bis.y = samples_rest_single[:,1] + y_min + y_min_all
        new_file_bis.z = samples_rest_single[:,2] + z_min + z_min_all
        
        new_file_bis.wood_proba = -1 * np.ones(len(samples_rest_single[:]))
        new_file_bis.leave_proba = -1 * np.ones(len(samples_rest_single[:]))
        # wood label=0, leaf label=1
        new_file_bis.predict = np.ones(len(samples_rest_single[:]))
        new_file_bis.true = samples_rest_single[:,3]
        new_file_bis.write(os.getcwd()+"/predict_res/rest_{:04}.las".format(i))

        i = i+1
        print(">>> cube - N°{} predicted ".format(i))
    
    print("\n###### End ######")

        
