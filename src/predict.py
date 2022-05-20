import argparse as ap
from model import *
from trainer import *
from build_dataset import *
from preprocess_data import *

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("data_path", help="The path of raw data (train data with labels).", type=str)
    parser.add_argument("--grid_size", help="The sliding window size.", type=float, default=5.0)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--sample_size", help="The sample size : number of points in one-time training.", type=int, default=5000)
    parser.add_argument("--nb_epoch", help="The epoch number.", type=int, default=300)
    parser.add_argument("--predict_threshold", help="The predict threshold.", type=float, default=0.5)
    args = parser.parse_args()

    # take arguments
    data_path = args.data_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    sample_size = args.sample_size
    nb_epoch = args.nb_epoch
    predict_threshold = args.predict_threshold

    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    
    # (2) prepare train dataset and validation dataset
    samples_test, sample_cuboid_index_test, train_voxel_nets, sw = prepare_procedure_predict(data_path, grid_size, voxel_size, voxel_sample_mode, sample_size, label_name="llabel", detail=False, naif_sliding=True)
    test_dataset = TestDataSet(samples_test, sample_cuboid_index_test, device=my_device)
    test_dataset.show_info()
    
    # (3) predict
    my_model = PointWiseModel()
    checkpoint = torch.load("/home/yuchen/Documents/PhD/phd_mission/src/checkpoints/checkpoint_epoch_23.tar")
    my_model.load_state_dict(checkpoint['model_state_dict'])
    my_model.eval()
    las = read_header(data_path)
    print("dataset len =", test_dataset.__len__())
    # batch_size must be 1!!!
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_voxel_nets = torch.from_numpy(train_voxel_nets.copy()).type(torch.float).to(my_device)
    i = 0
    for points, v, index_sw in test_loader:
        logits = my_model(points, train_voxel_nets[v])
        predict = (logits>0.5).float()
        local_x, local_y, local_z, adjust_x, adjust_y, adjust_z = sw[int(index_sw[0])]
        
        new_file = laspy.create(point_format=las.point_format, file_version="1.2")
        points = points.squeeze(0).cpu().detach().numpy()
        new_file.x = points[:,0] + adjust_x + local_x
        new_file.y = points[:,1] + adjust_y + local_y
        new_file.z = points[:,2] + adjust_z + local_z

        las.points = new_file.points
        las['llabel'] = predict.cpu().detach().numpy()
        las.write(os.getcwd()+"/predict_res/res_{:04}.las".format(i))
        i = i+1
        print(">>> cube - N°{} predicted ".format(i))

    print("\n###### End ######")
        