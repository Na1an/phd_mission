import argparse as ap
from model import *
from unet import *
from trainer import *
from build_dataset import *
from preprocess_data import *

if __name__ == "__main__":

    # start
    time_start = datetime.now()
    print("\n###### start the programme : {} ######\n".format(time_start.strftime("%Y-%m-%d %H:%M:%S")))
    
    # build arguments
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, let's figure it out! --")
    parser.add_argument("train_data_path", help="The path of raw data (train data with labels).", type=str)
    parser.add_argument("val_data_path", help="The path of raw data (val/test data with labels).", type=str)
    parser.add_argument("--grid_size", help="The sliding window size.", type=float, default=5.0)
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.2)
    parser.add_argument("--sample_size", help="The sample size : number of points in one-time training.", type=int, default=5000)
    parser.add_argument("--nb_epoch", help="The epoch number.", type=int, default=300)
    parser.add_argument("--batch_size", help="The batch size.", type=int, default=4)
    parser.add_argument("--predict_threshold", help="The predict threshold.", type=float, default=0.5)
    parser.add_argument("--global_height", help="The global_height.", type=int, default=50)
    parser.add_argument("--nb_window", help="int(nb_window**0.5) is the number of cuboids we want on the training set.", type=int, default=400)
    parser.add_argument("--augmentation", help="if we do the augmentation or not.", type=bool, default=False)
    parser.add_argument("--resolution", help="resolution of data", type=int, default=25)
    
    args = parser.parse_args()

    # take arguments
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    grid_size = args.grid_size
    voxel_size = args.voxel_size
    sample_size = args.sample_size
    nb_epoch = args.nb_epoch
    predict_threshold = args.predict_threshold
    batch_size = args.batch_size
    global_height = args.global_height
    resolution = args.resolution
    augmentation = False

    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    # (2) prepare train dataset and validation dataset
    # traindataset
    samples_train = read_data_from_directory(
                                train_data_path, 
                                resolution, 
                                voxel_sample_mode, 
                                label_name="WL", 
                                sample_size=sample_size, 
                                augmentation=augmentation)
    train_dataset = TrainDataSet(samples_train, 0, 0, my_device)
    train_dataset.show_info()
    
    # validation dataset
    samples_val = read_data_from_directory(
                                val_data_path, 
                                resolution, 
                                voxel_sample_mode, 
                                label_name="WL", 
                                sample_size=sample_size, 
                                augmentation=augmentation)
    val_dataset = TrainDataSet(samples_val, 0, 0, my_device)
    val_dataset.show_info()

    # (3) create model and trainning
    # create a model
    #global_height = z_max - z_min # the absolute height, set to 50 for the moment
    my_model = PointWiseModel(device=my_device)
    #my_model = UNet(dim=3)
    
    my_trainer = Trainer(
                my_model, 
                my_device, 
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                sample_size=sample_size,
                predict_threshold=predict_threshold,
                num_workers=0,
                grid_size=grid_size,
                global_height=global_height,
                )

    my_trainer.train_model(nb_epoch=nb_epoch)
    
    # end 
    time_end = datetime.now()
    time_diff = (time_end - time_start).total_seconds()
    print(time_diff)
    print("\n###### End {} ######".format(time_end.strftime("%Y-%m-%d %H:%M:%S")))
    print("### Energy Consumption : {} J - {} kWh ###".format(round(260*time_diff,2), round((260*time_diff/3600000),2)))

        
