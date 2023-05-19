import argparse as ap
from model import *
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
    parser.add_argument("label", help="The ground truth.", type=str, default="WL")
    parser.add_argument("--voxel_size", help="The voxel size.", type=float, default=0.6)
    parser.add_argument("--sample_size", help="The sample size : number of points in one-time training.", type=int, default=5000)
    parser.add_argument("--nb_epoch", help="The epoch number.", type=int, default=300)
    parser.add_argument("--batch_size", help="The batch size.", type=int, default=4)
    parser.add_argument("--predict_threshold", help="The predict threshold.", type=float, default=0.5)
    
    args = parser.parse_args()

    # take arguments
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    label_name = args.label
    voxel_size = args.voxel_size
    sample_size = args.sample_size
    nb_epoch = args.nb_epoch
    predict_threshold = args.predict_threshold
    batch_size = args.batch_size
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
    train_dataset = read_data_from_directory(
                                train_data_path, 
                                voxel_sample_mode, 
                                voxel_size_ier=voxel_size,
                                label_name=label_name, 
                                sample_size=sample_size, 
                                augmentation=augmentation)
    train_dataset = TrainDataSet(train_dataset, 0, my_device)
    train_dataset.show_info()
    
    # validation dataset
    val_dataset = read_data_from_directory(
                                val_data_path, 
                                voxel_sample_mode, 
                                voxel_size_ier=voxel_size,
                                label_name=label_name, 
                                sample_size=sample_size, 
                                augmentation=augmentation)
    val_dataset = TrainDataSet(val_dataset, 0, my_device)
    val_dataset.show_info()

    # (3) create model and trainning
    # create a model
    my_model = PointWiseModel(device=my_device)
    
    my_trainer = Trainer(
                my_model, 
                my_device, 
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                sample_size=sample_size,
                predict_threshold=predict_threshold,
                num_workers=0,
                )

    my_trainer.train_model(nb_epoch=nb_epoch)
    
    # end 
    time_end = datetime.now()
    time_diff = (time_end - time_start).total_seconds()
    print(time_diff)
    print("\n###### End {} ######".format(time_end.strftime("%Y-%m-%d %H:%M:%S")))
    print("### Energy Consumption : {} J - {} kWh ###".format(round(260*time_diff,2), round((260*time_diff/3600000),2)))

        
