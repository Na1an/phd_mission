import argparse as ap
from model import *
from trainer import *
from build_dataset import *
from preprocess_data import *

if __name__ == "__main__":
    print("\n###### start the programme ######\n")
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

    # set by default
    voxel_sample_mode = 'mc'

    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    
    # (2) prepare train dataset and validation dataset
    samples_train, sample_cuboid_index_train, train_voxel_nets = prepare_procedure(train_data_path, grid_size, voxel_size, voxel_sample_mode, sample_size, global_height=global_height, label_name="llabel", detail=True)
    train_dataset = TrainDataSet(samples_train, sample_cuboid_index_train, my_device)
    train_dataset.show_info()
    
    samples_val, sample_cuboid_index_val, val_voxel_nets = prepare_procedure(val_data_path, grid_size, voxel_size, voxel_sample_mode, sample_size, global_height=global_height, label_name="WL", detail=True)
    val_dataset = TrainDataSet(samples_val, sample_cuboid_index_val, my_device)
    val_dataset.show_info()

    # (3) create model and trainning

    # create a model
    #global_height = z_max - z_min # the absolute height, set to 50 for the moment
    my_model = PointWiseModel(device=my_device)
    
    my_trainer = Trainer(
                my_model, 
                my_device, 
                train_dataset=train_dataset,
                train_voxel_nets=train_voxel_nets,
                val_dataset=val_dataset,
                val_voxel_nets=val_voxel_nets,
                batch_size=batch_size,
                sample_size=sample_size,
                predict_threshold=predict_threshold,
                num_workers=0)

    my_trainer.train_model(nb_epoch=nb_epoch)
    
    print("\n###### End ######")

        
