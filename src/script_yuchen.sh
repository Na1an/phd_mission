python --version
pwd
source /services/scratch/mistis/yuchen/miniconda3/bin/activate phd_mission 
python --version
python main.py ../data/dls_new_train.las ../data/dls_new_val.las --grid_size=3 --voxel_size=0.1 --sample_size=5000 --nb_epoch=600 --batch_size=4 --global_height=50
