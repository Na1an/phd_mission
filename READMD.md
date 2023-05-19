# SOUL (Semantic segmentation On ULs)



### 1. Installation

---------

The code will be published on github later.

#### 1.1 System

Normally, the code is able to be executed on all linux-like system. (Or, only python and a few packages are need.)

#### 1.2 Create env on conda

You will need to install all packages in the requirements.txt file.  Executing the routine requires```miniconda``` (```Anaconda``` will certainly work). 

```
cd path-to-code/phd_mission
conda env create -f phd_mission.yml
```

If you have any difficulties or find any bugs, please get in touch and I will try to help you get it going. Suggestions for improvements are greatly appreciated.

If you have an available Nvidia GPU, the code will use it automatically. If not, CPU will be used for training and prediction.



### 2. Lunch the code

---------------



#### 2.1 For training:

For training model, you need go to the directory ```path-to-code/phd_mission/src```, the ```main.py``` will be used for training. The useful parameters are explained below.

* ```
  python main.py path-to-train path-to-validation label
  ```

  * path-to-train: a string. The directory path to training data. For example, we have one training data inside the ```train``` directory, it looks like : ```path-to-train/train/training_example.las```
  * path-to-validation: a string. The directory path to validation data. Same to path-to-train.
  * label : a string. The feature name of your ground truth.
  * sample_size : an integer. The point number for one sample, the default value is 3000.
  * nb_epoch : an integer. The number of epoch you want to train ,default value is 300.
  * batch_size : an integer. The batch size, default value is 4.
  * voxel_size : a float. The size of voxelization for GVD, default value is 0.6.

Execution example

* ```
  python main.py "path-to-data/train" "path-to-data/validation" "WL" --sample_size=3000 --nb_epoch=10000 --batch_size=16 --voxel
  ```

**[*] Checkpoint** Be attention,  if you already have one checkpoint in side ```path-to-code/phd_mission/src/checkpoints```, the number of this archive is bigger than your setting (like ```--nb_epoch=10000``` set on execution example), the program will terminate. 



#### 2.2 For prediction

For prediction, you need go to the directory ```path-to-code/phd_mission/src```, the ```predict.py``` will be used for predicting. The useful parameters are explained below.

* ```shell
  python predict.py path-to-testfile path-to-checkpoint
  ```

  * path-to-testfile : a string. Path to the test file.
  * path-to-checkpoint : a string. path to the check point.
  * voxel_size : a float. The size of voxelization for GVD, default value is 0.6. Must be more consistent with training settings.
  * sample_size : an integer. The point number for one sample, the default value is 3000.
  * label : a string. The feature name of your ground truth.

* ```shell
  python predict.py "path-to-test/test.las" checkpoints/checkpoint_epoch_003161.pth --sample_size=3000 --label_name="Amplitude"
  ```



**[*] label_name** For prediction, this is not important. If you have ground truth for the test data. Leave the name here, or you can just leave one feature name here.



### 3. Output

--------

