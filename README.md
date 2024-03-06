# SOUL (Semantic segmentation On ULs)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-segmentation-of-sparse-irregular/lidar-semantic-segmentation-on-uls-labeled)](https://paperswithcode.com/sota/lidar-semantic-segmentation-on-uls-labeled?p=semantic-segmentation-of-sparse-irregular)

> [Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination](https://neurips.cc/virtual/2023/poster/72647)
>
> [Yuchen BAI](https://na1an.github.io/), [Jean-Baptiste Durand](https://amap.cirad.fr/fr/edit-article.php?id=433), [Florence Forbes](https://mistis.inrialpes.fr/people/forbes/), [Grégoire Vincent](https://lab.ird.fr/personne/eHBCWllvN3FjJTJGQVNWTzhlRkdwZ2FoTDMlMkIlMkJRS3BCdUR1eDZjdGtwZzlqRSUzRA/show)
>
>  Université Grenoble-Alpes, Inria, AMAP

*A more detailed README.md will be released soon.*

#### News

--------

**[2023-10-02]** The dataset used in the article is open access now: https://zenodo.org/record/8398853

**[2023-09-22]** This paper is accepted by **NeurIPS 2023**! :tada:



### 1. Installation

---------

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

#### 1.3 Code and Folder

* ```main.py``` is for training.

* ```predict.py``` is for prediction.

* ```data``` has a small test dataset.

* ```checkpoints``` holds the model parameters. (3161th epoch in this case.)

* ```predict_res``` will keep the prediction result.

  

### 2. Usage

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

* ```sh
  python main.py "path-to-data/train" "path-to-data/validation" "WL" --sample_size=3000 --nb_epoch=10000 --batch_size=16 --voxel_size=0.6
  ```

**[*] Checkpoint** Be attention,  if you already have one checkpoint in side ```path-to-code/phd_mission/src/checkpoints```, the number of this archive is bigger than your setting (like ```--nb_epoch=10000``` set on execution example), the program will terminate. 



#### 2.2 For prediction

For prediction, you need go to the directory ```path-to-code/phd_mission/src```, the ```predict.py``` will be used for predicting. The useful parameters are explained below.

* ```sh
  python predict.py path-to-testfile path-to-checkpoint
  ```

  * path-to-testfile : a string. Path to the test file.
  * path-to-checkpoint : a string. path to the check point.
  * voxel_size : a float. The size of voxelization for GVD, default value is 0.6. Must be more consistent with training settings.
  * sample_size : an integer. The point number for one sample, the default value is 3000.
  * label : a string. The feature name of your ground truth.

* ```sh
  python predict.py ./data/dls_merged_test.las --sample_size=3000 --label_name="WL" checkpoints/checkpoint_epoch_003161.pth
  ```

**[*] --label_name** For prediction, this is not important. If you have ground truth for the test data. Leave the name here, or you can just leave one feature name here.

* The data will be processed as ```data_las = data_las[data_las[label_name]>0]```, So if you have missing points, maybe check your input label name? ```The default version have commented this line.```

* Inside the ```predict.py```, you need to pay attentions on the **point_format**. The float type (e.g. **float32** or **float64**) is crucial to coordinate precision. You can check the doc [here](https://laspy.readthedocs.io/en/latest/intro.html#point-format-6).

**[*] ** ```tls_mode``` is activated par default.


