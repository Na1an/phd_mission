# SOUL (Semantic segmentation On ULs)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-segmentation-of-sparse-irregular/lidar-semantic-segmentation-on-uls-labeled)](https://paperswithcode.com/sota/lidar-semantic-segmentation-on-uls-labeled?p=semantic-segmentation-of-sparse-irregular)

> [Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination](https://neurips.cc/virtual/2023/poster/72647)
>
> [Yuchen BAI](https://na1an.github.io/), [Jean-Baptiste Durand](https://amap.cirad.fr/fr/edit-article.php?id=433), [Florence Forbes](https://mistis.inrialpes.fr/people/forbes/), [Grégoire Vincent](https://lab.ird.fr/personne/eHBCWllvN3FjJTJGQVNWTzhlRkdwZ2FoTDMlMkIlMkJRS3BCdUR1eDZjdGtwZzlqRSUzRA/show)
>
>  Université Grenoble-Alpes, Inria, AMAP



#### News

--------

**[2025-05-10]** The docker image is open access.

**[2024-05-22]** The epoch 0 weights of SOUL are available.

**[2023-10-02]** The dataset used in the article is open access now: https://zenodo.org/record/8398853

**[2023-09-22]** This paper is accepted by **NeurIPS 2023**! :tada:

### 1. Installation via docker

-----

So, for the easy usage, we provide here a docker image for the fast deployment.

* The SOUL docker image is here: https://zenodo.org/records/15377222

* After install docker, you can simply run ```docker load -i soul_docker.tar ``` for load the docker image.

* Use ```docker run -it soul:latest``` to execute the Linux system inside docker.

* Then, simple run:

  ```
  cd home/phd_mission/
  conda activate phd_mission
  ```

* Normally, you can now run SOUL model inside the docker.

* You can use: 

  ```docker cp file_in_your_computer 7bf378620045:/path_in_docker/file_in_your_computer``` 

  or 

  ```docker cp file_in_your_computer 7bf378620045:/path_in_docker/file_in_your_computer``` 

  to copy/paste the file between your system and the docker image.

* [*] If you want to use GPU, make all GPU visible to the image: ```docker run --gpus all --name <container_name> -d -t <image_id>```

### 2. Installation via Conda

---------

#### 2.1 System

Normally, the code is able to be executed on all linux-like system. (Or, only python and a few packages are need.)

#### 2.2 Create env on conda

You will need to install all packages in the requirements.txt file.  Executing the routine requires```miniconda``` (```Anaconda``` will certainly work). 

```
cd path-to-code/phd_mission
conda env create -f phd_mission.yml
```

* If you have any difficulties or find any bugs, please get in touch and I will try to help you get it going. Suggestions for improvements are greatly appreciated.

* If you have an available Nvidia GPU, the code will use it automatically. If not, CPU will be used for training and prediction.

* If you have problem with ```torch```, use code below to install specific torch version:

  ```
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  ```

* if you have problem with ```laspy```:

  ```
  pip install laspy[lazrs,laszip]
  ```

* You may need to install the missing packages:

  ```
  pip install six seaborn scikit-learn tensorboard captum jakteristics torchviz
  ```



#### 2.3 Code and Folder

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



### 3. Reproducibility

-----------

The epoch 0 weights (i.e. default weights) for network are now available in ```path-to-code/phd_mission/src/checkpoints_default_weights```, I hope this facilitates reproducing my work. 

* In ```TLS_samplesize_20000_version```, you can find the TLS version (dense pts version), the batch size is 20000.
* In ```ULS_samplesize_3000_version```, you can find the ULS version (drone data version), the batch size is 3000.

Because the code is able to resume from checkpoint, you can simply clear old checkpoints in `src/checkpoints` directory, add the latest ones, and run the code as usual.



### 4. Citation

---------

If you find this repository help, please consider citing:

```
@inproceedings{bai2023soul,
 author = {Bai, Yuchen and Durand, Jean-Baptiste and Vincent, Gr\'{e}goire and Forbes, Florence},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {48293--48313},
 publisher = {Curran Associates, Inc.},
 title = {Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination},
 volume = {36},
 year = {2023}
}
```



