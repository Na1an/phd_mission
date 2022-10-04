# phd_mission
For my PhD project



## 1. Installation

### 1.1 CUDA version

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html  * uncheck this option, for many reason...

* Preserve global shift on save
  * uncheck this option, it will cause shift bug.



### 1.2 miniconda

* see this guide

  * https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

    

### 1.3 create env and download requirements_files

```conda env create --name phd_mission --file requirement_files.yml```

