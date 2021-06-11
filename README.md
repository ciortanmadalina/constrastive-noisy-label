# A Framework using Contrastive Learning for Classification with Noisy Labels

This repository contains the pytorch implementation of the paper "A Framework using Contrastive Learning for Classification with Noisy Labels" (MDPI publication https://www.mdpi.com/2306-5729/6/6/61 ).   


A framework using contrastive learning as a pre-training task is proposed to perform classification in the presence of noisy labels. This paper provides an extensive empirical study showing that a preliminary contrastive learning step brings a significant gain in performance using various different loss functions: non robust, robust, and early-learning regularized losses. Several recent strategies boosting the performance of noisy-label models are also evaluated: pseudo-labelling, sample selection with Gaussian Mixture models, weighted supervised contrastive learning, and mixup with bootstrapping. The experiments are performed on standard benchmarks and real-world datasets and demonstrate that: i) the contrastive pre-training can increase robustness of any loss function to noisy labels and ii) the full framework with all recent strategies can achieve results close to the state of the art. 

# Overview of the repository
- **notebooks** folder contains all jupyter notebooks to run the project, as detailed below.
- **data/models** contains model dumps
- **data/results** contains the results of running all experiments, needed to reproduce the plots
- **docker** contains the Dockerfile to create the image used to run all python experiments
- train.py contains the main functionalities for training and evaluating the model results
- model.py contains the network definition
- loss.py contains the implementation of the loss functions
- utils.py contains various utility functions
- dataset.py contains datasets and data loaders
- augment.py contains the custom image augmentations


### Overview of notebooks
- **notebooks/Demo.ipynb** represents the main entry point, contains code to run the proposed method on an arbitrary datset
- **notebooks/*_data_preparation.ipynb** provides data download and the preprocessing needed to reproduce the results for CIFAR10, CIFAR100, Webvision and Clothing1M


## Environment Setup
We have employed a docker container to facilitate reproducing the paper results.

### Python environment
It can be launched by running the following:

```
cd docker  
docker build -t contrastive .
```

The image has been created for GPU usage. In order to run it on CPU, in the Dockerfile, the line "pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime" should be replaced with a CPU version.

The command above created a docker container tagged as **contrastive** . Assuming the project has been cloned locally in a parent folder named notebooks, the image can be launched locally with:

```
docker run -it --runtime=nvidia -v ~/notebooks:/workspace/notebooks -p 8888:8888 contrastive
```
This starts up a jupyter notebook server, which can be accessed at http://localhost:8888/tree/notebooks


## Data

All benckmarked datasets can be downloaded and preprocessed using the notebooks:
- **notebooks/CIFAR_data_preparation.ipynb** provides data download and the preprocessing needed to reproduce the results for CIFAR10, CIFAR100
- **notebooks/Webvision_data_preparation.ipynb** provides data download and the preprocessing needed to reproduce the results for Webvision dataset
- **notebooks/Clothing1M_data_preparation.ipynb** provides data download and the preprocessing needed to reproduce the results for Clothing1M dataset


# Video

An overview of our method is presented in this video: https://www.youtube.com/watch?v=08LCrsZ_eww

# Poster

The poster presented at CAP can be found in CAP_poster_31_Ciortan_et_al.pdf
