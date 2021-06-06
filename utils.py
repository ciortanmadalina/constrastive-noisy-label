import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

import loss as lossfile
import models
import utils


def get_params(dataset_name,
               img_size=32,
               batch_size_classif=64,
               batch_size_representation=64,
               num_workers_classif=3,
               num_workers_representation=1,
               nb_classes=10,
               model_name="moco",
               use_protype=True,
               use_validation=False,
               arch='resnet18',
               data_path='../datasets/',
               use_tqdm=True,
               classification_arch='multilayer',
               projection_head=True,
               mopro_use_ce=False,
               high_dim=False,
               augmentation=None):
    """
    All learning rate parameters are encoded wrt a batch size of 256 
    (to be aligned with values in original implementations).
    For this reason, we dynamically scale the learning rate depending
    on the input batch size.

    Args:
        dataset_name ([type]): [description]
        img_size (int, optional): [description]. Defaults to 32.
        batch_size_classif (int, optional): [description]. Defaults to 64.
        batch_size_representation (int, optional): [description]. Defaults to 64.
        num_workers_classif (int, optional): [description]. Defaults to 3.
        num_workers_representation (int, optional): [description]. Defaults to 1.
        nb_classes (int, optional): [description]. Defaults to 10.
        model_name (str, optional): [description]. Defaults to "moco".
        use_protype (bool, optional): [description]. Defaults to True.
        use_validation (bool, optional): [description]. Defaults to False.
        arch (str, optional): [description]. Defaults to 'resnet18'.
        data_path (str, optional): [description]. Defaults to '../datasets/'.
        use_tqdm (bool, optional): [description]. Defaults to True.
        classification_arch (str, optional): [description]. Defaults to 'multilayer'.
        projection_head (bool, optional): [description]. Defaults to True.
        mopro_use_ce (bool, optional): [description]. Defaults to False.
        high_dim (bool, optional): [description]. Defaults to False.
        augmentation ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    common_params = {
        'dataset_name': dataset_name,
        'use_protype': use_protype,
        'schedule': False,  #[40, 80],
        'cos': False,
        'momentum': 0.9,
        'low_dim': 128,
        'high_dim': high_dim,
        'moco_queue': 8192,
        'moco_m': 0.999,
        'proto_m': 0.999,
        'temperature': 0.1,
        'w_inst': 1,
        'w_proto': 1,
        'start_clean_epoch': 11,
        'pseudo_th': 0.8,
        'alpha': 0.5,
        'pseudo_w': 0.5,
        'is_moco': model_name.startswith("moco"),
        'is_mopro': model_name.startswith("mopro"),
        'mopro_use_ce': mopro_use_ce,
        'model_name': model_name,
        'img_size': img_size,
        'num_neighbors': 20,
        'num_workers_classif': num_workers_classif,
        'num_workers_representation': num_workers_representation,
        'batch_size_classif': batch_size_classif,
        'batch_size_representation': batch_size_representation,
        'use_validation': use_validation,
        'nb_classes': nb_classes,
        'arch': arch,
        'data_path': data_path,
        'use_tqdm': use_tqdm,
        'classification_arch': classification_arch,
        'projection_head': projection_head,
        'nesterov': False,
        'monitoring_time': False,
        'gmm_negative_entropy': False,
        'cos': False,
        'schedule': False,
        'adam': False
    }

    if dataset_name == "cifar10":
        p = get_cifar10_params(common_params)

    if dataset_name == "cifar100":
        p = get_cifar100_params(common_params)

    if dataset_name == "clothing1M":
        p = get_clothing1M_params(common_params)
        
    if dataset_name == "tinyImageNet":
        p = get_tiny_imagenet_params(common_params)

    if dataset_name == "webvision":
        p = get_webvision_params(common_params)

    p = {**p, **common_params}
    # overwrite augmentation
    if augmentation is not None:
        p["augmentation"] = augmentation

    if p['batch_size_classif'] != 256:
        factor = p['batch_size_classif'] / 256
        print(
            f"Scaling supervised LR : {p['lr']['supervised']} -> " +
            f"{p['lr']['supervised']*factor}"
            +
            f"Scaling supervised classifier LR : " +
            f"{p['lr']['supervised_classifier']} " +
            f"-> {p['lr']['supervised_classifier'] * factor}"
            +
            f" and finetune {p['lr']['supervised_fine']}" +
            f"-> {p['lr']['supervised_fine']*factor}"
        )
        p['lr']['supervised'] = p['lr']['supervised'] * factor
        p['lr']['supervised_classifier'] = p['lr']['supervised_classifier'] * factor
        p['lr']['supervised_fine'] = p['lr']['supervised_fine'] * factor

    if p['batch_size_representation'] != 256:
        factor = p['batch_size_representation'] / 256
        print(f"Scaling representation LR {p['lr']['representation']} -> " +
              f"{p['lr']['representation']*factor}")
        p['lr']['representation'] = p['lr']['representation'] * factor

    return p


def get_cifar10_params(common_params):
    p = {
        'nb_classes': 10,
        'augmentation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.201]
            },
            'num_strong_augs': 4,
            'cutout_kwargs': {
                'n_holes': 1,
                'length': 16,
                'random': True
            },
            'jitter_strength': 1.0
        },
        'transformation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.201]
            }
        },
        'optimizer': 'adam',
        'optimizer_kwargs': {
            'lr': 0.0001,
            'weight_decay': 0.0001
        },
        'lr': {
            'representation': 0.03,
            'supervised': 0.01,
            'supervised_classifier': 0.01,
            'supervised_fine': 0.01
        },
        'weight_decay': {
            'representation': 1e-4,
            'supervised': 1e-4,
            'supervised_classifier': 1e-4,
            'supervised_fine': 1e-5
        },
        'augmentation': {
            "supervised_train": "standard",
            "supervised_train_strong": "standard",
            "supervised_test": "standard",
            "representation_train": "mopro",
            "representation_train_strong": "mopro_strong",
            "representation_test": "standard",
        },
        'scheduler': 'constant'
    }
    return p


def get_cifar100_params(common_params):
    p = {
        'nb_classes': 100,
        'num_neighbors': 20,
        'augmentation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.5071, 0.4865, 0.4409],
                'std': [0.2673, 0.2564, 0.2762]
            },
            'num_strong_augs': 4,
            'cutout_kwargs': {
                'n_holes': 1,
                'length': 16,
                'random': True
            },
            'jitter_strength': 1.0
        },
        'transformation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.5071, 0.4865, 0.4409],
                'std': [0.2673, 0.2564, 0.2762]
            }
        },
        'optimizer': 'adam',
        'optimizer_kwargs': {
            'lr': 0.0001,
            'weight_decay': 0.0001
        },
        'lr': {
            'representation': 0.03,
            'supervised': 0.1,
            'supervised_classifier': 0.1,
            'supervised_fine': 0.1
        },
        'weight_decay': {
            'representation': 1e-4,
            'supervised': 1e-5,
            'supervised_classifier': 1e-4,
            'supervised_fine': 1e-5
        },
        'augmentation': {
            "supervised_train": "standard",
            "supervised_train_strong": "standard",
            "supervised_test": "standard",
            "representation_train": "mopro",
            "representation_train_strong": "mopro_strong",
            "representation_test": "standard",
        },
        'scheduler': 'constant'
    }
    return p


def get_clothing1M_params(common_params):
    p = {
        'augmentation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.6959, 0.6537, 0.6371],
                'std': [0.3113, 0.3192, 0.3214]
            },
            'num_strong_augs': 4,
            'cutout_kwargs': {
                'n_holes': 1,
                'length': 16,
                'random': True
            },
            'jitter_strength': 1.0
        },
        'transformation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.6959, 0.6537, 0.6371],
                'std': [0.3113, 0.3192, 0.3214]
            }
        },
        'optimizer': 'adam',
        'optimizer_kwargs': {
            'lr': 0.0001,
            'weight_decay': 0.0001
        },
        'lr': {
            'representation': 0.03,
            'supervised': 0.01,
            'supervised_classifier': 0.1,
            'supervised_fine': 0.001
        },
        'weight_decay': {
            'representation': 1e-4,
            'supervised': 1e-4,
            'supervised_classifier': 1e-4,
            'supervised_fine': 1e-5
        },
        'augmentation': {
            "supervised_train": "standard",
            "supervised_train_strong": "standard",
            "supervised_test": "standard",
            "representation_train": "mopro",
            "representation_train_strong": "mopro_strong",
            "representation_test": "standard",
        },
        'scheduler': 'constant'
    }
    return p


def get_webvision_params(common_params):
    p = {
        'augmentation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'num_strong_augs': 4,
            'cutout_kwargs': {
                'n_holes': 1,
                'length': 16,
                'random': True
            },
            'jitter_strength': 1.0
        },
        'transformation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'optimizer': 'adam',
        'optimizer_kwargs': {
            'lr': 0.0001,
            'weight_decay': 0.0001
        },
        'lr': {
            'representation': 0.1,
            'supervised': 0.4,
            'supervised_classifier': 0.04,
            'supervised_fine': 0.04
        },
        'weight_decay': {
            'representation': 1e-4,
            'supervised': 3e-5,
            'supervised_fine': 3e-6,
            'supervised_classifier': 3e-6,
        },
        'augmentation': {
            "supervised_train": "standard",
            "supervised_train_strong": "standard",
            "supervised_test": "standard",
            "representation_train": "mopro",
            "representation_train_strong": "mopro_strong",
            "representation_test": "mopro_test",
        },
        'scheduler': 'constant',
        'only_google': False
    }
    return p




def get_tiny_imagenet_params(common_params):
    p = {
        'augmentation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010]
            },
            'num_strong_augs': 4,
            'cutout_kwargs': {
                'n_holes': 1,
                'length': 16,
                'random': True
            },
            'jitter_strength': 1.0
        },
        'transformation_kwargs': {
            'crop_size': common_params['img_size'],
            'normalize': {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010]
            }
        },
        'optimizer': 'adam',
        'optimizer_kwargs': {
            'lr': 0.0001,
            'weight_decay': 0.0001
        },
        'lr': {
#             'representation': 0.03,
#             'supervised': 0.1,
#             'supervised_classifier': 30,
#             'supervised_fine': 0.1,
            'representation': 0.1,
            'supervised': 0.1,
            'supervised_classifier': 30,
            'supervised_fine': 0.1

        },
        'weight_decay': {
            'representation': 1e-4,
            'supervised': 1e-4,
            'supervised_fine': 1e-4,
            'supervised_classifier': 0,
        },
        'augmentation': {
            "supervised_train": "standard",
            "supervised_train_strong": "standard",
            "supervised_test": "standard",
            "representation_train": "mopro",
            "representation_train_strong": "mopro_strong",
            "representation_test": "mopro_test",
        },
        'scheduler': 'constant',
        'only_google': False
    }
    return p


def get_model_and_criterion(dataset_name,
                            loss_name="nfl_rce",
                            model="mopro",
                            num_samples=50000,
                            p={}):
    """
    Instantiates a model and supervised loss based on input parameters
    Args:
        dataset_name ([type]): [description]
        loss_name (str, optional): [description]. Defaults to "nfl_rce".
        model (str, optional): [description]. Defaults to "simple".
        num_samples (int, optional): [description]. Defaults to 50000.
        p (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    model_instance = None
    device = utils.get_device()
    if loss_name == "ce":
        criterion_sup = torch.nn.CrossEntropyLoss()
        
    if model == "simple":
        model_instance = models.SimpleModel(nb_classes = p["nb_classes"])
    if model == "resnet18":
        model_instance = models.SimCLR(
            feature_dim=p["low_dim"],
            nb_classes=p["nb_classes"],
            arch=p["arch"],
            bn_splits=8,
            img_size=p["img_size"],
            classification_arch=p["classification_arch"],
            mlp=p['projection_head'],
            mlp_dim=p['high_dim'])
    
    if model == "resnet50":
        model_instance = models.SimCLR(
            feature_dim=p["low_dim"],
            nb_classes=p["nb_classes"],
            arch=p["arch"],
            bn_splits=8,
            img_size=p["img_size"],
            classification_arch=p["classification_arch"],
            mlp=p['projection_head'],
            mlp_dim=p['high_dim'])

    if model == "mopro":
        model_instance = models.MoPro(p,
                                      K=4096,
                                      m=0.99,
                                      T=0.1,
                                      arch=p['arch'],
                                      mlp=p['projection_head'])
        
    if model == "moco_sup":
        model_instance = models.ModelSupMoCo(p,
                                          K=4096,
                                          m=0.99,
                                          T=0.1,
                                          arch=p['arch'],
                                          mlp=p['projection_head'])


    if model == "moco":
        model_instance = models.ModelMoCo(p,
                                          K=4096,
                                          m=0.99,
                                          T=0.1,
                                          arch=p['arch'],
                                          mlp=p['projection_head'])

    # Robust losses are dependent on the number of target classes
    if dataset_name == "webvision":
        if loss_name == "nfl_rce":
            criterion_sup = lossfile.NFLandRCE(alpha=50,
                                               beta=0.1,
                                               gamma=0.5,
                                               num_classes=p['nb_classes'])
            
        if loss_name == "elr":
            # TODO check loss parameters
            criterion_sup = lossfile.ELR(num_samples,
                                         num_classes=p['nb_classes'],
                                         beta=0.7,
                                         lambda_=3)
    if dataset_name == "clothing1M":
        if loss_name == "nfl_rce":
            criterion_sup = lossfile.NFLandRCE(alpha=1,
                                               beta=1,
                                               gamma=0.5,
                                               num_classes=p['nb_classes'])
        if loss_name == "elr":
            criterion_sup = lossfile.ELR(num_samples,
                                         num_classes=p['nb_classes'],
                                         beta=0.7,
                                         lambda_=3)
    if dataset_name == "cifar10":
        if loss_name == "nfl_rce":
            criterion_sup = lossfile.NFLandRCE(alpha=1,
                                               beta=1,
                                               gamma=0.5,
                                               num_classes=p['nb_classes'])
        if loss_name == "elr":
            criterion_sup = lossfile.ELR(num_samples,
                                         num_classes=p['nb_classes'],
                                         beta=0.7,
                                         lambda_=3)

    if dataset_name == "cifar100":
        if loss_name == "nfl_rce":
            criterion_sup = lossfile.NFLandRCE(alpha=10,
                                               beta=0.1,
                                               gamma=0.5,
                                               num_classes=p['nb_classes'])
        if loss_name == "elr":
            criterion_sup = lossfile.ELR(num_samples,
                                         num_classes=p['nb_classes'],
                                         beta=0.9,
                                         lambda_=7)
    if model_instance is not None:
        model_instance = model_instance.to(device)
    return model_instance, criterion_sup


def get_device():
    """[summary]

    Returns:
        [type]: [description]
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def get_checkpoint_file(checkpoint_folder, name=""):
    """[summary]

    Args:
        checkpoint_folder ([type]): [description]
        name (str, optional): [description]. Defaults to "".

    Returns:
        [type]: [description]
    """
    import datetime

    if os.path.isdir(checkpoint_folder) == False:
        os.makedirs(checkpoint_folder, exist_ok=True)

    checkpoint_path_file = checkpoint_folder + "/" + datetime.datetime.now(
    ).strftime('%Y_%m_%d-%H_%M_%S')
    checkpoint_path_file = f"{checkpoint_path_file}_{name}"
    print(f"Saving model to {checkpoint_path_file}")
    return checkpoint_path_file


def get_lr(optimizer):
    """[summary]

    Args:
        optimizer ([type]): [description]

    Returns:
        [type]: [description]
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_round_results(exp):
    for i in range(4):
        name = f"{exp}_representation_round_{i}"
        if os.path.isfile(f'../data/results/{name}.pickle'):
            with open(f'../data/results/{name}.pickle', 'rb') as handle:
                r = pickle.load(handle)
            print(f"Round {name}: acc : {r['acc']}")

        name = f"{exp}_supervised_round_{i}"
        if os.path.isfile(f'../data/results/{name}.pickle'):
            with open(f'../data/results/{name}.pickle', 'rb') as handle:
                r = pickle.load(handle)
            print(f"Round {name}: acc : {r['acc']}")


def clustering_acc(y_true, y_pred):
    """
    Comute accuaracy.

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    acc = accuracy_score(y_true, y_pred)
    return acc
