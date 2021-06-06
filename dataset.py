import random
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from numpy.testing import assert_array_almost_equal
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import dataset
from augment import Augment, Cutout


def get_test_transformations(augmentation_strategy, p):
    """
    Implements standard image augmentation.

    Args:
        p ([type]): [description]

    Returns:
        [type]: [description]
    """
    if augmentation_strategy == 'standard':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize'])
        ])

    elif augmentation_strategy == 'standard_v3':
        return transforms.Compose([
            transforms.CenterCrop(p['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    if augmentation_strategy == 'mopro':
        img_size = p['img_size']
        bigger_img_size = int(img_size + 0.15 * img_size)
        return transforms.Compose([
            transforms.Resize((bigger_img_size, bigger_img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize']),
        ])

    if augmentation_strategy == 'mopro_test':
        img_size = p['img_size']
        bigger_img_size = int(img_size + 0.15 * img_size)
        return transforms.Compose([
            transforms.Resize((bigger_img_size, bigger_img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize']),
        ])
    return transforms.Compose([
        transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(**p['transformation_kwargs']['normalize'])
    ])


def get_train_transformations(augmentation_strategy, p):
    """
    Implements strong data augmentation.

    Args:
        p ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if augmentation_strategy == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomCrop(p['img_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif augmentation_strategy == 'standard_v2':
        # Standard augmentation strategy

        return transforms.Compose([
            transforms.RandomResizedCrop(p['img_size'], scale=(0.2, 1.)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.4, 0.4, 0.4,
                                           0.1)  # not strengthened
                ],
                p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif augmentation_strategy == 'standard_v3':
        # Standard augmentation strategy

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif augmentation_strategy == 'simclr':
        # Augmentation strategy from the SimCLR paper
        strength = p['augmentation_kwargs']['jitter_strength']
        return transforms.Compose([
            transforms.RandomResizedCrop(p['img_size'], scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4 * strength, 0.4 * strength,
                                       0.4 * strength, 0.1 * strength)
            ],
                                   p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif augmentation_strategy == 'scan':
        # Augmentation strategy from our paper
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes=p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length=p['augmentation_kwargs']['cutout_kwargs']['length'],
                random=p['augmentation_kwargs']['cutout_kwargs']['random'])
        ])

    elif augmentation_strategy == 'moco':
        strength = p['augmentation_kwargs']['jitter_strength']
        # Augmentation strategy from the SimCLR paper but without gaussian blur
        return transforms.Compose([
            transforms.RandomResizedCrop(p['img_size'], scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4 * strength, 0.4 * strength,
                                       0.4 * strength, 0.1 * strength)
            ],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif augmentation_strategy == 'mopro_strong':
        crop_size = 0.2
        img_size = p['img_size']
        bigger_img_size = int(img_size + 0.15 * img_size)

        return transforms.Compose([
            transforms.Resize((bigger_img_size, bigger_img_size)),
            transforms.RandomResizedCrop(img_size, scale=(crop_size, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif augmentation_strategy == 'mopro':
        crop_size = 0.2
        img_size = p['img_size']
        bigger_img_size = int(img_size + 0.15 * img_size)
        return transforms.Compose([
            transforms.Resize((bigger_img_size, bigger_img_size)),
            transforms.RandomResizedCrop(img_size, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    else:
        raise ValueError(
            f'Invalid augmentation strategy {augmentation_strategy}')


###############################################################################
###################### Utils to corrupt datasets ##############################
###############################################################################


def get_corrupted_labels(targets, nosiy_rate, num_classes=10):
    """
    Generate corrupted labels with symmetric noise (percentual rate = nosiy_rate)

    Args:
        targets ([type]): [description]
        nosiy_rate ([type]): [description]
        num_classes (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    n_samples = len(targets)
    corrupted_targets = np.copy(targets)
    n_noisy = int(nosiy_rate * n_samples)
    print("%d Noisy samples" % (n_noisy))
    class_index = [
        np.where(np.array(targets) == i)[0] for i in range(num_classes)
    ]
    class_noisy = int(n_noisy / num_classes)
    noisy_idx = []
    # select the images to corrupt
    for d in range(num_classes):
        noisy_class_index = np.random.choice(class_index[d],
                                             class_noisy,
                                             replace=False)
        noisy_idx.extend(noisy_class_index)

    for i in noisy_idx:
        corrupted_targets[i] = other_class(n_classes=num_classes,
                                           current_class=targets[i])
    return corrupted_targets


def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert (noise >= 0.) and (noise <= 1.)
    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i + 1] = noise

    # adjust last row
    P[size - 1, 0] = noise
    return P


def get_asym_corrupted_labels(true_targets, noisy_rate, num_classes):
    corrupted_targets = true_targets.copy()
    if num_classes == 10:
        source_class = [9, 2, 3, 5, 4]
        target_class = [1, 0, 5, 3, 7]
        for s, t in zip(source_class, target_class):
            cls_idx = np.where(np.array(true_targets) == s)[0]
            n_noisy = int(noisy_rate * cls_idx.shape[0])
            noisy_sample_index = np.random.choice(cls_idx,
                                                  n_noisy,
                                                  replace=False)
            for idx in noisy_sample_index:
                corrupted_targets[idx] = t
    if num_classes == 100:
        P = np.eye(num_classes)
        nb_superclasses = 20
        nb_subclasses = 5
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses,
                                                       noisy_rate)

            flipper = np.random.RandomState(np.random.randint(0, 1000))

        for idx in np.arange(true_targets.shape[0]):
            i = true_targets[idx]
            # draw a vector with onltrue_targets an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            corrupted_targets[idx] = np.where(flipped == 1)[0]
    return corrupted_targets


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class_result = np.random.choice(other_class_list)
    return other_class_result


def asymmetric_noise(origSamples, noise=0.4, uniformShare=0.7, num_classes=10):
    """
    Corrupts orig samples with asymmetric noise (percentual rate = noise)

    Args:
        origSamples ([type]): [description]
        noise (float, optional): [description]. Defaults to 0.4.
        uniformShare (float, optional): [description]. Defaults to 0.7.
        num_classes (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    normal_share = 1 - uniformShare
    numClasses = len(np.unique(origSamples))
    perturbationWeights = np.random.normal(0, 1, 100000)
    perturbationWeights = perturbationWeights - np.min(perturbationWeights)
    perturbationWeights = perturbationWeights / np.max(perturbationWeights)
    perturbationWeights = perturbationWeights * numClasses

    a = np.digitize(perturbationWeights, np.arange(num_classes)) - 1
    x = np.bincount(a)
    x = (x / 100000) * normal_share
    x = x + uniformShare / numClasses
    perturbationWeights = x

    idx_to_change = Counter(
        np.random.choice(np.arange(numClasses),
                         size=int(len(origSamples) * noise),
                         p=perturbationWeights))

    corrupted = origSamples.copy()
    for classNb, count in idx_to_change.items():
        allIdx = np.where(origSamples == classNb)[0]
        corrupted_idx = np.random.choice(allIdx, count, replace=False)
        corrupted_values = np.mod(
            np.array([classNb] * count) +
            np.random.randint(1, numClasses, count), numClasses).astype(int)

        corrupted[corrupted_idx] = corrupted_values
    return corrupted


#########################################################################
########################### Define data sets ############################
#########################################################################
class AugmentedCIFAR(Dataset):
    """
    Custom data loader returning
    - two augmented versions of the same image for contrastive learinng
    - image indexes (e.g. for ELR loss)

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
                 labels,
                 transform=None,
                 transform_strong=None,
                 return_index=False,
                 select_idx=None,
                 dataset_name="cifar10",
                 augment=False):
        self.select_idx = select_idx
        self.return_index = return_index
        if dataset_name == "cifar10":
            data_raw = datasets.CIFAR10('../datasets/',
                                        train=True,
                                        download=True)
        if dataset_name == "cifar100":
            data_raw = datasets.CIFAR100('../datasets/',
                                         train=True,
                                         download=True)

        self.true_targets = np.array(
            data_raw.targets)  # this is for logging purpuses only
        self.all_targets = labels
        if self.select_idx is not None:
            self.data = data_raw.data[select_idx]
            self.targets = labels[select_idx]
        else:
            self.data = data_raw.data
            self.targets = labels
        self.transform = transform
        self.transform_strong = transform_strong
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def update_labels(self, new_labels):
        """
            Update dataset labels with pseudo-labels
        Args:
            new_labels ([type]): [description]
        """
        acc1 = accuracy_score(self.true_targets, new_labels)
        print(
            f"Updating dataset {len(new_labels)} pseudo_labels (ACC = {acc1})")
        if self.select_idx is not None:
            acc2 = accuracy_score(self.true_targets[self.select_idx],
                                  new_labels[self.select_idx])
            self.targets = new_labels[self.select_idx]
            print(f"Clean subset ACC = {acc2} ({len(self.targets)} samples)")
        else:
            self.targets = new_labels

    def __getitem__(self, index):
        img, target = Image.fromarray(self.data[index]), self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        return_data = []

        if self.transform is not None:
            img1 = self.transform(img)
            return_data.append(img1)
        if self.augment:
            if self.transform_strong is not None:
                img2 = self.transform_strong(img)
            elif self.transform is not None:
                img2 = self.transform(img)
            return_data.append(img2)
        return_data.append(target)
        if self.return_index:
            return_data.append(index)
        return return_data


class WebvisionDatasetH5:
    """
    Custom data loader for Clothing1M dataset
    It expects for each split train/test/val a pickle file
    located at {path}/annotations/{split}.pkl containing the
    image path and the class.
    These splits have been processed in DataExploration.ipynb
    """
    def __init__(self,
                 path,
                 dataset_type='train',
                 transform=None,
                 transform_strong=None,
                 augment=False,
                 return_index=False,
                 select_idx=None,
                 p={}):
        self.path = path

        if dataset_type not in ["train", "val", "test", "test_knn"]:
            raise ValueError(f"Incorrect dataset type {dataset_type}")
        self.nb_classes = p['nb_classes']
        img_size = p['img_size']
        if p['only_google']:
            prefix = "google"
        else:
            prefix = ""  #"google" # or  for google + flicker TODO pass as config param
        self.targets = np.load(
            f"{path}/info/{prefix}{dataset_type}_{self.nb_classes}_{img_size}.npy"
        )
        self.all_targets = self.targets.copy()
        self.hdf5_path = f"{path}/info/{prefix}{dataset_type}_{self.nb_classes}_{img_size}.hdf5"
        self.select_idx = select_idx
        if self.select_idx is not None:
            self.targets = self.targets[self.select_idx]
        self.imgs = None
        self.path = path
        self.augment = augment
        self.return_index = return_index
        self.transform = transform
        self.transform_strong = transform_strong

        self._h5_gen = None

    def __len__(self):
        return len(self.targets)

    def get_images(self):
        self.h = h5py.File(self.hdf5_path, 'r')
        if self.select_idx is not None:
            return self.h['images'][self.select_idx]
        return self.h['images']

    def __getitem__(self, index):
        if self.imgs is None:
            self.imgs = self.get_images()
        img = Image.fromarray(self.imgs[index])

        target = self.targets[index]
        target = int(target)
        if target < 0 or target >= self.nb_classes:
            print(
                f"Incorrect label {target} @ {index} Min {min(self.targets)}, Max {max(self.targets)}"
            )
        return_data = []
        if self.transform is not None:
            img1 = self.transform(img)
            return_data.append(img1)
            if self.augment:
                if self.transform_strong is not None:
                    img2 = self.transform_strong(img)
                else:
                    img2 = self.transform(img)
                return_data.append(img2)
            return_data.append(target)
            if self.return_index:
                return_data.append(index)
        return return_data

    def update_labels(self, new_labels):
        """
            Update dataset labels with pseudo-labels
        Args:
            new_labels ([type]): [description]
        """
        print(
            f"Updating dataset {len(new_labels)} labels, previous size: {len(self.targets)}"
        )

        if self.select_idx is not None:
            self.targets = new_labels[self.select_idx]
            print(f"Using clean selection of {len(self.select_idx)} samples")
        else:
            self.targets = new_labels


class Clothing1MDatasetH5:
    """
    Custom data loader for Clothing1M dataset
    It expects for each split train/test/val a pickle file
    located at {path}/annotations/{split}.pkl containing the
    image path and the class.
    These splits have been processed in DataExploration.ipynb
    """
    def __init__(self,
                 path,
                 dataset_type='train',
                 transform=None,
                 transform_strong=None,
                 augment=False,
                 select_idx=None,
                 return_index=False):
        self.path = path

        if dataset_type not in ["train", "val", "test", "test_knn"]:
            raise ValueError(f"Incorrect dataset type {dataset_type}")
        self.hdf5_path = f"{path}/annotations/{dataset_type}.hdf5"

        self.imgs = None
        self.targets = np.load(f"{path}/annotations/{dataset_type}.npy")
        self.all_targets = self.targets.copy()
        self.select_idx = select_idx
        if self.select_idx is not None:
            self.targets = self.targets[self.select_idx]
        self.path = path
        self.augment = augment
        self.return_index = return_index
        self.transform = transform
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.targets)

    def get_images(self):
        self.h = h5py.File(self.hdf5_path, 'r')
        if self.select_idx is not None:
            return self.h['images'][self.select_idx]
        return self.h['images']

    def __getitem__(self, index):
        if self.imgs is None:
            self.imgs = self.get_images()
        img = Image.fromarray(self.imgs[index])
        target = self.targets[index]
        if target < 0 or target > 14:
            print(
                f"Incorrect label {target} @ {index} Min {min(self.targets)}, Max {max(self.targets)}"
            )
        target = int(target)
        return_data = []
        if self.transform is not None:
            img1 = self.transform(img)
            return_data.append(img1)
            if self.augment:
                if self.transform_strong is not None:
                    img2 = self.transform_strong(img)
                else:
                    img2 = self.transform(img)
                return_data.append(img2)
            return_data.append(target)
            if self.return_index:
                return_data.append(index)
        return return_data

    def update_labels(self, new_labels):
        """
            Update dataset labels with pseudo-labels
        Args:
            new_labels ([type]): [description]
        """
        print(
            f"Updating dataset {len(new_labels)} labels, previous size: {len(self.targets)}"
        )

        if self.select_idx is not None:
            self.targets = new_labels[self.select_idx]
            print(f"Using clean selection of {len(self.select_idx)} samples")
        else:
            self.targets = new_labels


class TinyImageNetH5:
    """
    Custom data loader for TinyImageNet dataset
    It expects for each split train/test/val a pickle file
    located at {path}/annotations/{split}.pkl containing the
    image path and the class.
    These splits have been processed in DataExploration.ipynb
    """
    def __init__(self,
                 path,
                 dataset_type='train',
                 transform=None,
                 transform_strong=None,
                 augment=False,
                 select_idx=None,
                 return_index=False):
        self.path = path

        if dataset_type not in ["train", "test"]:
            raise ValueError(f"Incorrect dataset type {dataset_type}")
        self.hdf5_path = f"{path}/annotations/{dataset_type}.hdf5"

        self.imgs = None
        self.targets = np.load(f"{path}/annotations/{dataset_type}.npy")
        self.all_targets = self.targets.copy()
        self.select_idx = select_idx
        if self.select_idx is not None:
            self.targets = self.targets[self.select_idx]
        self.path = path
        self.augment = augment
        self.return_index = return_index
        self.transform = transform
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.targets)

    def get_images(self):
        self.h = h5py.File(self.hdf5_path, 'r')
        if self.select_idx is not None:
            return self.h['images'][self.select_idx]
        return self.h['images']

    def __getitem__(self, index):
        if self.imgs is None:
            self.imgs = self.get_images()
        img = Image.fromarray(self.imgs[index])
        target = self.targets[index]

        target = int(target)
        return_data = []
        if self.transform is not None:
            img1 = self.transform(img)
            return_data.append(img1)
            if self.augment:
                if self.transform_strong is not None:
                    img2 = self.transform_strong(img)
                else:
                    img2 = self.transform(img)
                return_data.append(img2)
            return_data.append(target)
            if self.return_index:
                return_data.append(index)
        return return_data

    def update_labels(self, new_labels):
        """
            Update dataset labels with pseudo-labels
        Args:
            new_labels ([type]): [description]
        """
        print(
            f"Updating dataset {len(new_labels)} labels, previous size: {len(self.targets)}"
        )

        if self.select_idx is not None:
            self.targets = new_labels[self.select_idx]
            print(f"Using clean selection of {len(self.select_idx)} samples")
        else:
            self.targets = new_labels


#########################################################################
########################### Define data loaders #########################
#########################################################################


def get_dataset(p,
                dataset_type='test',
                transform=None,
                transform_strong=None,
                augment=False,
                return_index=False,
                select_idx=None,
                corrupted_labels=None):
    path = p['data_path']
    if p['dataset_name'] == "webvision":
        dataset = WebvisionDatasetH5(path,
                                     dataset_type=dataset_type,
                                     transform=transform,
                                     transform_strong=transform_strong,
                                     augment=augment,
                                     return_index=return_index,
                                     select_idx=select_idx,
                                     p=p)

    if p['dataset_name'] == "clothing1M":
        dataset = Clothing1MDatasetH5(path,
                                      dataset_type=dataset_type,
                                      transform=transform,
                                      transform_strong=transform_strong,
                                      augment=augment,
                                      return_index=return_index,
                                      select_idx=select_idx)

    if p['dataset_name'] == "tinyImageNet":
        dataset = TinyImageNetH5(path,
                                 dataset_type=dataset_type,
                                 transform=transform,
                                 transform_strong=transform_strong,
                                 augment=augment,
                                 return_index=return_index,
                                 select_idx=select_idx)
    if p['dataset_name'] in ["cifar10", "cifar100"]:
        if dataset_type == 'test':
            if p['dataset_name'] == "cifar10":
                dataset = torchvision.datasets.CIFAR10(root=path,
                                                       train=False,
                                                       download=True,
                                                       transform=transform)
            else:
                dataset = torchvision.datasets.CIFAR100(root=path,
                                                        train=False,
                                                        download=True,
                                                        transform=transform)
        else:
            dataset = AugmentedCIFAR(corrupted_labels,
                                     transform=transform,
                                     transform_strong=transform_strong,
                                     select_idx=select_idx,
                                     return_index=return_index,
                                     dataset_name=p['dataset_name'],
                                     augment=augment)
    return dataset


def get_classification_loaders(p, label_data={}):
    """
    Gets data loaders for supervised learning.
    Args:
        labels ([type]): [description]
        p ([type]): [description]
        dataset_name (str, optional): [description]

    Returns:
        [type]: [description]
    """
    corrupted_labels = label_data.get("labels", None)
    train_instances = label_data.get("train_idx", None)
    if "weights" in label_data:
        train_instances = np.where(label_data['weights'] > 0.5)[0]
        print(
            f"Using train WEIGHTED samples only (# {len(train_instances)}) ..."
        )
    test_transform = get_test_transformations(
        p['augmentation']['supervised_test'], p)
    batch_size = p['batch_size_classif']
    num_workers = p['num_workers_classif']
    # 1. Create test loader

    testset = get_dataset(p,
                          dataset_type='test',
                          transform=test_transform,
                          transform_strong=None,
                          augment=False,
                          return_index=False,
                          select_idx=None,
                          corrupted_labels=None)

    test_loader = torch.utils.data.DataLoader(testset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False,
                                              drop_last=False)

    train_transform = get_train_transformations(
        p['augmentation']['supervised_train'], p)
    transform_strong = get_train_transformations(
        p['augmentation']['supervised_train_strong'], p)

    trainset = get_dataset(p,
                           dataset_type='train',
                           transform=train_transform,
                           transform_strong=None,
                           augment=False,
                           return_index=True,
                           select_idx=train_instances,
                           corrupted_labels=corrupted_labels)

    gmm_trainset = get_dataset(p,
                               dataset_type='train',
                               transform=test_transform,
                               transform_strong=None,
                               augment=True,
                               return_index=False,
                               select_idx=train_instances,
                               corrupted_labels=corrupted_labels)

    if "pred_train" in label_data:
        trainset.update_labels(label_data["pred_train"])
        gmm_trainset.update_labels(label_data["pred_train"])

    train_loader = torch.utils.data.DataLoader(trainset,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               shuffle=True,
                                               drop_last=True)

    gmm_loader = torch.utils.data.DataLoader(gmm_trainset,
                                             num_workers=num_workers,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             shuffle=False,
                                             drop_last=False)

    loaders = {"train": train_loader, "gmm": gmm_loader, "test": test_loader}

    if p['use_validation']:
        val_instances = label_data.get("val_idx", None)
        valset = get_dataset(p,
                             dataset_type='val',
                             transform=test_transform,
                             transform_strong=None,
                             augment=True,
                             return_index=False,
                             select_idx=val_instances,
                             corrupted_labels=corrupted_labels)
        if "pred_val" in label_data:
            valset.update_labels(label_data["pred_val"])

        val_loader = torch.utils.data.DataLoader(valset,
                                                 num_workers=num_workers,
                                                 batch_size=batch_size,
                                                 pin_memory=True,
                                                 shuffle=False,
                                                 drop_last=False)
        loaders["val"] = val_loader

    print(
        f"batch_size {batch_size}, num_workers {num_workers}, img size {p['img_size']}"
        +
        f"; Augmentation : [{p['augmentation']['supervised_train']}, {p['augmentation']['supervised_train_strong']}, "
        + f" {p['augmentation']['supervised_test'] }]" +
        f" Train #{len(trainset)}, test #{len(testset)}")
    return loaders


def get_representation_loaders(p, label_data={}):
    """
    Gets data loaders for supervised learning.
    Args:
        labels ([type]): [description]
        p ([type]): [description]
        dataset_name (str, optional): [description]

    Returns:
        [type]: [description]
    """

    test_transform = get_test_transformations(
        p['augmentation']['representation_test'], p)
    train_transform = get_train_transformations(
        p['augmentation']['representation_train'], p)
    transform_strong = get_train_transformations(
        p['augmentation']['representation_train_strong'], p)

    corrupted_labels = label_data.get("labels", None)
    train_instances = label_data.get("train_idx", None)
    val_instances = label_data.get("val_idx", None)

    batch_size = p['batch_size_representation']
    num_workers = p['num_workers_representation']
    print(
        f"batch_size {batch_size}, num_workers {num_workers}, img size {p['img_size']}"
        +
        f"; Augmentation : [{p['augmentation']['representation_train']}, {p['augmentation']['representation_train_strong']}, "
        + f" {p['augmentation']['representation_test'] }]")

    # 1. Create test loader
    testset = get_dataset(p,
                          dataset_type='test',
                          transform=test_transform,
                          transform_strong=None,
                          augment=False,
                          return_index=False,
                          select_idx=None,
                          corrupted_labels=None)

    test_loader = torch.utils.data.DataLoader(testset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False,
                                              drop_last=False)
    print(f"Test : {len(testset)} #")

    loaders = {"test": test_loader}

    # Save loader for supervised training on clean images
    if label_data["sup_sample_ids"] is None or len(
            label_data["sup_sample_ids"]) > 0:
        if label_data["sup_sample_ids"] is None:
            select_idx = train_instances
        elif train_instances is not None:
            select_idx = train_instances[label_data["sup_sample_ids"]]
        else:
            select_idx = label_data["sup_sample_ids"]

        trainset_sup = get_dataset(p,
                                   dataset_type='train',
                                   transform=train_transform,
                                   transform_strong=transform_strong,
                                   augment=True,
                                   return_index=True,
                                   select_idx=select_idx,
                                   corrupted_labels=corrupted_labels)
        if "pred_train" in label_data:
            trainset_sup.update_labels(label_data["pred_train"])
        loaders["train_loader_sup"] = torch.utils.data.DataLoader(
            trainset_sup,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True)
    else:
        loaders["train_loader_sup"] = None

    # Save loader for unsupervised training on remaining images
    if label_data["unsup_sample_ids"] is None or len(
            label_data["unsup_sample_ids"]) > 0:

        if label_data["unsup_sample_ids"] is None:
            select_idx = train_instances
        elif train_instances is not None:
            select_idx = train_instances[label_data["unsup_sample_ids"]]
        else:
            select_idx = label_data["unsup_sample_ids"]

        trainset_unsup = get_dataset(p,
                                     dataset_type='train',
                                     transform=train_transform,
                                     transform_strong=transform_strong,
                                     augment=True,
                                     return_index=True,
                                     select_idx=select_idx,
                                     corrupted_labels=corrupted_labels)
        if "pred_train" in label_data:
            trainset_unsup.update_labels(label_data["pred_train"])
        loaders["train_loader_unsup"] = torch.utils.data.DataLoader(
            trainset_unsup,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True)
        print(f"Train : {len(trainset_unsup)} #")
    else:
        loaders["train_loader_unsup"] = None

    # Add memory loader for Romain
    loaders["memory"] = get_memory_loaders(label_data, p)

    return loaders


def get_memory_loaders(label_data, p):
    """
    Gets data loaders for supervised learning.
    Args:
        labels ([type]): [description]
        p ([type]): [description]

    Returns:
        [type]: [description]
    """
    dataset_name = p["dataset_name"]
    test_transform = get_test_transformations(
        p['augmentation']['supervised_test'], p)

    batch_size = p['batch_size_representation']
    num_workers = p['num_workers_representation']
    print(f"batch_size {batch_size}, num_workers {num_workers}")

    print(f"batch_size {batch_size}, num_workers {num_workers}")
    testset = None
    if dataset_name == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='../datasets/',
                                               train=False,
                                               download=True,
                                               transform=test_transform)
    if dataset_name == "cifar100":
        testset = torchvision.datasets.CIFAR100(root='../datasets/',
                                                train=False,
                                                download=True,
                                                transform=test_transform)

    if dataset_name == "clothing1M":
        testset = get_dataset(p,
                              dataset_type='train',
                              transform=test_transform,
                              transform_strong=None,
                              augment=False,
                              return_index=False)

    if dataset_name == "webvision":
        testset = get_dataset(p,
                              dataset_type='test',
                              transform=test_transform,
                              transform_strong=None,
                              augment=False,
                              return_index=False)

    if dataset_name == "tinyImageNet":
        testset = get_dataset(p,
                              dataset_type='test',
                              transform=test_transform,
                              transform_strong=None,
                              augment=False,
                              return_index=False)

    if testset is None:
        return None
    memory_loader = torch.utils.data.DataLoader(testset,
                                                num_workers=num_workers,
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=False)

    return memory_loader


def update_data_loader_labels(loader, new_labels, shuffle=False):
    """
    Updates existing data loader with new pseudo labels.

    Args:
        loader ([type]): [description]
        new_labels ([type]): [description]

    Returns:
        [type]: [description]
    """

    ds = loader.dataset
    ds.update_labels(new_labels)

    new_loader = torch.utils.data.DataLoader(ds,
                                             num_workers=loader.num_workers,
                                             batch_size=loader.batch_size,
                                             pin_memory=loader.pin_memory,
                                             shuffle=shuffle,
                                             drop_last=True)
    return new_loader


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
