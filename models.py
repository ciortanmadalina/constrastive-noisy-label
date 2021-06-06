import math
from functools import partial
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet

import utils
from PreResNet import ResNet18


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H,
                           W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits), True, self.momentum,
                self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(
                running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(
                running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(input, self.running_mean,
                                            self.running_var, self.weight,
                                            self.bias, False, self.momentum,
                                            self.eps)


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      padding=padding), nn.BatchNorm2d(out_planes), nn.ReLU())

    def forward(self, x):
        return self.out_conv(x)


class SimpleModel(nn.Module):
    def __init__(self, nb_classes=10, projection=True):
        super(SimpleModel, self).__init__()
        self.type = type
        self.fc_size = 4 * 4 * 196
        self.phase = "1"
        self.projection = projection

        self.block1 = nn.Sequential(ConvBrunch(3, 64, 3),
                                    ConvBrunch(64, 64, 3),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(ConvBrunch(64, 128, 3),
                                    ConvBrunch(128, 128, 3),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(ConvBrunch(128, 196, 3),
                                    ConvBrunch(196, 196, 3),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(4 * 4 * 196, 256),
                                 nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, 128))

        self.classifier = nn.Sequential(nn.BatchNorm1d(128), nn.ReLU(),
                                        nn.Linear(128, nb_classes))

        self.architecture = [self.block1, self.block2, self.block3]

        # projection MLP
        if projection:
            self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(128, 128))

        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,
                                         mode='fan_in',
                                         nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x, forward_pass='default'):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        output = self.classifier(x)
        if self.projection:
            x = self.fc(x)
        return output, x

    def get_projection_features(self, x):
        _, x = self.forward(x)
        return x

    def get_features(self, x):
        _, x = self.forward(x)
        # note: not normalized here
        return x

    def set_phase(self, phase):
        """
        Updates phase parameter and makes the underlying layers
        frozen/ updatable

        Args:
            phase ([type]): [description]
        """
        if phase not in ["1", "2", "3", "4"]:
            raise Exception(f'Invalid phase {phase} ')
        self.phase = phase
        if phase == "1":  # Representation with projection head
            for param in self.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = False

        if phase == "2":  # Supervised and freeze projection head
            for param in self.parameters():
                param.requires_grad = True
            if self.projection:
                for param in self.fc.parameters():
                    param.requires_grad = False

        if phase == "3":  # nb_epochs_output_training update only of the classification head
            for param in self.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

        if phase == "4":
            for param in self.parameters():
                param.requires_grad = True


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    Args:

    """
    def __init__(self,
                 img_size=224,
                 feature_dim=128,
                 nb_classes=100,
                 arch=None,
                 bn_splits=16,
                 classification_arch='multilayer'):
        """[summary]

        Args:
            img_size (int, optional): [description]. Defaults to 224.
            feature_dim (int, optional): [description]. Defaults to 128.
            nb_classes (int, optional): [description]. Defaults to 100.
            arch ([type], optional): [description]. Defaults to None.
            bn_splits (int, optional): [description]. Defaults to 16.
            classification_arch (str, optional): [description]. Defaults to 'multilayer'.
        """
        super(ModelBase, self).__init__()
        preact = False
        # use split batchnorm
        norm_layer = partial(
            SplitBatchNorm,
            num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d

        if 'pre' in arch:
            net = ResNet18(num_classes=feature_dim)
            preact = True
        else:
            resnet_arch = getattr(resnet, arch)

            net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)
        self.net = []

        if not preact:
            for name, module in net.named_children():
                # Modify the receptive field for small images
                if name == 'conv1':
                    if img_size <= 64:
                        module = nn.Conv2d(3,
                                           64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False)
                if isinstance(module, nn.MaxPool2d):
                    if img_size <= 64:
                        continue
                    else:
                        pass
                if isinstance(module, nn.Linear):
                    self.net.append(nn.Flatten(1))
                    self.fc = module
                    continue
                self.net.append(module)

            self.net = nn.Sequential(*self.net)
        else:
            for name, module in net.named_children():
                if isinstance(module, nn.Linear):
                    self.fc = module
                    net.linear = nn.Flatten(1)
            self.net = net

        if classification_arch == 'linear':
            self.classifier = nn.Linear(self.fc.weight.shape[1], nb_classes)
        if classification_arch == 'multilayer':
            self.classifier = nn.Sequential(
                nn.Linear(self.fc.weight.shape[1], feature_dim),
                nn.BatchNorm1d(feature_dim), nn.ReLU(),
                nn.Linear(feature_dim, nb_classes))
        if classification_arch == 'multilayer2048':
            self.classifier = nn.Sequential(
                nn.Linear(self.fc.weight.shape[1], 2048), nn.BatchNorm1d(2048),
                nn.ReLU(), nn.Linear(2048, nb_classes))

    def forward(self, x):
        """Forward model pass

        Args:
            x ([type]): image content

        Returns:
            [type]: the classification output and the projection layer
        """
        x = self.net(x)
        output = self.classifier(x)

        x = self.fc(x)
        x = x.squeeze()

        return output, x

    def get_features(self, x):
        x = self.net(x)
        x = x.squeeze()
        # note: not normalized here
        return x

    def get_projection_features(self, x):
        _, x = self.forward(x)
        return x

    def set_phase(self, phase):
        """
        Updates phase parameter and makes the underlying layers
        frozen/ updatable

        Args:
            phase ([type]): [description]
        """
        if phase not in ["1", "2", "3", "4"]:
            raise Exception(f'Invalid phase {phase} ')
        self.phase = phase
        if phase == "1":  # Representation with projection head
            for param in self.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = False

        if phase == "2":  # Supervised and freeze projection head
            for param in self.parameters():
                param.requires_grad = True
            for param in self.fc.parameters():
                param.requires_grad = False

        if phase == "3":  # nb_epochs_output_training update only of the classification head
            for param in self.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

        if phase == "4":
            for param in self.parameters():
                param.requires_grad = True


class SimCLR(ModelBase):
    """Implementents contrastive SimClr model https://arxiv.org/pdf/2002.05709.pdf
    Args:
        ModelBase ([type]): [description]
    """
    def __init__(self,
                 img_size=224,
                 feature_dim=128,
                 nb_classes=100,
                 arch=None,
                 bn_splits=16,
                 classification_arch='multilayer',
                 mlp=False,
                 mlp_dim=False):
        super(SimCLR, self).__init__(img_size=img_size,
                                     feature_dim=feature_dim,
                                     nb_classes=nb_classes,
                                     arch=arch,
                                     bn_splits=bn_splits,
                                     classification_arch=classification_arch)

        if mlp:  # hack: brute-force replacement
            dim_resnet_out = self.fc.weight.shape[1]
            if mlp_dim is False:
                mlp_dim = dim_resnet_out
            if img_size > 30:
                self.fc = nn.Sequential(nn.Linear(dim_resnet_out, mlp_dim),
                                        nn.ReLU(),
                                        nn.Linear(mlp_dim, feature_dim))
            else:
                # The projection head remains linear
                pass
        else:
            low_dim = self.fc.weight.shape[1]
            self.fc = nn.Identity()


class DictToObject(object):
    def __init__(self, d):
        self.__dict__ = d


class ModelMoCo(nn.Module):
    """Implements Moco model https://arxiv.org/abs/1911.05722

    Args:
        nn ([type]): [description]
    """
    def __init__(self,
                 p,
                 K=4096,
                 m=0.99,
                 T=0.1,
                 arch='resnet18',
                 bn_splits=8,
                 symmetric=True,
                 mlp=False):
        super(ModelMoCo, self).__init__()
        self.p = DictToObject(p)
        nb_classes = self.p.nb_classes
        low_dim = self.p.low_dim
        high_dim = self.p.high_dim
        img_size = self.p.img_size

        symmetric = False
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.mlp = mlp

        # create the encoders
        self.encoder_q = ModelBase(
            feature_dim=low_dim,
            nb_classes=nb_classes,
            arch=arch,
            bn_splits=bn_splits,
            img_size=img_size,
            classification_arch=self.p.classification_arch)
        self.encoder_k = ModelBase(
            feature_dim=low_dim,
            nb_classes=nb_classes,
            arch=arch,
            bn_splits=bn_splits,
            img_size=img_size,
            classification_arch=self.p.classification_arch)

        if mlp:  # hack: brute-force replacement
            dim_resnet_out = self.encoder_q.fc.weight.shape[1]
            if high_dim is False:
                high_dim = dim_resnet_out
            if img_size > 32:
                self.encoder_q.fc = nn.Sequential(
                    nn.Linear(dim_resnet_out, high_dim), nn.ReLU(),
                    nn.Linear(high_dim, low_dim))
                self.encoder_k.fc = nn.Sequential(
                    nn.Linear(dim_resnet_out, high_dim), nn.ReLU(),
                    nn.Linear(high_dim, low_dim))
            else:
                # The projection head remains linear
                pass
        else:
            low_dim = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Identity()
            self.encoder_k.fc = nn.Identity()

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(low_dim, K))
        self.register_buffer("queue_labels",
                             -1 * torch.ones(K, dtype=torch.long))
        self.register_buffer("queue_sample_index",
                             torch.ones(K, dtype=torch.long))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, class_labels=None, sample_index=None):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        if class_labels is not None:
            self.queue_labels[ptr:ptr + batch_size] = class_labels
        if sample_index is not None:
            self.queue_sample_index[ptr:ptr + batch_size] = sample_index
        ptr = (ptr + batch_size) % self.K  # move pointer
        #         print(f"Setting queue labels {self.queue_labels} ")
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss_inputs(self, im_q, im_k, supervised=False):
        # compute query features
        output, q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            _, k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        if supervised:
            # Comput product with current batch and augmented batch as well
            # The logits will contain : current batch, current augm batch, queue
            combined = torch.cat(
                [q.T, k.T, self.queue.clone().detach()], dim=1)
            l_neg = torch.einsum('nc,ck->nk', [q, combined])
            logits = l_neg
            # save bank index for weights
            self.current_queue_index = self.queue_sample_index.clone().detach()
        else:
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels, q, k, output

    def forward(self, im1, im2, class_labels=None, sample_index=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2, output = self.contrastive_loss(im1, im2)
            loss_21, q2, k1, output = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            logits, labels, q, k, output = self.contrastive_loss_inputs(
                im1, im2, supervised=class_labels is not None)

        self._dequeue_and_enqueue(k, class_labels, sample_index)
        if sample_index is not None:
            self.current_queue_index = torch.cat(
                [sample_index, sample_index, self.current_queue_index], dim=0)

        return logits, labels, q, k, output

    def set_phase(self, phase):
        """
        Updates phase parameter and makes the underlying layers
        frozen/ updatable

        Args:
            phase ([type]): [description]
        """
        if phase not in ["1", "2", "3", "4"]:
            raise Exception(f'Invalid phase {phase} ')
        self.phase = phase
        if phase == "1":  # Representation with projection head
            for param in self.encoder_q.parameters():
                param.requires_grad = True
            for param in self.encoder_q.classifier.parameters():
                param.requires_grad = False

        if phase == "2":  # Supervised and freeze projection head
            for param in self.encoder_q.parameters():
                param.requires_grad = True
            if self.mlp:
                for param in self.encoder_q.fc.parameters():
                    param.requires_grad = False

        if phase == "3":  # nb_epochs_output_training update only of the classification head
            for param in self.encoder_q.parameters():
                param.requires_grad = False
            for param in self.encoder_q.classifier.parameters():
                param.requires_grad = True

        if phase == "4":
            for param in self.encoder_q.parameters():
                param.requires_grad = True


class ModelSupMoCo(ModelMoCo):
    """Proposed supervised implementation of Moco model https://arxiv.org/abs/1911.05722

    Args:
        nn ([type]): [description]
    """
    def __init__(self, *args, **kwargs):
        super(ModelSupMoCo, self).__init__(*args, **kwargs)

        # create the queue
        self.register_buffer(
            "labels_bank",
            torch.randint(self.p.nb_classes, (1, self.K), dtype=torch.long))
        self.register_buffer("labels_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("labels_counter", torch.zeros(1,
                                                           dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue_labels(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.labels_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.labels_bank[:, ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.labels_ptr[0] = ptr

        self.labels_counter[0] += batch_size  # add to the counter

    def contrastive_loss_inputs(self, im_q, im_k, labels=None):
        device = utils.get_device()
        batch_size = im_q.shape[0]

        memory = False
        batch = True

        if self.labels_counter[0] < (self.K + 1):
            # self._dequeue_and_enqueue_labels(labels)
            return super(ModelSupMoCo,
                         self).contrastive_loss_inputs(im_q, im_k)
        else:
            if labels.shape[0] % batch_size != 0:
                raise ValueError('The batch size does not fit the labels size')

            labels = labels.view(-1, 1).to(device)

            # In memory
            index_ = torch.eq(labels, self.labels_bank).float()
            index_ = torch.nonzero(index_)

            duplicated_row_index = index_[:, 0]
            new_labels_ce = index_[:, 1]

            # In batch
            index_batch = torch.eq(labels, labels.t()).float()
            # mask-out self-contrast cases
            index_batch = torch.scatter(
                index_batch, 1,
                torch.arange(batch_size).view(-1, 1).to(device), 0)

            index_batch = torch.nonzero(index_batch)

            # compute query features
            output, q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)  # already normalized

            # compute key features
            with torch.no_grad():  # no gradient to keys
                # shuffle for making use of BN
                im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

                _, k = self.encoder_k(im_k_)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)  # already normalized

                # undo shuffle
                k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

            l_pos_full = torch.einsum('nc,ck->nk', [q, q.t()])

            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits_size = logits.shape[0]
            # labels: positive key indicators
            labels_ce = torch.zeros(logits_size, dtype=torch.long).to(device)

            los_pos_new = l_pos_full[index_batch[:, 0],
                                     index_batch[:, 1]].unsqueeze(-1)
            los_neg_new = l_neg[index_batch[:, 0]]

            logits_new = torch.cat([los_pos_new, los_neg_new], dim=1)
            labels_ce_new = torch.zeros(logits_new.shape[0],
                                        dtype=torch.long).to(device)

            logits = torch.cat([logits, logits_new], dim=0)
            labels_ce = torch.cat([labels_ce, labels_ce_new], dim=0)

            # Add bank
            logits = torch.cat([logits, logits[duplicated_row_index]], dim=0)
            labels_ce = torch.cat([labels_ce, new_labels_ce], dim=0)

            # apply temperature
            logits /= self.T

            return logits, labels_ce, q, k, output

    def forward(self, im1, im2, labels=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2, output = self.contrastive_loss(im1, im2)
            loss_21, q2, k1, output = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            logits, labels_ce, q, k, output = self.contrastive_loss_inputs(
                im1, im2, labels=labels)

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_labels(labels)

        return logits, labels_ce, q, k, output


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    return tensor
