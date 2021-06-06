import math
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import (adjusted_rand_score,
                                     normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture
from tqdm.notebook import tqdm

import dataset
import loss as lossfile
import train
import utils


###################################################################
############ Utils ################################################
###################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, p, mode, finetune_lr=False):
    """
    Get optimizer for dataset. It is based on SGD.

    Args:
        model (torch.nn.Module): Model
        p (dict: Dict of parameters
        mode (str): If it is representation of supervised
        finetune_lr (bool): If it is fine tuning

    Returns:
        [type]: [description]
    """
    if finetune_lr:
        nesterov = p['nesterov']
        if mode in ['representation', 'supervised']:
            mode = f"{mode}_{'fine'}"
        elif mode in ['supervised_classifier']:
            pass
        else:
            raise ValueError("Cannot use finetune lr in this mode")
    else:
        nesterov = False

    if mode in [
            'representation', 'supervised', 'supervised_classifier',
            'supervised_fine'
    ]:
        lr = p['lr'][mode]
        weight_decay = p['weight_decay'][mode]
    else:
        raise ValueError(f'{mode} is not defined')

    optimizer = torch.optim.SGD(filter(lambda q: q.requires_grad,
                                       model.parameters()),
                                lr=lr,
                                weight_decay=weight_decay,
                                momentum=0.9,
                                nesterov=nesterov)
    if p['adam'] is True:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-6)
    print(optimizer)
    return optimizer


# lr scheduler for training
def adjust_learning_rate(optimizer,
                         initial_lr,
                         epoch,
                         max_epoch=None,
                         cosinus=True,
                         schedule=[]):
    """Decay the learning rate based on schedule"""
    lr = initial_lr
    if cosinus:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


###################################################################
############ Supervised training ##################################
###################################################################
def train_supervised_one_epoch(loader,
                               model,
                               optimizer,
                               epoch,
                               criterion,
                               weights=None,
                               model_output={},
                               monitoring_time=False):
    """
    Supervised training with noise robust loss.

    Args:
        loader ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        epoch ([type]): [description]
        criterion ([type]): [description]
        weights ([type], optional): [description]. Defaults to None.
    """
    # switch to train mode
    model.train()
    device = utils.get_device()
    running_loss = 0

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for i, (input_tensor, target, index) in enumerate(tqdm(loader)):
        data_time.update(time.time() - end)

        #input_var = torch.autograd.Variable(input_tensor.to(device))
        input_var = input_tensor.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output, _ = model(input_var)
        if criterion.forward.__code__.co_argcount == 3:
            loss = criterion(output, target)
        else:
            loss = criterion(output, target, index, epoch)
        if weights is not None:
            loss = loss * weights[index]
            loss = loss.mean()
        optimizer.zero_grad()

        loss.backward()
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        running_loss += loss.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if monitoring_time:
            print('data time', data_time.val)
            print('batch time', batch_time.val - data_time.val)

    final_loss = running_loss / len(loader)
    model_output["train_losses"].append(final_loss)

    print('Train Epoch: [{}], lr: {:.6f}, Loss: {:.4f}'.format(
        epoch, optimizer.param_groups[0]['lr'], final_loss))


def eval_train_gmm(model,
                   dataloader,
                   device,
                   model_output,
                   gmm_true_labels=None,
                   penalty=False):
    """
    Evaluates model on train set with standard augmentation and compute
    the GMM on the loss.

    Args:
        model ([type]): [description]
        dataloader ([type]): [description]
        all_loss ([type]): [description]
        device ([type]): [description]
        return_labels (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    model.eval()
    losses = []
    CE = torch.nn.CrossEntropyLoss(reduction='none')
    conf_penalty = lossfile.NegEntropy(reduction='none')
    pred_labels = []
    targets = []
    with torch.no_grad():
        for i, (input_tensor, _, target) in enumerate(dataloader):
            index = np.arange(
                i * dataloader.batch_size,
                i * dataloader.batch_size + input_tensor.shape[0])
            input_var = torch.autograd.Variable(input_tensor.to(device))
            with torch.no_grad():
                vec, _ = model(input_var)
                targets.extend(target.data.cpu().numpy())
                _, indices = torch.max(vec, 1)
                pred_labels.extend(indices.data.cpu().numpy())
                # Negative Entropy - Penalty term
                #vec = torch.nn.Softmax(dim=1)(vec)
                if penalty:
                    if torch.isnan(conf_penalty(vec)).any():
                        raise ValueError('Nan in the penalty term')
                    loss_val = CE(vec, target.to(device)) + conf_penalty(vec)
                else:
                    loss_val = CE(vec, target.to(device))
                losses.extend(loss_val.cpu())

    targets = np.array(targets)
    train_acc = accuracy_score(targets, pred_labels)
    model_output["train_accs"].append(train_acc)

    losses = torch.stack(losses)
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    model_output["gmm_loss"].append(losses)
    pred_labels = np.array(pred_labels)
    if len(
            model_output["gmm_loss"]
    ) > 1:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(model_output["gmm_loss"])
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=100, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    idx = np.where(prob > 0.5)[0]
    model_output["pred_train"] = pred_labels

    if gmm_true_labels is not None:
        correct_preds = np.where((gmm_true_labels - pred_labels) == 0)[0]
        trustable_preds = np.where(prob > 0.5)[0]
        total_acc = round(accuracy_score(gmm_true_labels, pred_labels), 4)
        acc_subset = round(
            np.intersect1d(correct_preds, trustable_preds).shape[0] /
            max(len(trustable_preds), 1), 4)

        size_trustable = 100 * len(trustable_preds) / len(prob)
        print(
            f'GMM trustable ACC: {acc_subset} ( on {size_trustable} % of entire dataset)'
            + f' vs all pseudolabels ACC: {total_acc}')
        model_output["gmm_acc"].append(acc_subset)
        model_output["gmm_size"].append(size_trustable)

    return idx, prob


def test(loader, model, epoch, model_output):
    """
    Evaluate model performance on clean test.

    Args:
        loader ([type]): [description]
        model ([type]): [description]
        epoch ([type]): [description]
        return_labels (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    model.eval()
    # Forward and save predicted labels
    gnd_labels = []
    pred_labels = []
    device = utils.get_device()
    model.to(device)
    for i, (input_tensor, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input_tensor.to(device))
        with torch.no_grad():
            vec, _ = model(input_var.to(device))

        _, indices = torch.max(vec, 1)
        gnd_labels.extend(target.data.numpy())
        pred_labels.extend(indices.data.cpu().numpy())

    # Computing Evaluations
    gnd_labels = np.array(gnd_labels)
    pred_labels = np.array(pred_labels)

    acc = accuracy_score(gnd_labels, pred_labels)
    model_output["test_accs"].append(acc)
    model_output["pred_test"] = pred_labels

    # Logging
    print(f'Epoch: [{epoch}]\t Supervised test acc {round(acc, 4)}')

    return acc, pred_labels


def evaluate_validation(loader, model, model_output, rma=3):
    """
    Evaluate model performance on clean test.

    Args:
        loader ([type]): [description]
        model ([type]): [description]
        epoch ([type]): [description]
        return_labels (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    # Forward and save predicted labels
    gnd_labels = []
    pred_labels = []
    device = utils.get_device()
    val_loss = 0
    corrupted_loss = 0

    for i, (input_tensor, _, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input_tensor.to(device))
        idx = np.arange(i * loader.batch_size,
                        i * loader.batch_size + input_var.shape[0])
        with torch.no_grad():
            vec, _ = model(input_var)
            loss_val = criterion(vec, target.to(device))
            val_loss += loss_val.item()
        _, indices = torch.max(vec, 1)
        gnd_labels.extend(target.data.numpy())
        pred_labels.extend(indices.data.cpu().numpy())
    val_loss = val_loss / len(loader)

    # Computing Evaluations
    gnd_labels = np.array(gnd_labels)
    pred_labels = np.array(pred_labels)
    wrong_ids = np.where(model_output["true_val"] != pred_labels)[0]
    wrong_predictions = pred_labels[wrong_ids]
    if model_output["pred_val"] is None:
        model_output["val_nb_changed_pred"].append(np.nan)
        changed_ids = []
    else:
        changed_ids = np.where(model_output["pred_val"] != pred_labels)[0]
        model_output["val_nb_changed_pred"].append(len(changed_ids))

    acc = accuracy_score(gnd_labels, pred_labels)

    model_output["val_wrong_ids"].append(wrong_ids)
    model_output["val_wrong_predictions"].append(wrong_predictions)
    model_output["val_changed_ids"].append(changed_ids)
    model_output["val_losses"].append(val_loss)
    model_output["val_accs"].append(acc)
    model_output["pred_val"] = pred_labels
    # calculate rma only if we have rma values in the iteration to avoid sharp changes
    if len(model_output["val_changed_ids"]) < rma - 1 or np.any(
            np.isnan(model_output["val_nb_changed_pred"][-rma:])):
        model_output["val_rma_changed_pred"].append(np.nan)
    else:
        model_output["val_rma_changed_pred"].append(
            np.mean(model_output["val_nb_changed_pred"][-rma:]))
    print(
        f"Val Loss: {round(val_loss, 2)} \t val acc {round(acc,4)} \t nb changed predictions {len(changed_ids)}"
        + f" RMA {model_output['val_rma_changed_pred'][-1]}")

    return model_output


def supervised_training(model,
                        checkpoint_path_file,
                        loss_name,
                        label_data,
                        p,
                        epochs=100,
                        checkpoint_folder="../data/models/test",
                        dataset_name="cifar10",
                        name="",
                        nb_epochs_output_training=0,
                        finetune_lr=False,
                        early_stop={}):
    """
    Train entire model in a supervised way

    Args:
        model ([type]): [description]
        checkpoint_path_file ([type]): [description]
        criterion_sup ([type]): [description]
        labels ([type]): [description]
        p ([type]): [description]
        epochs (int, optional): [description]. Defaults to 100.
        checkpoint_folder (str, optional): [description]. Defaults to "../data/models/test".
        evaluate_every (int, optional): [description]. Defaults to 5.
        dataset_name (str, optional): [description]. Defaults to "cifar10".
        name (str, optional): [description]. Defaults to "".
        nb_epochs_output_training ([type], optional): [description]. Defaults to None.
        weights ([type], optional): [description]. Defaults to None.
        finetune_lr (bool, optional): [description]. Defaults to False.
        compute_soft_labels (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    original_name = name
    # Get classification loaders and pseudo labels (label_data["pred_train"])
    loaders = dataset.get_classification_loaders(p, label_data=label_data)
    monitoring_time = p['monitoring_time']
    gmm_penalty_term = p.get('gmm_negative_entropy', False)

    if "train_idx" not in label_data:
        label_data["train_idx"] = np.arange(len(loaders["train"].dataset))
    _, criterion_sup = utils.get_model_and_criterion(
        dataset_name,
        loss_name=loss_name,
        model="",
        num_samples=len(label_data["train_idx"]),
        p=p)

    device = utils.get_device()
    model_output = {
        "train_losses": [],
        "test_accs": [],
        "gmm_loss": [],
        "train_accs": [],
        "gmm_acc": [],
        "gmm_size": []
    }
    if p['use_validation']:
        model_output = {
            **model_output,
            **{
                "val_losses": [],
                "val_accs": [],
                "val_wrong_ids": [],
                "val_changed_ids": [],
                "val_rma_changed_pred": [],
                "val_nb_changed_pred": [],
                "val_wrong_predictions": [],
                "pred_val":
                None,
                "true_val":
                label_data["true_targets"][label_data["val_idx"]] if "true_targets" in label_data else None,
            }
        }

    if checkpoint_path_file is not None:
        print(f"Loading pretrained model")
        checkpoint = torch.load(checkpoint_path_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if (p['is_moco'] or p['is_mopro']
            ):  #and 'queue' not in checkpoint["model"].keys():
            # we are starting after the supervised phase where only encoder q was saved
            model.encoder_k.load_state_dict(checkpoint['model'], strict=False)
            model.encoder_q.load_state_dict(checkpoint['model'], strict=False)

    original_model = model

    clean_idx = None
    gmm_weights = None
    if "true_targets" in label_data:
        if "weights" in label_data:
            # select only train labels, prefiltered by weights
            gmm_true_labels = label_data["true_targets"][np.where(
                label_data['weights'] > 0.5)[0]]
        else:
            gmm_true_labels = label_data["true_targets"]
        if "train_idx" in label_data:  # use validation set
            gmm_true_labels = gmm_true_labels[label_data["train_idx"]]
    else:
        gmm_true_labels = None
    # Train only output layer
    if nb_epochs_output_training > 0:
        print("Training only classifier layer")
        original_model.set_phase("3")

        if p['is_moco'] or p['is_mopro']:
            model = original_model.encoder_q
        else:
            model = original_model

        optimizer = get_optimizer(model,
                                  p,
                                  "supervised_classifier",
                                  finetune_lr=finetune_lr)
        if p['cos'] is True:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=np.maximum(nb_epochs_output_training, 10),
                eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=1000,
                                                        gamma=0.1)
        for epoch in range(nb_epochs_output_training):
            acc, _ = test(loaders["test"], model, epoch, model_output)
            train_supervised_one_epoch(loaders["train"],
                                       model,
                                       optimizer,
                                       epoch,
                                       criterion_sup,
                                       model_output=model_output,
                                       monitoring_time=monitoring_time)
            scheduler.step()
            if p['use_validation']:
                evaluate_validation(loaders["val"], model, model_output)

            clean_idx, gmm_weights = eval_train_gmm(
                model,
                loaders["gmm"],
                device,
                model_output,
                gmm_true_labels=gmm_true_labels,
                penalty=gmm_penalty_term)
            model_output["classification_head_acc"] = acc
        print(f"Done pretraining")

    original_model = model

    # Train entire network
    original_model.set_phase("2")
    if hasattr(original_model, "encoder_q"):  #p['is_moco'] or p['is_mopro']:
        model = original_model.encoder_q
    else:
        model = original_model

    optimizer = get_optimizer(model, p, "supervised", finetune_lr=finetune_lr)
    if p['cos'] is True or epochs >= 100:
        print("Creating cosine scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=np.maximum(epochs, 10), eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=1000,
                                                    gamma=0.1)

    for epoch in range(epochs):
        train_supervised_one_epoch(loaders["train"],
                                   model,
                                   optimizer,
                                   epoch,
                                   criterion_sup,
                                   model_output=model_output,
                                   monitoring_time=monitoring_time)
        acc, _ = test(loaders["test"], model, epoch, model_output)

        scheduler.step()
        if p['use_validation']:
            evaluate_validation(loaders["val"], model, model_output)


#         if epoch > epochs - 10:
        clean_idx, gmm_weights = eval_train_gmm(
            model,
            loaders["gmm"],
            device,
            model_output,
            gmm_true_labels=gmm_true_labels,
            penalty=gmm_penalty_term,
        )
    checkpoint_path_file = utils.get_checkpoint_file(checkpoint_folder,
                                                     original_name)

    unsup_sample_ids = np.setdiff1d(
        np.arange(len(label_data["train_idx"])),
        clean_idx) if clean_idx is not None else None

    ##### PLOTTING RESULTS #####
    plot_results(model_output)

    torch.save(
        {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'weights': gmm_weights,
            'model_output': model_output
        }, checkpoint_path_file)

    result = {
        "checkpoint_path_file": checkpoint_path_file,
        "sup_sample_ids": label_data["train_idx"][clean_idx],
        "unsup_sample_ids": label_data["train_idx"][unsup_sample_ids],
        "acc": acc,
        'weights': gmm_weights,
        'label_data': label_data,
        'model_output': model_output,
    }
    return result


def plot_results(model_output):
    # Plot validation stats
    if 'val_changed_ids' in model_output and len(
            model_output["val_changed_ids"]) > 0:
        nb_wrong_ids = [
            len(x) if ~np.any(np.isnan(x)) else np.nan
            for x in model_output["val_wrong_ids"]
        ]
        plt.figure(figsize=(14, 4))
        ax = plt.subplot(121)
        plt.plot(model_output["val_losses"], label="val_losses", c="blue")
        plt.legend(bbox_to_anchor=(1.05, 0.9))
        ax.twinx()
        plt.plot(model_output["val_accs"], label="val_accs", c="green")
        plt.plot(model_output["test_accs"],
                 label="test_accs",
                 linestyle="--",
                 c="black")
        plt.legend(bbox_to_anchor=(1.05, 0.3))

        ax = plt.subplot(122)
        plt.plot(model_output["test_accs"],
                 label="test_accs",
                 linestyle="--",
                 c="black")
        plt.legend(bbox_to_anchor=(1.2, 0.9))
        ax.twinx()
        plt.plot(model_output["val_nb_changed_pred"],
                 label="Nb changed pred (val)",
                 c="blue")
        plt.plot(nb_wrong_ids, label="Nb wrong pred (val)", c="red")
        plt.legend(bbox_to_anchor=(1.2, 0.3))
        plt.tight_layout()
        plt.show()
        c = model_output["val_changed_ids"]
        common = [
            np.intersect1d(c[i], c[i + 1]).shape[0] if
            (~np.any(np.isnan(c[i]))
             and ~np.any(np.isnan(c[i + 1]))) else np.nan
            for i in range(len(c) - 1)
        ]
        perc_common = [
            np.intersect1d(c[i], c[i + 1]).shape[0] / len(c[i + 1]) if
            (~np.any(np.isnan(c[i])) and ~np.any(np.isnan(c[i + 1]))
             and len(c[i + 1]) != 0) else np.nan for i in range(len(c) - 1)
        ]

        plt.figure(figsize=(14, 4))
        ax = plt.subplot(121)
        plt.plot(common, c="blue", label="# common changed (i, i+1)")
        plt.plot(model_output["val_nb_changed_pred"],
                 c="red",
                 label="# changed pred")
        plt.legend(bbox_to_anchor=(1.6, 0.9))
        ax.twinx()
        plt.plot(perc_common,
                 c="green",
                 label="perc common (i, i+1)",
                 linestyle="--")
        plt.legend(bbox_to_anchor=(1.05, 0.3))

        ax = plt.subplot(122)
        plt.title("Rolling mean avg (2) of changed predictions")
        plt.plot(model_output["val_nb_changed_pred"],
                 label="# changed predictions",
                 c="blue",
                 alpha=0.3)
        plt.plot(model_output["val_rma_changed_pred"],
                 c="red",
                 label="# rma changed predictions")
        plt.legend(bbox_to_anchor=(0.1, 0.9))
        ax.twinx()
        plt.plot(model_output["test_accs"],
                 linestyle="--",
                 c="black",
                 label="Test Accuracy")
        plt.legend(bbox_to_anchor=(0.1, 0.5))
        plt.show()
    else:
        plt.figure(figsize=(14, 4))
        plt.title("Test accuracy vs train loss")
        plt.plot(model_output["test_accs"], label="Clean Test ACC", c="orange")
        if "train_accs" in model_output:
            plt.plot(model_output["train_accs"],
                     label="Corrupted Train ACC",
                     c="blue")

        plt.legend(bbox_to_anchor=(1.05, 0.3))
        plt.gca().twinx()
        plt.plot(model_output["train_losses"], label="train_losses", c="green")
        plt.legend(bbox_to_anchor=(1.05, 0.5))
        plt.show()


###################################################################
############ Representation learning ##############################
###################################################################
def moco_supervised_loss(model, logits, targets, weights):
    """
    Compute MOCO supervised loss for batch and bank
    """
    logits_index = model.current_queue_index.clone().detach()
    #     criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
    device = utils.get_device()
    batch_size = len(targets)

    targets = targets.view(-1, 1).to(device)
    index_batch = torch.eq(targets, targets.t()).float()
    index_batch = torch.scatter(
        index_batch, 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0)
    index_batch = torch.nonzero(index_batch)

    index_batch = torch.eq(targets, targets.t()).float()
    index_batch1 = torch.nonzero(index_batch)  # q*k
    index_batch1[:, 1] += batch_size
    index_batch2 = torch.scatter(
        index_batch, 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0)
    index_batch2 = torch.nonzero(index_batch2)  # q* q

    new_logits = logits[:, :2 * batch_size]
    new_logits_weights = weights[logits_index[:2 * batch_size]]
    loss1 = criterion_ce(new_logits[index_batch1[:, 0]], index_batch1[:, 1])
    loss1 = torch.mean(loss1 * new_logits_weights[index_batch1[:, 1]])

    loss2 = criterion_ce(new_logits[index_batch2[:, 0]], index_batch2[:, 1])
    loss2 = torch.mean(loss2 * new_logits_weights[index_batch2[:, 1]])
    loss = loss1 + loss2

    bank_logits = logits[:, 2 * batch_size:]
    bank_labels = model.queue_labels[:model.queue_ptr.detach()].detach()
    bank_weights = weights[logits_index[2 * batch_size:]]
    if len(bank_labels) > 0:
        positives = bank_labels.repeat(batch_size).view(
            batch_size, -1) - targets.view(-1).repeat(len(bank_labels)).view(
                len(bank_labels), -1).T
        index_bank = torch.where(positives == 0)
        loss3 = criterion_ce(bank_logits[index_bank[0]], index_bank[1])
        loss3 = torch.mean(loss3 * bank_weights[index_bank[1]])
        loss += loss3

    return loss


def representation_supervised(model, train_loader_sup, optimizer, p,
                              model_output, epoch, weights):

    device = utils.get_device()
    if p['is_moco'] == False:
        criterion_rep = lossfile.SupConLoss(temperature=0.07)
        criterion_rep = criterion_rep.to(device)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_ce.to(device)
    total_loss, total_num, running_loss = 0.0, 0, 0
    loader = train_loader_sup
    loader = tqdm(train_loader_sup)
    if p['is_mopro'] == False and p['is_moco'] == False:  # simclr
        #         criterion_rep = lossfile.SupConLoss(temperature=0.07)
        criterion_rep = lossfile.WeightedSupConLoss(temperature=0.07)
        criterion_rep = criterion_rep.to(device)
    for i, (anchors, neighbors, targets, index) in enumerate(loader):

        if anchors.shape[0] == 1:
            continue

        im_q, im_k, targets, index = anchors.to(
            device, non_blocking=True), neighbors.to(
                device, non_blocking=True), targets.to(
                    device, non_blocking=True), index.to(device,
                                                         non_blocking=True)

        if p['is_mopro']:
            model.set_phase("4")
            loss = 0.0
            cls_out, target, logits_moco, inst_labels, logits_proto = \
                model(im_q, im_k, target=targets.to(device), is_proto=(epoch > 0),
                      is_clean=(epoch >= p['start_clean_epoch']),
                      is_supervised = True,
                      sample_index = index)

            if epoch > 0:
                # prototypical contrastive loss
                loss_proto = criterion_ce(logits_proto, target)
                loss += p['w_proto'] * loss_proto

            # classification loss
            if p['mopro_use_ce']:
                loss_cls = criterion_ce(cls_out, target)
            else:
                loss_cls = 0.0

            # instance contrastive loss
            loss_inst = train.moco_supervised_loss(model, logits_moco, targets,
                                                   weights)

            loss += (loss_cls + p['w_inst'] * loss_inst)

        elif p['is_moco']:
            model.set_phase("1")
            logits, labels, _, __, ___ = model(im_q, im_k, targets, index)
            loss = train.moco_supervised_loss(model, logits, targets, weights)
        else:
            _, anchors_output = model(anchors.to(device))
            _, neighbors_output = model(neighbors.to(device))
            anchors_output = torch.nn.functional.normalize(anchors_output,
                                                           dim=1)
            neighbors_output = torch.nn.functional.normalize(neighbors_output,
                                                             dim=1)
            features = torch.cat(
                [anchors_output.unsqueeze(1),
                 neighbors_output.unsqueeze(1)],
                dim=1)
            loss = criterion_rep(features, targets, weights=weights[index])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_num += train_loader_sup.batch_size
        total_loss += loss.item() * train_loader_sup.batch_size
        if p['use_tqdm']:
            loader.set_description(
                'Train Epoch: [{}], lr: {:.6f}, Loss: {:.4f}'.format(
                    epoch, optimizer.param_groups[0]['lr'],
                    total_loss / total_num))

    epoch_loss = running_loss / len(train_loader_sup)
    print(f"Loss sup {epoch_loss}")
    model_output["train_losses"].append(epoch_loss)


def representation_unsupervised(model, train_loader_unsup, optimizer, p,
                                model_output, epoch):
    device = utils.get_device()
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_ce.to(device)
    total_loss, total_num = 0.0, 0
    loader = train_loader_unsup
    loader = tqdm(train_loader_unsup)
    if p['is_mopro'] == False and p['is_moco'] == False:  # simclr
        #criterion_rep = lossfile.SupConLoss(temperature=0.07)
        criterion_rep = lossfile.SimCLRLoss(temperature=0.5)
        print('Representation loss: SIMCLRLoss')
        criterion_rep = criterion_rep.to(device)

    for i, (anchors, neighbors, targets, index) in enumerate(loader):
        if anchors.shape[0] == 1:
            continue

        im_q, im_k = anchors.to(device, non_blocking=True), neighbors.to(
            device, non_blocking=True)

        if p['is_mopro']:
            model.set_phase("4")
            loss = 0.0
            cls_out, target, logits_moco, inst_labels, logits_proto = \
                model(im_q, im_k, target=targets.to(device), is_proto=(epoch > 0), is_clean=(epoch >= p['start_clean_epoch']))

            if epoch > 0:
                # prototypical contrastive loss
                loss_proto = criterion_ce(logits_proto, target)
                loss += p['w_proto'] * loss_proto

            # classification loss
            if p['mopro_use_ce']:
                loss_cls = criterion_ce(cls_out, target)
            else:
                loss_cls = 0.0

            # instance contrastive loss
            loss_inst = criterion_ce(logits_moco, inst_labels)

            loss += (loss_cls + p['w_inst'] * loss_inst)

        elif p['is_moco']:
            model.set_phase("1")
            logits, labels, _, __, ___ = model(im_q, im_k)
            loss = criterion_ce(logits, labels)
        else:  # simclr
            _, anchors_output = model(anchors.to(device))
            _, neighbors_output = model(neighbors.to(device))
            anchors_output = torch.nn.functional.normalize(anchors_output,
                                                           dim=1)
            neighbors_output = torch.nn.functional.normalize(neighbors_output,
                                                             dim=1)
            features = torch.cat(
                [anchors_output.unsqueeze(1),
                 neighbors_output.unsqueeze(1)],
                dim=1)
            loss = criterion_rep(features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += train_loader_unsup.batch_size
        total_loss += loss.item() * train_loader_unsup.batch_size
        loader.set_description(
            'Train Epoch: [{}], lr: {:.6f}, Loss: {:.4f}'.format(
                epoch, optimizer.param_groups[0]['lr'],
                total_loss / total_num))

    epoch_loss = total_loss / total_num
    print(f"Loss unsup {epoch_loss}")
    model_output["train_losses"].append(epoch_loss)


def representation_training(model,
                            checkpoint_path_file,
                            label_data,
                            p,
                            epochs=100,
                            checkpoint_folder="../data/models/test",
                            name="",
                            early_stop={}):
    """Performs unsupervised and supervised representation training

    Args:
        model ([type]): [description]
        checkpoint_path_file ([type]): [description]
        label_data ([type]): [description]
        p ([type]): [description]
        epochs (int, optional): [description]. Defaults to 100.
        checkpoint_folder (str, optional): [description]. Defaults to "../data/models/test".
        name (str, optional): [description]. Defaults to "".
        early_stop (dict, optional): [description]. Defaults to {}.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    device = utils.get_device()
    original_name = name
    loaders = dataset.get_representation_loaders(p, label_data)
    model_output = {"train_losses": [], "test_accs": [], "faiss_test_accs": []}

    if checkpoint_path_file is not None:
        print(f"Loading pretrained model")
        checkpoint = torch.load(checkpoint_path_file, map_location='cpu')
        if (p['is_moco'] or
                p['is_mopro']) and 'queue' not in checkpoint["model"].keys():
            # we are starting after the supervised phase where only encoder q was saved
            model.encoder_k.load_state_dict(checkpoint['model'], strict=False)
            model.encoder_q.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint['model'], strict=False)

    model.set_phase("1")
    optimizer = get_optimizer(model, p, "representation", finetune_lr=False)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_ce.to(device)

    initial_lr = optimizer.param_groups[0]['lr']
    current_early_stop = early_stop.copy()
    if p["is_mopro"] or p["is_moco"]:
        model_instance = model.encoder_q
    else:
        model_instance = model

    for epoch in range(epochs):
        model.train()
        if p['is_mopro'] or p['is_moco'] or p[
                'cos']:  # don't adjust lr for simclr
            adjust_learning_rate(optimizer,
                                 initial_lr,
                                 epoch,
                                 max_epoch=epochs,
                                 cosinus=True)
        elif p['schedule']:
            if isinstance(p['schedule'], list) == False:
                raise ValueError(f'The schedule must be a list')
            adjust_learning_rate(optimizer,
                                 initial_lr,
                                 epoch,
                                 max_epoch=epochs,
                                 cosinus=False,
                                 schedule=p['schedule'])
        # 1. Supervised contrastive loss
        if loaders["train_loader_sup"] is not None:
            weights = (label_data["weights"].to(device)
                       if "weights" in label_data else
                       torch.ones(len(
                           loaders["train_loader_sup"].dataset)).to(device))
            representation_supervised(model, loaders["train_loader_sup"],
                                      optimizer, p, model_output, epoch,
                                      weights)

        # 2. Unupervised contrastive loss
        if loaders["train_loader_unsup"] is not None:
            representation_unsupervised(model, loaders["train_loader_unsup"],
                                        optimizer, p, model_output, epoch)

        # compute classification accuracy when running original MOPRO
        if p['mopro_use_ce']:
            acc, _ = test(loaders["test"], model_instance, epoch, model_output)
        else:  # compute KNN representation
            if loaders["memory"] is None or loaders["test"] is None:
                pass
            else:
                test_moco(model_instance,
                          loaders["memory"],
                          loaders["test"],
                          epoch,
                          epochs,
                          p=p,
                          model_output=model_output,
                          device=device)
    checkpoint_path_file = utils.get_checkpoint_file(checkpoint_folder,
                                                     original_name)
    plot_results(model_output)
    torch.save(
        {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch + 1,
            'label_data': label_data,
            'model_output': model_output
        }, checkpoint_path_file)

    result = {
        "checkpoint_path_file": checkpoint_path_file,
        'acc': model_output["test_accs"][-1],
        'label_data': label_data,
        'model_output': model_output
    }
    return result


# test using a knn monitor
def test_moco(net,
              memory_data_loader,
              test_data_loader,
              epoch,
              epochs,
              knn_k=200,
              knn_t=0.1,
              p={},
              model_output={},
              device=None):
    net.eval()
    classes = p["nb_classes"]  #len(memory_data_loader.dataset.classes)
    device = device if device is not None else 'cpu'

    if p["dataset_name"] == "webvision":
        remove_diag = True
    else:
        remove_diag = False
    #remove_diag = True

    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader,
                                 desc='Feature extracting'):
            feature = net.get_features(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets,
                                      device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for counter_test, (data, target) in enumerate(test_bar):
            data, target = data.cuda(non_blocking=True), target.cuda(
                non_blocking=True)
            feature = net.get_features(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature,
                                      feature_bank,
                                      feature_labels,
                                      classes,
                                      knn_k,
                                      knn_t,
                                      counter_test,
                                      remove_diag=remove_diag,
                                      device=device)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(
                    epoch, epochs, total_top1 / total_num * 100))

        print('Moco accuracy', total_top1 / total_num * 100)
        model_output["test_accs"].append(total_top1 / total_num * 100)


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature,
                feature_bank,
                feature_labels,
                classes,
                knn_k,
                knn_t,
                counter_test,
                remove_diag=False,
                device=None):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    if remove_diag:
        batch_size = sim_matrix.shape[0]
        start_shift = counter_test * batch_size
        end_shift = start_shift + batch_size
        index_filter = torch.arange(start_shift, end_shift).view(-1,
                                                                 1).to(device)
        filter_diag = torch.scatter(
            torch.ones_like(sim_matrix).to(device), 1, index_filter, 0)
        sim_matrix = sim_matrix * filter_diag

    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1,
                              index=sim_indices)

    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) *
                            sim_weight.unsqueeze(dim=-1),
                            dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
