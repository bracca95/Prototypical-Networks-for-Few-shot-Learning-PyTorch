# coding=utf-8
from src.prototypical_batch_sampler import PrototypicalBatchSampler, DefectViewsSampler
from src.prototypical_loss import prototypical_loss as loss_fn
from src.omniglot_dataset import OmniglotDataset
from src.protonet import ProtoNet
from src.parser_util import get_parser
from src.tools import Utils, Logger
from src.custom_datasets import NoBreaks
from config.consts import Consts as _C

from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

import numpy as np
import torch
import sys
import os


def init_seed(args):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)


def init_dataset(args, mode):
    dataset = OmniglotDataset(mode=mode, root=args.dataset_root)
    n_classes = len(np.unique(dataset.y))
    if n_classes < args.classes_per_it_tr or n_classes < args.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_[tr/val] option and try again.'))
    return dataset


def init_sampler(args, labels, mode):
    if 'train' in mode:
        classes_per_it = args.classes_per_it_tr
        num_samples = args.num_support_tr + args.num_query_tr
    else:
        classes_per_it = args.classes_per_it_val
        num_samples = args.num_support_val + args.num_query_val

    # return PrototypicalBatchSampler(labels=labels, classes_per_it=classes_per_it, num_samples=num_samples, iterations=args.iterations)
    return DefectViewsSampler(labels=labels, classes_per_it=classes_per_it, num_samples=num_samples, iterations=args.iterations)


def init_dataloader(args, mode):
    dataset = init_dataset(args, mode)
    sampler = init_sampler(args, dataset.y, mode)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(args):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(args, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)


def init_lr_scheduler(args, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=args.lr_scheduler_gamma,
                                           step_size=args.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write(f"{item}\n")


def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(args.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(args.experiment_root, 'last_model.pth')

    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, n_support=args.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-args.iterations:])
        avg_acc = np.mean(train_acc[-args.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=args.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-args.iterations:])
        avg_acc = np.mean(val_acc[-args.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(args.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(args, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    avg_acc = list()
    for epoch in tqdm(range(10)):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=args.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(args):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(args.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(args=options,
         test_dataloader=test_dataloader,
         model=model)


if __name__ == '__main__':
    args = get_parser().parse_args()
    init_seed(args)

    while not os.path.exists(args.dataset_root):
        args.dataset_root = input("Insert dataset path: ")

    if not os.path.exists(args.experiment_root):
        os.makedirs(args.experiment_root)
    
    # compute mean and variance for dataset normalization
    dataset = NoBreaks(args.dataset_root, args.crop_size)
    if _C.DATASET_MEAN is None and _C.DATASET_STD is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        NoBreaks.compute_mean_std(dataset)
        sys.exit(0)

    # train and test datasets
    train_test_split = int(len(dataset)*0.8)
    trainset, testset = random_split(dataset, [train_test_split, len(dataset) - train_test_split])

    # extra info needed
    train_label_list = [dataset[idx][1] for idx in trainset.indices]
    test_label_list = [dataset[idx][1] for idx in testset.indices]

    Logger.instance().debug(f"samples per class: { {dataset.idx_to_label[i]: train_label_list.count(i) for i in set(train_label_list)} }")
    Logger.instance().debug(f"samples per class: { {dataset.idx_to_label[i]: test_label_list.count(i) for i in set(test_label_list)} }")
    
    train_sampler = init_sampler(args, train_label_list, 'train')
    test_sampler = init_sampler(args, test_label_list, 'test')
    
    trainloader = DataLoader(trainset, batch_sampler=train_sampler)
    testloader = DataLoader(testset, batch_sampler=test_sampler)

    model = init_protonet(args)
    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)

    pre_trained_model_path = os.path.join(args.experiment_root, "best_model.pth")
    if not os.path.exists(pre_trained_model_path):
        Logger.instance().debug("No model found, training mode ON!")
        res = train(args=args,
                    tr_dataloader=trainloader,
                    val_dataloader=trainloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
        
        best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    else:
        Logger.instance().debug("Model found! Testing on your dataset!")
        model.load_state_dict(torch.load(pre_trained_model_path))
        test(args=args, test_dataloader=testloader, model=model)