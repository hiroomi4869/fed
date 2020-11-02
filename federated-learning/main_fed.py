#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

import numpy as np
from torchvision import datasets, transforms
import torch

import time
from utils.options import args_parser
from PIL import Image
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, AlexNet, AlexNet28, ResNet, LeNet, VGG
from models.Fed import FedAvg
from models.test import test_img

def read(path, type):
    img = []
    img = np.asarray(img)

    for jpgfile in glob.glob(path+"/*."+type):
        temp = Image.open(jpgfile)
        temp = np.asarray(temp)
        img = np.append(img,temp)
    return img

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users=100):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.zeros([len(dataset)],dtype='int64')

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 100, 44
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.train_labels,dtype='int64')

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        # print(len(idx_shard))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # print(dict_users)
    return dict_users

import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])


#定义自己的test集合
class FlameSetTrain(data.Dataset):
    def __init__(self, folder, transform=None):
        folds = os.listdir(folder)

        x = [os.path.join(folder,folds[0]),os.path.join(folder,folds[1]),os.path.join(folder,folds[2])] #,os.path.join(folder,folds[2])
        self.train_image_file_paths = []
        y = np.array([], dtype='float')
        self.train_labels = y
        for image_files in x:
            for image_file in os.listdir(image_files):
                # print(image_files)
                # print(image_file)
                if image_files.split((os.path.sep))[1] == 'normal':
                    self.train_labels=np.append(self.train_labels, 0)
                elif image_files.split((os.path.sep))[1] == 'corona':
                    self.train_labels=np.append(self.train_labels, 1)
                elif image_files.split((os.path.sep))[1] == 'gan':
                    self.train_labels=np.append(self.train_labels, 2)
                elif image_files.split((os.path.sep))[1] == 'pneumonia':
                    self.train_labels=np.append(self.train_labels, 3)
                a = image_files + '/' + image_file
                self.train_image_file_paths.append(a)
        # print(self.train_image_file_paths)
        # print(os.listdir(folder))
        UNIT_SIZE = 28  # 每张图片的宽度是固定的
        size = (14, UNIT_SIZE)
        self.transform = transforms.Compose([
                                             # 转化为pytorch中的tensor

                                            transforms.Resize(28),
                                            transforms.CenterCrop(28),
                                            transforms.RandomRotation(30, resample=Image.BICUBIC, expand=False, center=(14, 14)),
                                            # transforms.FiveCrop(size=size),
                                            transforms.ToTensor(),
                                            # transforms.Lambda(lambda x: x.repeat(1,1,1)), # 由于图片是单通道的，所以重叠三张图像，获得一个三通道的数据
                                            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                            ]) # 主要改这个地方


    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root).convert('L')
        # print(type(image))
        # image = np.asarray(image,dtype='float32')
        # image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        if image_root.split((os.path.sep))[1]=='normal':
            label = 0
        elif image_root.split((os.path.sep))[1]=='corona':
            label = 1
        elif image_root.split((os.path.sep))[1] == 'gan':
            label = 2
        elif image_root.split((os.path.sep))[1] == 'pneumonia':
            label = 3
        # print(image_name,label)
        return image, label

class FlameSet(data.Dataset):
    def __init__(self, folder, transform=None):
        folds = os.listdir(folder)

        x = [os.path.join(folder,folds[0]),os.path.join(folder,folds[1]),os.path.join(folder,folds[2])] #,os.path.join(folder,folds[2])

        y=np.array([], dtype='float')
        self.train_labels = y
        self.train_image_file_paths = []
        print(self.train_labels)
        i = 0
        for image_files in x:
            for image_file in os.listdir(image_files):
                # print(image_files)
                # print(image_file)

                if image_files.split((os.path.sep))[1] == 'normal':
                    self.train_labels=np.append(self.train_labels, 0)
                elif image_files.split((os.path.sep))[1] == 'corona':
                    self.train_labels=np.append(self.train_labels, 1)
                elif image_files.split((os.path.sep))[1] == 'gan':
                    self.train_labels=np.append(self.train_labels, 2)
                elif image_files.split((os.path.sep))[1] == 'pneumonia':
                    self.train_labels=np.append(self.train_labels, 3)
                a = image_files + '/' + image_file
                self.train_image_file_paths.append(a)
                i = i + 1
        print(i)
        print(self.train_labels)
        # print(self.train_image_file_paths)
        # print(os.listdir(folder))
        self.transform = transforms.Compose([
                                             # 转化为pytorch中的tensor
                                            transforms.Resize(28),
                                            transforms.CenterCrop(28),
                                            transforms.ToTensor(),
                                            # transforms.Lambda(lambda x: x.repeat(1,1,1)), # 由于图片是单通道的，所以重叠三张图像，获得一个三通道的数据
                                            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                            ]) # 主要改这个地方


    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        if image_root.split((os.path.sep))[1]=='normal':
            label = 0
        elif image_root.split((os.path.sep))[1]=='corona':
            label = 1
        elif image_root.split((os.path.sep))[1] == 'gan':
            label = 2
        elif image_root.split((os.path.sep))[1] == 'pneumonia':
            label = 3
        # print(image_name,label)
        return image, label


def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # args.device = torch.device('cpu')

    # load dataset and split users
    #if args.dataset == 'mnist':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        # dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # dataset_train = FlameSet('/home/hiroomi/下载/PPGANs-Privacy-preserving-GANs-master/PPGANS/train/')
        # transform = transforms.Compose([transforms.Resize(28),
        #                               transforms.CenterCrop(28),
                                        # transforms.RandomHorizontalFlip(p=0.5),
                                        # transforms.RandomApply(random_transforms, p=0.3),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset_train = datasets.ImageFolder('/home/hiroomi/下载/PPGANs-Privacy-preserving-GANs-master/PPGANS/train/',
        #                               transform=transform)
    data = 'imgdatagan'
    dataset_train = FlameSetTrain(data)
    dataset_test = FlameSet('test')
        # sample users
        #if args.iid:
            # dataset_train = read('/home/hiroomi/下载/PPGANs-Privacy-preserving-GANs-master/PPGANS/test/', 'jpeg')
    num = 100

    if args.iid:
        dict_users = mnist_iid(dataset_train, num)
    else:
        dict_users = noniid(dataset_train, num)
    print(dict_users)
        #else:
        #    dict_users = mnist_noniid(dataset_train, args.num_users)
    '''
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('/home/hiroomi/下载/PPGANs-Privacy-preserving-GANs-master/PPGANS/train/', train=True, download=False, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    '''
    print(dataset_train[0][0].shape)
    img_size = dataset_train[0][0].shape

    print(dataset_train)
    print(img_size)



    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        print('gooooooooooooooooooooooooooooooooooooooood')
    elif args.model == 'mlp':
        print('yeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeear')
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'alexnet':
        net_glob = AlexNet28().to(args.device)
    elif args.model == 'resnet':
        net_glob = ResNet().to(args.device)
    elif args.model == 'lenet':
        net_glob = LeNet().to(args.device)
    elif args.model == 'vgg16':
        net_glob = VGG().to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    acctrain = [0.0]
    acctest = [0.0]
    accstring =['ACCstring']

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        if iter % 1 == 0:
            acc_train, _ = test_img(net_glob, dataset_train, args)
            acc_test, _ = test_img(net_glob, dataset_test, args)

            temp = "{:.2f}".format(acc_train)
            acctrain.append(float(temp))
            temp = "{:.2f}".format(acc_test)
            acctest.append(float(temp))
            if iter % 50 == 0:
                trainString = "Round {:3d} Training accuracy: {:.2f}".format(iter, acc_train)
                testString = "Round {:3d} Testing accuracy: {:.2f}".format(iter, acc_test)
                print(trainString)
                print(testString)
                accstring.append(trainString)
                accstring.append(testString)


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    print(range(len(loss_train)), loss_train)
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    net_glob.eval()
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_time{}_acctrain{:.2f}_acctest{:.2f}_datasat:{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, time.time(), acc_train, acc_test, data))
    plt.clf()

    plt.figure()
    plt.plot(range(len(acctest)), acctest)
    net_glob.eval()
    plt.ylabel('Test_acc')
    plt.title('Test_acc of {} on {} iid {}'.format(args.model, data, args.iid))
    plt.savefig('./save/Test_acc of {}_on {}_iid {}.png'.format(args.model, data, args.iid))

    # testing
    print('Dataset {}, iid {}, Model {}'.format(data, args.iid, args.model))
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print(accstring)
    print(acctrain)
    print(acctest)


