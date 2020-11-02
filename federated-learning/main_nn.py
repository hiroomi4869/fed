#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar
import glob

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
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
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

def readimage():
    batch_size = 32
    random_transforms = [
        # transforms.ColorJitter(brightness=0.75, contrast=0.75, saturation=0.75, hue=0.51),
        transforms.RandomRotation(degrees=5)]
    transform = transforms.Compose([transforms.Resize(28),
                                    transforms.CenterCrop(28),
                                    # transforms.RandomHorizontalFlip(p=0.5),
                                    # transforms.RandomApply(random_transforms, p=0.3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder('/home/hiroomi/下载/PPGANs-Privacy-preserving-GANs-master/PPGANS/train/',
                                      transform=transform)
    print(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False,
                                               batch_size=batch_size, num_workers=4)

    # imgs, label = next(iter(train_loader))
    # imgs = imgs.numpy().transpose(0, 2, 3, 1)
    # img_size = train_data[0][0].shape

    # print(img_size)
    return train_data

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


#定义自己的数据集合
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

def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss, (100. * correct / len(data_loader.dataset))


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        '''
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        img_size = dataset_train[0][0].shape
        '''
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    mydataset = 'imgdatagan'
    dataset_train = FlameSetTrain(mydataset)
    img_size = dataset_train[0][0].shape
    # build model
    if args.dataset == 'mnist':
        '''
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
        '''
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')
    dataset_test = FlameSet('test')
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)



    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    accstring = ['ACCstring']
    acctest = [0.0]
    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
        if epoch % 1 == 0:
            _, _, acc_train = test(net_glob, train_loader)
            _, _, acc_test = test(net_glob, test_loader)
            acctest.append(acc_test)
            if epoch %50 == 0:
                trainString = "Round {:3d} Training accuracy: {:.2f}".format(epoch, acc_train)
                testString = "Round {:3d} Testing accuracy: {:.2f}".format(epoch, acc_test)
                accstring.append(trainString)
                accstring.append(testString)

    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('nn/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    plt.clf()

    plt.figure()
    plt.plot(range(len(acctest)), acctest)
    net_glob.eval()
    plt.ylabel('Test_acc')
    plt.title('Test_acc of {}_{}epoch'.format(args.model,args.epochs))
    plt.savefig('./save/Test_acc of {}.png'.format(args.model))
    # testing


    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss, _ = test(net_glob, test_loader)
    print('Dataset {}, Model {}'.format(mydataset, args.model))
    print(accstring)

