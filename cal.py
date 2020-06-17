# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
from densenet import DenseNet3

# CUDA_DEVICE = 0

start = time.time()
# loading data sets

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
# ])

# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
# nnName = "densenet10"

# imName = "Imagenet"


criterion = nn.CrossEntropyLoss()

def DenseNetBC_50_12():
    return DenseNet3(depth=50, num_classes=2, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.2)

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    model = DenseNetBC_50_12()
    model.load_state_dict(torch.load("../models/{}.pth".format(nnName)))
    optimizer1 = optim.SGD(model.parameters(), lr=0, momentum=0)
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.cuda(CUDA_DEVICE)

    transform_test = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    testsetout = torchvision.datasets.ImageFolder("/home/yoon/jyk416/odin-pytorch/data/{}".format(dataName),
                                                  transform=transform_test)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1, shuffle=False, num_workers=2)

    # if dataName != "Uniform" and dataName != "Gaussian":
    #     testsetout = torchvision.datasets.ImageFolder("../data/{}".format(dataName), transform=transform)
    #     testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
    #                                                 shuffle=False, num_workers=2)

    train_test_dir = '/home/yoon/jyk416/odin-pytorch/data/train3'
    if nnName == "model99":
        testset = torchvision.datasets.ImageFolder(train_test_dir, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    # if nnName == "densenet10" or nnName == "wideresnet10":
    #     testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    #     testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                                shuffle=False, num_workers=2)
    # if nnName == "densenet100" or nnName == "wideresnet100":
    #     testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    #     testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                                shuffle=False, num_workers=2)


    # if dataName == "Gaussian":
    #     d.testGaussian(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
    #     m.metric(nnName, dataName)
    #
    # elif dataName == "Uniform":
    #     d.testUni(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
    #     m.metric(nnName, dataName)
    # else:
    #     d.testData(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature)
    #     m.metric(nnName, dataName)

    d.testData(model, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, epsilon, temperature)
    m.metric(nnName, dataName)
