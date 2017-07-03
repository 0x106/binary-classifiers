from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np
import os, sys, math

import mlp
import data as data_

import matplotlib.pyplot as plt
plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--input_size', type=int, default=28)
parser.add_argument('--feature_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--cuda', type=bool, default=False)
opt = parser.parse_args()

opt.dataroot = '/helix/sandbox/binary-classifiers/data'
opt.experiment = '/helix/sandbox/binary-classifiers/output/samples'

if opt.cuda:
    opt.experiment = '/output/samples'

os.system('mkdir {0}'.format(opt.experiment))
print(opt)

# dataloader = data_.MNIST(opt, class_=2)
# classifier = mlp.Classifier(opt.input_size, opt.feature_size)
# criterion = nn.BCELoss()
# optimiser = optim.Adam(classifier.parameters(), lr=opt.lr)

def train_as_binary():
    num_classes = 1
    dataloader = [data_.MNIST(opt, class_=i) for i in range(num_classes)]
    classifier = [mlp.Classifier(opt.input_size, opt.feature_size) for i in range(num_classes)]
    criterion = [nn.BCELoss() for i in range(num_classes)]
    optimiser = [optim.Adam(classifier[i].parameters(), lr=opt.lr) for i in range(num_classes)]
    logs = [[] for i in range(num_classes)]

    for i in range(num_classes):
        for epoch in range(opt.epochs):
            for idx in range(len(dataloader[i])):

                classifier[i].zero_grad()

                dataset, targets = dataloader[i].next()
                output = classifier[i](Variable(dataset))
                loss = criterion[i](output, Variable(targets))

                loss.backward()
                optimiser[i].step()

            test_dataset, test_targets = dataloader[i].next_test()
            test_output = classifier[i](Variable(test_dataset))
            loss = criterion[i](test_output, Variable(test_targets))

            pred = torch.round(test_output.data)
            positive_correct = pred.eq(test_targets) * test_targets.byte()
            positive_correct = float(positive_correct.sum()) / (test_targets.sum())

            negative_correct = pred.eq(test_targets) * (1-test_targets).byte()
            negative_correct = float(negative_correct.sum()) / (test_targets.sum())
            # correct = float(pred.eq(test_targets).cpu().sum()) / test_output.size(0)

            logs[i].append(positive_correct)

            print("{0} - {1} - {2}".format(epoch, logs[i][-1], negative_correct))

            for k in range(len(logs)):
                plt.plot(logs[k])
            plt.pause(0.01)
            plt.clf()

    for k in range(len(logs)):
        plt.plot(logs[k])
    plt.pause(1000)


def train_standard():

    num_classes = 10
    logs = []#[[] for i in range(num_classes)]

    dataloader = data_.MNIST_all(opt)

    opt.feature_size *= 10
    opt.output_size = 10
    classifier = mlp.Classifier(opt.input_size, opt.feature_size, opt.output_size)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(classifier.parameters(), lr=opt.lr)

    fname = "/helix/sandbox/binary-classifiers/output/logs.txt"

    for epoch in range(opt.epochs):
        for idx in range(len(dataloader)):

            classifier.zero_grad()

            dataset, targets = dataloader.next()
            output = classifier(Variable(dataset))
            loss = criterion(output, Variable(targets))

            loss.backward()
            optimiser.step()

        test_dataset, test_targets = dataloader.next_test()
        test_output = classifier(Variable(test_dataset))
        loss = criterion(test_output, Variable(test_targets))

        pred = test_output.data.max(1)[1] # index of largest prob

        correct = []
        for i in range(1,num_classes+1):
            a = (torch.LongTensor(test_targets.size(0),1).fill_(i))     # labels
            b = ((test_targets.long() + 1).eq(a)).long() * i            # targets == labels?
            c = (pred.long() + 1).eq(b)                                 # true positives
            correct.append(float(c.sum()) / (b.sum()/(i)))

        logs.append(correct)

        with open(fname, 'a+') as f:
            for i in range(len(logs[-1])):
                f.write(str(logs[-1][i]) + " ")
            f.write("\n")

        # correct = float(pred.eq(test_targets).cpu().sum()) / test_output.size(0)
        # logs[i].append(correct)
        # print("{0} - {1}".format(epoch, logs[i][-1]))

        plt.plot(logs[-40:])
        plt.pause(0.01)
        plt.clf()

if __name__ == '__main__':

    # train_as_binary()
    train_standard()

#
