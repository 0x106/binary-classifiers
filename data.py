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


class MNIST_all():
	def __init__(self, opt):

		self.B = opt.batch_size
		self.cuda = opt.cuda

		data_path = opt.dataroot
		if self.cuda:
			data_path = '/input'

		self.trData = dset.MNIST(data_path, train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.testData = dset.MNIST(data_path, train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.train_loader = torch.utils.data.DataLoader(self.trData, batch_size=self.B, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(self.testData, batch_size=len(self.testData), shuffle=True)

		self.train_iter = iter(self.train_loader)
		self.test_iter = iter(self.test_loader)

	def __len__(self):
		return len(self.train_loader)

	def next(self):
		try:
			output = self.train_iter.next()
		except:
			self.train_iter = iter(self.train_loader)
			output = self.train_iter.next()

		return output

	def next_test(self):
		try:
			output = self.test_iter.next()
		except:
			self.test_iter = iter(self.test_loader)
			output = self.test_iter.next()

		return output

class MNIST():

	def __init__(self, opt, class_=-1):

		self.B = opt.batch_size
		self.cuda = opt.cuda
		self.num_local_classes = 10
		self.C = self.B // self.num_local_classes

		data_path = opt.dataroot
		if self.cuda:
			data_path = '/input'

		self.trData = dset.MNIST(data_path, train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.testData = dset.MNIST(data_path, train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.dataset, self.targets = self.build_dataset(opt, class_)
		self.test_dataset, self.test_targets = self.build_dataset(opt, class_, train=False)

		self.pointer = -self.B
		self.test_pointer = -self.B
		self.N *= 2
		self.data_length = self.dataset.size(0) // self.B
		self.class_shuffle = torch.randperm(self.N)
		self.test_N = self.test_dataset.size(0) // self.B
		self.test_class_shuffle = torch.randperm(self.test_N)

	def __len__(self):
		return self.data_length

	# def __next__(self):
	def next(self):

		self.pointer += self.B

		if self.pointer + self.B >= self.N:
			self.class_shuffle = torch.randperm(self.N)
			self.pointer = 0

		return self.dataset[self.class_shuffle[self.pointer:self.pointer+self.B]], \
				self.targets[self.class_shuffle[self.pointer:self.pointer+self.B]]

		# return self.positive[self.class_shuffle[self.pointer:self.pointer+self.B]], \
		# 		self.negative[self.class_shuffle[self.pointer:self.pointer+self.B]]

	def next_test(self):
		return self.test_dataset, self.test_targets

	def build_dataset(self, opt, class_, train=True):

		if train:
			data = self.trData
		else:
			data = self.testData

		self.num_examples = len(data)
		self.labels = [ [] for i in range(self.num_local_classes) ]

		for i in range(self.num_examples):
			self.labels[data[i][1]].append(i)

		self.class_selector = class_
		self.N = len(self.labels[self.class_selector])
		self.data_length = self.N // self.B

		self.positive = torch.FloatTensor(self.N, 1, opt.input_size, opt.input_size)
		self.negative = torch.FloatTensor(self.N, 1, opt.input_size, opt.input_size)
		dataset = torch.FloatTensor(self.N*2, 1, opt.input_size, opt.input_size)
		targets = torch.FloatTensor(self.N*2, 1)

		for k in range(self.N):
			self.positive[k].copy_( data[ self.labels[self.class_selector][k] ][0] )

		training_size = self.N // (self.num_local_classes - 1)
		drop_in_shuffle, counter = np.random.permutation(self.N), 0
		for i in range(self.num_local_classes):
			if i != self.class_selector:
				random_selection = np.random.permutation(len(self.labels[i]))[:training_size]
				for k in range(training_size):
					self.negative[drop_in_shuffle[counter]].copy_( data[ self.labels[i][k] ][0] )
					counter += 1

		dataset[:self.N].copy_(self.positive)
		dataset[self.N:].copy_(self.negative)

		targets[:self.N].fill_(1)
		targets[self.N:].fill_(0)

		indices = torch.LongTensor(self.N*2).copy_(torch.from_numpy(np.random.permutation(self.N*2)))

		return dataset[indices], targets[indices]
