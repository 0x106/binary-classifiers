import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math, sys
import numpy as np

class Classifier(nn.Module):
	def __init__(self, input_size, feature_size, output_size=1):
		super(Classifier, self).__init__()

		self.main = nn.Sequential(
			nn.Linear(input_size * input_size, feature_size*16),
			nn.ReLU(True),
			nn.Linear(feature_size*16, feature_size*8),
			nn.ReLU(True),
			nn.Linear(feature_size*8, feature_size*4),
			nn.ReLU(True),
			nn.Linear(feature_size*4, feature_size*2),
			nn.ReLU(True),
			nn.Linear(feature_size*2, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, output_size),
			nn.Sigmoid()
		)

	def forward(self, x):
		classification = self.main(x.view(x.size(0), x.size(2)*x.size(3)))
		return classification
