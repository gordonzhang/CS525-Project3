#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    
    This is just a hint. You can build your own structure.
    """
    def __init__(self, h, w, inputs, outputs):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ks1, ks2, ks3 = 3, 3, 3
        stride1, stride2, stride3 = 1, 1, 1
        # ks1, ks2, ks3 = 5,5,5
        # stride1, stride2, stride3 = 2,2,2
        self.conv1 = nn.Conv2d(inputs, 16, kernel_size=ks1, stride=stride1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=ks2, stride=stride2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=ks3, stride=stride3)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, ks1, stride1), ks2, stride2), ks3, stride3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, ks1, stride1), ks2, stride2), ks3, stride3)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x
