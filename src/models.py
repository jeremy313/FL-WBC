#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import copy



class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        self.feature_fc1 = None
        self.feature_fc2 = None

    def forward(self, x):
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        self.feature_fc1 = x.cpu().detach().numpy()
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        self.feature_fc2 = x.cpu().detach().numpy()
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

class CNNFeMnist_sim(nn.Module):
    def __init__(self, args):
        super(CNNFeMnist_sim, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(7*7*20, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)
        return F.log_softmax(out, dim=1)

class CNNFeMnist(nn.Module):
    def __init__(self, args):
        super(CNNFeMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(7*7*64, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)
        return F.log_softmax(out, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        self.feature_fc1 = None
        self.feature_fc2 = None
        self.feature_fc3 = None
        #self.feature_fc1_graph = None
        #self.feature_fc2_graph = None
        #self.feature_fc3_graph = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        #self.feature_fc1_graph = x
        self.feature_fc1 = x.cpu().detach().numpy()
        x = F.relu(self.fc1(x))
        #self.feature_fc2_graph = x
        self.feature_fc2 = x.cpu().detach().numpy()
        x = F.relu(self.fc2(x))
        #self.feature_fc3_graph = x
        self.feature_fc3 = x.cpu().detach().numpy()
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNORL(nn.Module):
    def __init__(self):
        super(CNNORL, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(8000, 1024)
        self.fc2 = nn.Linear(1024, 84)
        self.fc3 = nn.Linear(84, 40)
        self.feature_fc1 = None
        self.feature_fc2 = None
        self.feature_fc3 = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        f1_out = x.view(x.size(0), -1)
        self.feature_fc1 = f1_out.cpu().detach()
        f2_out = self.fc1(f1_out)
        self.feature_fc2 = f2_out.cpu().detach()
        f2 = F.relu(f2_out)
        f3_out = self.fc2(f2)
        self.feature_fc3 = f3_out.cpu().detach()
        f3 = F.relu(f3_out)
        out = self.fc3(f3)
        return F.log_softmax(out, dim=1), f1_out, f2_out, f3_out, F.softmax(out, dim=1)


class CNNMiniImagenet(nn.Module):
    def __init__(self, args):
        super(CNNMiniImagenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(10, 20, 3)
        #self.conv4 = nn.Conv2d(32, 32, 3)
        #self.fc1 = nn.Linear(10368, 4096)
        self.fc1 = nn.Linear(7220, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 100)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class LeNet(nn.Module):
    def __init__(self, channel):
        super(LeNet, self).__init__()
        act = nn.ReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        if channel == 1:
            self.fc = nn.Sequential(
                nn.Linear(588, 10),
                #act(),
                #nn.Linear(256, 100)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(768, 10),
                #act(),
                #nn.Linear(256, 100)
            )
        
    def forward(self, x):
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(feature)
        return F.log_softmax(out, dim=1)

class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),
            
            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten())
            #('linear', torch.nn.Linear(36 * width, num_classes))
        ]))
        self.linear = torch.nn.Linear(36 * width, num_classes)
        #self.feature = None

    def forward(self, input):
        feature = self.model(input)
        out = self.linear(feature)
        return F.log_softmax(out, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1) 
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
