#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, Y=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.mal=False
        if Y is not None:
            self.mal = True
            self.mal_Y = Y

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.mal==True:
            label_mal = self.mal_Y[item]
            return torch.tensor(image), torch.tensor(label_mal), torch.tensor(label)
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, mal, mal_X, mal_Y, dataset_test, idxs_test=None):
        self.args = args
        self.logger = logger
        self.mal = mal
        if args.dataset == 'femnist' or args.dataset == 'cifar10_extr_noniid' or args.dataset == 'miniimagenet_extr_noniid' or args.dataset == 'mnist_extr_noniid' or args.dataset == 'HAR' or args.dataset == 'HAD':
            if dataset_test is None or idxs_test is None:
                print('error: femnist and cifar10_extr_noniid need dataset_test and idx_test in LocalUpdate!\n')
            self.trainloader, self.validloader, self.testloader = self.train_val_test_femnist(dataset, list(idxs), dataset_test, list(idxs_test))
        else:
            self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        if mal is True:
            self.malloader = self.mal_loader(dataset_test, mal_X, mal_Y)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        #idxs_int = [int(i) for i in list(idxs)]
        #self.label_list = list(set(np.array(dataset.targets)[idxs_int]))

    def train_val_test_femnist(self, dataset, idxs, dataset_test, idxs_test):
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):]
        idxs_test = idxs_test

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train, self.mal, self.mal_target),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val, self.mal, self.mal_target),
                                 batch_size=int(len(idxs_val)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset_test, idxs_test, self.mal, self.mal_target),
                                batch_size=40, shuffle=True)
        return trainloader, validloader, testloader    


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    def mal_loader(self, dataset, idxs, Y):
        malloader = DataLoader(DatasetSplit(dataset, idxs, Y),
                               batch_size=self.args.local_bs, shuffle=True)
        return malloader

    def update_weights(self, model, global_round, device):
        EPS = 1e-6
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
            #psu_optimizer = torch.optim.SGD(psu_model.parameters(), lr=self.args.lr,
            #                            momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
            #psu_optimizer = torch.optim.Adam(psu_model.parameters(), lr=self.args.lr,
            #                             weight_decay=1e-4)
        if self.mal is True:
            for iter in range(self.args.local_mal_ep):
                '''
                # benign train, you can uncomment this to optimize the benign task on malicious devices
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
 
                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    
                    optimizer.step()
                '''
                batch_loss = []
                for batch_idx, (images, labels, _) in enumerate(self.malloader):
                    images, labels = images.to(self.device), labels.to(self.device)
 
                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)

                    loss.backward()

                    optimizer.step()
                    
                    
                    
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        else:
            for iter in range(self.args.local_ep):
                batch_loss = []
                old_gradient = {}
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
 
                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    
                    optimizer.step()

                    if self.args.defense == "WBC":
                        if batch_idx != 0:
                            for name, p in model.named_parameters():
                                if 'weight' in name:
                                    grad_tensor = p.grad.data.cpu().numpy()
                                    grad_diff = grad_tensor - old_gradient[name]
                                    pertubation = np.random.laplace(0, self.args.pert_strength, size=grad_tensor.shape).astype(np.float32)
                                    pertubation = np.where(abs(grad_diff) > abs(pertubation), 0, pertubation)
                                    p.data = torch.from_numpy(p.data.cpu().numpy()+pertubation*self.args.lr).to(device)
                        for name, p in model.named_parameters():
                            if 'weight' in name:
                                old_gradient[name] = p.grad.data.cpu().numpy()

                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def mal_inference(self, model):
        """ Returns the malicious inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels, _) in enumerate(self.malloader):

            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

           
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
        accuracy = correct/total
        return accuracy, loss


    
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
           
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def mal_inference(args, model, test_dataset, mal_X_list, mal_Y):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct, confidence_sum = 0.0, 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    malloader = DataLoader(DatasetSplit(test_dataset, mal_X_list, mal_Y),
                           batch_size=args.local_bs, shuffle=True)

    for batch_idx, (images, labels, labels_true) in enumerate(malloader):
        images, labels, labels_true = images.to(device), labels.to(device), labels_true.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        label_list = []
        idx_list = []
        for i in range(len(labels)):
            idx_list.append(int(i))
            label_list.append([int(labels[i].item())])
        confidence_sum += sum((F.softmax(outputs.data.detach(), dim=1).cpu().data)[idx_list, label_list])


    accuracy = correct/total
    confidence = confidence_sum/total
    return accuracy, loss, confidence




