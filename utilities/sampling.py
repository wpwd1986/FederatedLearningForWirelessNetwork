#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms
#import random

#%%
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users) # 60000/100 = 600
    num_items = 150
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # Randomly pick num_items pictures for this user
        all_idxs = list(set(all_idxs) - dict_users[i]) # Remove selected
    return dict_users

#%%
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 2000, 30
    user_shards = 5
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs) # 0-59999
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # Stack new arrays vertically
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # Sort and extract index
    idxs = idxs_labels[0,:]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, user_shards, replace=False)) # Each user chooses 5*30=150
        #shardnum = random.randint(1,3)
        #rand_set = set(np.random.choice(idx_shard, shardnum, replace=False))
        idx_shard = list(set(idx_shard) - rand_set) # Remove selected
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

#%%
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = 150
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # Randomly pick num_items pictures for this user
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

#%%
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_shards, num_imgs = 100, 500
    user_shards = 3
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs) # 0-49999
    list_labels = np.empty(0)
    for images, labels in dataset:
        list_labels = np.append(list_labels,labels)
    list_labels = list_labels.astype(int)
    # sort labels
    idxs_labels = np.vstack((idxs, list_labels)) # Stack new arrays vertically
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # Sort and extract index
    idxs = idxs_labels[0,:]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, user_shards, replace=False)) # Each user chooses 5*30=150
        #shardnum = random.randint(1,3)
        #rand_set = set(np.random.choice(idx_shard, shardnum, replace=False))
        idx_shard = list(set(idx_shard) - rand_set) # Remove selected
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

#%%
if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    #d = mnist_iid(dataset_train, num)
