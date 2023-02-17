#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
#import heapq
#import random
import pandas as pd
import scipy.stats
import scipy.io as scio

# From other proejct files
from utilities.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utilities.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

#%% Data set initialization
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # read saved data directory
    if args.setmode == 'load':
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist) # load traning data
            dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
            dict_users = np.load(args.dictfile, allow_pickle='TRUE').item()
        elif args.dataset == 'cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
            dict_users = np.load(args.dictfile, allow_pickle='TRUE').item()
        else:
            exit('Error: unrecognized dataset')
    # create new directory
    elif args.setmode == 'create':
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Convert pixels and normalize with given mean and variance
            dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist) # load traning data
            dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users) # Assign data to users
                np.save('dictmn_iid_1000_new.npy', dict_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
                np.save('dictmn_niid_500120_5_new.npy', dict_users)
        elif args.dataset == 'cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
                np.save('dictcf_iid_c1.npy', dict_users)
            else:
                dict_users = cifar_noniid(dataset_train, args.num_users)
                np.save('dictcf_niid_100500_3_new.npy', dict_users)
                exit('Error: only consider IID setting in CIFAR10')
        else:
            exit('Error: unrecognized dataset')
    else:
        exit('Error: unrecognized mode')
    img_size = dataset_train[0][0].shape # [1, 28, 28]
    # Analyze user categories
    usercla = 3
    classlist = [[0 for col in range(usercla)] for row in range(args.num_users)]
    for i in range(args.num_users):
        for j in range(usercla):
            classlist[i][j] = dataset_train[dict_users[i][j*30]][1]

#%% Load model
    for iter_sim in range(args.sim):
        # build model
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x # 1*28*28=784
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)
        net_glob.train() # Sets the module in training mode
        # copy weights
        wt_glob = net_glob.state_dict() # Returns a dictionary containing a whole state of the module

#%% Environment parameter initialization
        loss_train = []
        acc_train = [0]*args.epochs
        acc_test = [0]*args.epochs
        loss_train_fin = [0]*args.epochs
        loss_test_fin = [0]*args.epochs
        user_imp = [0]*args.num_users
        #user_weight = [0]*args.num_users
        grad_globa = np.zeros((args.epochs+1,img_size[0]*img_size[1]*img_size[2]*10))
        user_select = []
        acc_test_sum = 0
        acc_test_scale = []
        acc_test_iter = []
        user_sortord = [[] for i in range(args.epochs)]
        #sumimp = [0]*args.epochs
        #sumimp_avg = [0]*args.epochs
        loss_f_none = nn.KLDivLoss(reduction='none')
        loss_f_mean = nn.KLDivLoss(reduction='mean')
        loss_f_bs_mean = nn.KLDivLoss(reduction='batchmean')
        glob_weight = torch.zeros(784)
        idx_chanres = [None]*args.num_chan
        size_remain = [0]*args.num_chan

#%% Model distribution training
        print('Total Rounds {:2d}'.format(args.epochs))
        print('Total Users {:3d}'.format(args.num_users))
        print('Total Bandwidth {:2f} MHz'.format(args.totband*args.num_chan/1000000))
        print('Sub-channel Bandwidth {:2f} MHz'.format(args.totband/1000000))
        print("--------------------")
        for iter in range(args.epochs):
            wt_locals, loss_locals = [], []
            grad_local = np.zeros((args.num_users,img_size[0]*img_size[1]*img_size[2]*10)) # 10
            print('Round {:3d}'.format(iter+1))
            print('Training user ', end='')
            for idx in range(args.num_users):
                print('{} '.format(idx+1), end='')
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                wt, loss, grad_norm, grad_local[idx] = local.train(net=copy.deepcopy(net_glob).to(args.device),
                                                                            cla = classlist, rd=iter+1, user=idx+1)
                grad_abs = [abs(i)*1000 for i in grad_local[idx]]
                if args.method == 3:
                    user_imp[idx] = scipy.stats.entropy(grad_abs, base=2) * loss
                    user_method = 'imp'
                #user_weight[idx] = wt['layer_input.weight']
                wt_locals.append(copy.deepcopy(wt)) # Weight
                loss_locals.append(copy.deepcopy(loss)) # Loss
            print('Done')

#%% Measure user value
            if args.cond == 1:
                col = np.arange(iter, 30*100+iter, 100) # 需要在输入文件中修改
                cfr = pd.read_csv(open(args.snrsingle), usecols=col, engine='python', header=None).values
                cfr = cfr.T
                pool_user = list(range(args.num_users))
                pool_chan = list(range(args.num_chan))
                pool_user_temp = pool_user.copy()
                pool_chan_temp = pool_chan.copy()
                cfr_value = cfr.copy()
                factor_imp = [0]*args.num_users
                factor_chan = np.zeros((args.num_users,args.num_chan))
                factor_grad = [0]*args.num_users
                for i in range(args.num_users):
                    factor_imp[i] = (user_imp[i] - np.min(user_imp))/(np.max(user_imp) - np.min(user_imp)) # 定义数据价值因子
                    factor_grad[i] = np.linalg.norm(grad_local[i]-grad_globa[iter])
                    for j in range(args.num_chan):
                        factor_chan[i,j] = (cfr[i,j] - np.min(cfr))/(np.max(cfr) - np.min(cfr))

#%% Subchannel allocation
                for i in range(args.num_chan):
                    if idx_chanres[pool_chan[i]] is None:
                        continue
                    else:
                        pool_chan_temp.remove(pool_chan[i])
                        pool_user_temp.remove(idx_chanres[pool_chan[i]])
                        cfr[:,pool_chan[i]] = [-100]*len(cfr[:,pool_chan[i]])
                        cfr[idx_chanres[pool_chan[i]]] = [-100]*len(cfr[idx_chanres[pool_chan[i]]])
                        factor_imp[idx_chanres[pool_chan[i]]] = -0.1
                        factor_grad[idx_chanres[pool_chan[i]]] = 1
                        print('Channel {:1d} is occupied by User {:1d} '.format(pool_chan[i], idx_chanres[pool_chan[i]]))
                pool_user = pool_user_temp.copy()
                pool_chan = pool_chan_temp.copy()
                print('Available users: ')
                print(pool_user)
                print('Available channels: ')
                print(pool_chan)
                while len(pool_chan) > 0:
                    sel_user = None
                    sel_chan = None
                    # 【随机选择】 在池中随机选择用户与信道
                    if args.method == 1:
                        sel_user = np.random.choice(pool_user)
                        sel_chan = np.random.choice(pool_chan)
                        user_method = 'rand'
                    # 【信道质量优先】 选择响应矩阵中最大值对应的用户与信道
                    elif args.method == 2:
                        sel_user, sel_chan = np.where(cfr == np.max(cfr))
                        sel_user = int(sel_user[0])
                        sel_chan = int(sel_chan[0])
                        user_method = 'chan'
                    # 【梯度差优先】 选择本地梯度与全局梯度差最小的用户
                    elif args.method == 4:
                        sel_user = np.argmin(factor_grad)
                        sel_chan = np.random.choice(pool_chan)
                        factor_grad[sel_user] = 1
                        user_method = 'grad'
                    # 【数据价值优先】 选择数据价值最高的用户及其信道
                    elif args.method==3 or args.method==6 or args.method==7 or args.method==8 or args.method==9 or args.method==0:
                        # 只考虑数据价值
                        if args.bal == 0:
                            sel_user = np.argmax(factor_imp) # 选取价值最大者
                            #sel_user = np.argmin(factor_imp) #选取价值最小者
                            #sel_chan = np.argmax(cfr[sel_user]) # 最优信道
                            sel_chan = np.random.choice(pool_chan) # 随机信道
                            factor_imp[sel_user] = -1
                            #user_method = 'imp'
                        # 平衡价值 选择平衡响应矩阵中最大值对应的用户与信道
                        elif args.bal == 1:
                            valmx_bal = np.zeros((args.num_users,args.num_chan))
                            for i in range(args.num_users):
                                for j in range(args.num_chan):
                                    #前置预设权重
                                    #valmx_bal[i,j] = (args.rho**(-iter)/args.beta)*factor_chan[i,j] + (1 - (args.rho**(-iter)/args.beta))*factor_imp[i]
                                    #后置预设权重
                                    valmx_bal[i,j] = ((args.rho**(-iter))*(1-args.beta)+args.beta)*factor_chan[i,j]+(1-((args.rho**(-iter))*(1-args.beta)+args.beta))*factor_imp[i]
                            cfr_check = np.zeros((args.num_users,args.num_chan))
                            for i in range(args.num_users):
                                for j in range(args.num_chan):
                                    if cfr[i,j] > -100:
                                        cfr_check[i,j] = 1
                                    else:
                                        cfr_check[i,j] = -1
                            valmx_bal = valmx_bal * cfr_check
                            sel_user, sel_chan = np.where(valmx_bal == np.max(valmx_bal))
                            sel_user = int(sel_user[0])
                            sel_chan = int(sel_chan[0])
                            user_method = 'bal'+str(args.rho)+'-'+str(args.beta)
                    else:
                        exit('Error: unrecognized method')
                    #print(cfr)
                    #print('Select User {:1d} with Channel {:1d} '.format(sel_user, sel_chan))
                    idx_chanres[sel_chan] = sel_user
                    pool_chan.remove(sel_chan)
                    pool_user.remove(sel_user)
                    cfr[:,sel_chan] = [-100]*len(cfr[:,sel_chan])
                    cfr[sel_user] = [-100]*len(cfr[sel_user])
                #print(idx_chanres)

#%% 子信道用量
                idx_select = []
                for i in range(len(idx_chanres)):
                    print('Channel {:1d}, User {:1d}'.format(i, idx_chanres[i]))
                    if size_remain[i] == 0:
                        userband = (args.size*1024*8)/(args.uptime*np.log2((1+10**(cfr_value[idx_chanres[i],i]/10))))
                        print('  Band required {:.8f} MHz'.format(userband/1000000))
                        leftband = args.totband - userband
                        if leftband >= 0:
                            idx_select.append(idx_chanres[i])
                            idx_chanres[i] = None
                            print('  Joined')
                        if leftband < 0:
                            size_remain[i] = args.size-((args.totband*args.uptime*np.log2((1+10**(cfr_value[idx_chanres[i],i]/10))))/(1024*8))
                            print('  {:.4f} MB remaining'.format(size_remain[i]))
                    else:
                        userband = (size_remain[i]*1024*8)/(args.uptime*np.log2((1+10**(cfr_value[idx_chanres[i],i]/10))))
                        print('  Band required {:.8f} MHz'.format(userband/1000000))
                        leftband = args.totband - userband
                        if leftband >= 0:
                            idx_select.append(idx_chanres[i])
                            idx_chanres[i] = None
                            size_remain[i] = 0
                            print('  Finally joined')
                        if leftband < 0:
                            size_remain[i] = size_remain[i]-((args.totband*args.uptime*np.log2((1+10**(cfr_value[idx_chanres[i],i]/10))))/(1024*8))
                            print('  {:.4f} MB still remaining'.format(size_remain[i]))
                print('Channel Status: ')
                print(idx_chanres)
                #print(idx_select)

#%% 更新全局模型
                if len(idx_select) != 0:
                    wt_locals_sel = [0]*len(idx_select)
                    loss_locals_sel = [0]*len(idx_select)
                    for i in range(len(idx_select)):
                        wt_locals_sel[i] = wt_locals[idx_select[i]]
                        loss_locals_sel[i] = loss_locals[idx_select[i]]
                    # update global weights
                    wt_glob = FedAvg(wt_locals_sel)
                    if args.model == 'mlp':
                        glob_weight = wt_glob['layer_input.weight'] # MLP
                    elif args.model == 'cnn':
                        glob_weight = wt_glob['conv1.weight'] # CNN
                    net_glob.load_state_dict(wt_glob)
                    loss_avg = sum(loss_locals_sel) / len(loss_locals_sel)
                elif len(idx_select) == 0 and iter == 0:
                    loss_avg = 1
            else:
                idx_select = list(range(args.num_users))
                wt_glob = FedAvg(wt_locals)
                if args.model == 'mlp':
                    glob_weight = wt_glob['layer_input.weight'] # MLP
                elif args.model == 'cnn':
                    glob_weight = wt_glob['conv1.weight'] # CNN
                net_glob.load_state_dict(wt_glob)
                loss_avg = sum(loss_locals) / len(loss_locals)
                user_method = 'alljoin'
            print('Joined users {:1d}'.format(len(idx_select)))
            #print('Joined users {:1d}, Bandwidth cost {:2f} MHz'.format(len(idx_select), sumband/1000000))
            #print('Total value {:2f}, Average value {:2f}'.format(sumimp[iter],sumimp_avg[iter]))
            idx_select.sort()
            user_select.append(idx_select)
            print('Average loss {:.3f}'.format(loss_avg))
            loss_train.append(loss_avg)
            if len(idx_select)!=0:
                for i in idx_select:
                    grad_globa[iter+1] = grad_globa[iter+1] + grad_local[i]
                grad_globa[iter+1] = grad_globa[iter+1]/len(idx_select)
            else:
                grad_globa[iter+1] = grad_globa[iter]

#%% 测试模型
            net_glob.eval() # Sets the module in evaluation mode
            acc_train[iter], loss_train_fin[iter] = test_img(net_glob, dataset_train, args)
            acc_test[iter], loss_test_fin[iter] = test_img(net_glob, dataset_test, args)
            print("Training accuracy: {:.2f}".format(acc_train[iter]))
            print("Testing accuracy: {:.2f}".format(acc_test[iter]))
            # Scale
            scale = 1
            acc_test_iter.append(float(acc_test[iter]))
            acc_test_sum = acc_test_sum + float(acc_test[iter])
            if iter % scale == 0:
                acc_test_scale.append(acc_test_sum/scale)
                acc_test_sum = 0

#%% 结果绘图
        #user_weight = []
        idx_chanres = []
        plt.figure()
        #plt.title('fed_{}_{}_{}_iid{}'.format(args.dataset, args.model, args.epochs, args.iid)
        plt.xlabel('Communications rounds')
        plt.ylabel('Accuracy')
        plt.xlim((0,int(args.epochs/scale)))
        plt.ylim((0,100))
        #plt.ylim((min(acc_test_scale),100))
        if args.epochs < 10:
            plt.plot(range(args.epochs), acc_test, label="Testing accuracy")
        else:
            plt.plot(range(int(args.epochs/scale)), acc_test_scale, label="Testing accuracy")
        plt.savefig('./save/fed_{}_rd{}_lr{}_{}_{}.png'.format(args.model, args.epochs, args.lr, user_method, iter_sim+1))
        scio.savemat('./save/{}_{}.mat'.format(user_method, iter_sim+1),
                     {'acc_test_iter':acc_test_iter, 'acc_test_scale':acc_test_scale, 'user_select':user_select})
