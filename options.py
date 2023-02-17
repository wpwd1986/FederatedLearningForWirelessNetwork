#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # load dataset and split users
    parser.add_argument('--setmode', type=str, default='load', help="load or create")
    parser.add_argument('--dataset', type=str, default='mnist', help="mnist or cifar")
    parser.add_argument('--dictfile', type=str, default='./dicts/dictmn_niid_200030_3.npy', help="split users")
    parser.add_argument('--snrsingle', type=str, default='./dicts/snr_CFR_u30c10_30db.csv', help="singleBS chan info")
    # federated arguments
    parser.add_argument('--sim', type=int, default=1, help="rounds of simulations") # 仿真次数
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training") # 每次仿真通信轮数
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate") # 学习率 原0.01
    parser.add_argument('--method', type=int, default=1, help="1=rand,2=chan,3=imp") # 用户选择方法
    parser.add_argument('--bal', type=int, default=0, help="balance selection") # 是否开启平衡模式
    parser.add_argument('--rho', type=float, default=1.015, help="balance factor") # 平衡系数
    parser.add_argument('--beta', type=float, default=0.5, help="base factor") # 基础系数
    # wireless arguments
    parser.add_argument('--num_users', type=int, default=30, help="number of users: K") # 用户数量
    parser.add_argument('--num_chan', type=int, default=10, help="number of channels: M") # 子信道数量
    parser.add_argument('--totband', type=int, default=1*1000000, help="total bandwidth (MHz)") # 子信道带宽 1MHz
    parser.add_argument('--size', type=float, default=100, help="packetsize (MB)") # 数据包大小
    parser.add_argument('--uptime', type=float, default=0.1, help="upload time") # 上传时间
    parser.add_argument('--cond', type=int, default=1, help="channel condition") # 是否考虑信道状况
    # training arguments
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E") # 本地训练轮数
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B") # 本地训练batch样本数
    parser.add_argument('--bs', type=int, default=128, help="test batch size") # 测试batch样本数
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name') # model类型 mlp/cnn
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    # other arguments
    #parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset") # dataset
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C") # 用户采用比例 原0.1
    parser.add_argument('--iid', action='store_true', default=False, help='whether i.i.d or not') # 独立同分布
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    #parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args
