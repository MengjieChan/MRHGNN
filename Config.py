# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:13:53 2023

@author: Mengjie Chen
"""

import argparse
import os
import torch

file = os.path.dirname(__file__) # current file

abs_path = os.path.abspath(file) # absolute path

data_file = "%s/DATA" % abs_path


Remove_Repeated_Self_Loops = True


Use_Cell_Line_Feature = True



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse():
    p = argparse.ArgumentParser("MRHGNN: Enhanced Multimodal Relation Hypergraph Neural Network for Synergistic Drug Combination Forecasting", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset',type = str, default = 'ONEIL',help = 'the name of the dataset' ) # 'ONEIL', 'ALMANAC'
    p.add_argument('--model_name', type = str, default = 'Synergy')
    p.add_argument('--threshold',type = int, default = 30,help = 'the threshold of positive samples') # 30, 10
    p.add_argument('--k_fold', type = int, default = 10, help = 'k-fold cross validation')
    p.add_argument('--n_data_worker', type = int, default = 5, help = 'number of producers')
    
    p.add_argument('--cuda', type = str, default = '0', help = 'gpu id to use')
    
    p.add_argument('--alpha', type = float, default = 0.4, help = 'the parameters of the auxiliary task') # 0.4, 0.6
    p.add_argument('--learning_rate', type = float, default = 1e-3, help = 'learning rate')
    p.add_argument('--weight_decay', type = float, default = 1e-5, help = 'weight decay')
    p.add_argument('--epochs', type = int, default = 1500, help = 'number of epochs to train')
    p.add_argument('--seed', type = int, default = 5, help = 'seed for randomness')
    

    args = p.parse_args()
    
    args.data_file = data_file
    args.device = device
    args.Use_Cell_Line_Feature = Use_Cell_Line_Feature

    return args