# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:29:34 2021

@author: Can Hou, Haowen Liu
"""
import sys
import argparse
import os
#------------------
parser = argparse.ArgumentParser(description='New_depression project')
parser.add_argument("--number", type=int)
parser.add_argument("--coe", type=float)
args = parser.parse_args()
number = args.number
coe = args.coe

import pandas as pd
import numpy as np
path = r'~/depression/'

# conditional logistic
phe = []
for dir_,_,files in os.walk(path+'age/result/conlogistic'):
    for file in files:
        if 'logistic_' in file:
            path_file = os.path.join(dir_,file)
            csv = pd.read_csv(path_file,index_col=0)
        else:
            continue
        try:
            phe = pd.concat([csv,phe])
        except:
            phe = csv.copy()
            
phe_ = phe.copy()
phe_ = phe_.sort_values(by=['p'])
phe_['order'] = np.arange(len(phe_))+1
phe_['q'] = (phe_['p']*len(phe_))/phe_['order']
phe_select = phe_.loc[phe_['q']<0.05]
phe_select.to_csv(path + 'age/result/conlogistic_summary_%s.csv' % (coe))

#unconditional logistic
phe = []
for dir_,_,files in os.walk(path+'age/result/unconlogistic'):
    for file in files:
        if 'unconlogistic_' in file:
            path_file = os.path.join(dir_,file)
            csv = pd.read_csv(path_file,index_col=0)
        else:
            continue
        try:
            phe = pd.concat([csv,phe])
        except:
            phe = csv.copy()
            
phe_ = phe.copy()
phe_ = phe_.sort_values(by=['p_1'])
phe_['order_1'] = np.arange(len(phe_))+1
phe_['q_1'] = (phe_['p_1']*len(phe_))/phe_['order_1']
phe_select = phe_.loc[(phe_['q_1']<0.05)]
phe_select.to_csv(path + 'age/result/unconlogistic_summary_%s.csv' % (coe))
