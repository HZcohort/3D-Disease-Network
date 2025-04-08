# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 10:08:43 2021

@author: Can Hou, Haowen Liu
"""
import argparse
#-----------------------------------------
parser = argparse.ArgumentParser(description='New_depression project')
parser.add_argument('--number', type=int)
args = parser.parse_args()
number = args.number
#-----------------------------------------------
import pandas as pd
import numpy as np
import os
path = r'~/depression/'

array = np.load(path + 'age/result/baseline_merged_main_group.npy', allow_pickle=True)
array_columns = np.load(path + 'age/result/baseline_merged_columns_main_group.npy', allow_pickle=True)

df_matched_group = pd.DataFrame(array, columns=array_columns)

phe = []
for dir_,_,files in os.walk(path+'age/result/comorbidityResult'):
    for file in files:
        if 'comorbidity_' in file:
            path_file = os.path.join(dir_,file)
            csv = pd.read_csv(path_file,index_col=0)
        else:
            continue
        try:
            phe = pd.concat([csv,phe])
        except: 
            phe = csv.copy()
#
phe_ = phe.loc[~phe['p_rr'].isna()]
#RR
phe_ = phe_.sort_values(by=['p_rr'])
phe_['order_rr'] = np.arange(len(phe_))+1
phe_['q_rr'] = (phe_['p_rr']*len(phe_))/phe_['order_rr']
#phi
phe_ = phe_.sort_values(by=['p_phi'])
phe_['order_phi'] = np.arange(len(phe_))+1
phe_['q_phi'] = (phe_['p_phi']*len(phe_))/phe_['order_phi']
#    
phe_select = phe_.loc[(phe_['q_rr']<0.05) & (phe_['RR']>1) & 
                      (phe_['phi']>0) & (phe_['q_phi']<0.05)]
phe_select.to_csv(path + 'age/result/comorbidity_summary.csv')
