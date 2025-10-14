# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 21:25:33 2021

@author: Can Hou, Haowen Liu
"""
import pandas as pd
import numpy as np
import os

path = r'~/depression/'
phe = []
for dir_,_,files in os.walk(path+'age/result/binomial'):
    for file in files:
        if 'binomial_' in file:
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
phe_directional = phe_.loc[phe_['q']<0.05]
phe_directional.to_csv(path + 'age/result/binomial_directional.csv')
phe_.to_csv(path + 'age/result/binomial_comorbidity.csv')
