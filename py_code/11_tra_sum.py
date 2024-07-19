# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:38:47 2021

@author: Can Hou, Haowen Liu
"""
import argparse
#------------------
parser = argparse.ArgumentParser(description='New_depression project')
parser.add_argument("--coe", type=float)
args = parser.parse_args()
coe = args.coe
import pandas as pd
import numpy as np
path = r'~/depression/'

def long_tra(name):
    global tra_result
    
    d2 = int(name.split('-')[-1])
    d_lst = conlogistic.loc[conlogistic['d1']==d2].d2.values
    if len(d_lst) == 0:
        tra_result.append(name)
    else:
        for d in d_lst:
            long_tra('%s-%i' % (name,d))

group_dict = {'depression':999}
group = 'depression'
phewas = pd.read_csv(path + 'age/result/phewas_summary_L1L2.csv',index_col=0)
commorbidity = pd.read_csv(path + 'age/result/comorbidity_summary.csv',index_col=0)

d_com = {'%i-%i' % (i,j):np.log(k) for i,j,k in commorbidity[['d1','d2','RR']].values}
d_com_ = {'%i-%i' % (j,i):np.log(k) for i,j,k in commorbidity[['d1','d2','RR']].values}
d_com.update(d_com_)
d_num = {i:j for i,j in phewas[['disease','number']].values}
d_coef = {i:j for i,j in phewas[['disease','coef']].values}
#
array = np.load(path + 'age/result/baseline_merged_main_group.npy', allow_pickle=True)
array_columns = np.load(path + 'age/result/baseline_merged_columns_main_group.npy', allow_pickle=True)
df_matched_group = pd.DataFrame(array, columns=array_columns)
d1d2_array = df_matched_group['d1d2'].values

#-------------------------------------------------------------------------------------------
#conditional logistic
conlogistic = pd.read_csv(path + 'age/result/conlogistic_summary_%s.csv' % (coe),index_col=0)
conlogistic['d1'] = conlogistic['name'].apply(lambda x: float(x.split('-')[0]))
conlogistic['d2'] = conlogistic['name'].apply(lambda x: float(x.split('-')[1]))
conlogistic['link_type'] = 'dir'
#d0-d1
d_first = conlogistic.loc[~conlogistic['d1'].isin(conlogistic.d2.values)]
d_first_lst = set(d_first.d1.values)
d0_d1 = []
for d in d_first_lst:
    d0_d1.append(['%i-%i' % (group_dict[group],d),group_dict[group],d,'dir'])
d0_d1 = pd.DataFrame(d0_d1,columns=['name','d1','d2','link_type'])
conlogistic = pd.concat([conlogistic,d0_d1])
#
for var in ['d1','d2']:
    conlogistic['number_'+var] = conlogistic[var].apply(lambda x: d_num.get(x))
    conlogistic['coef_'+var] = conlogistic[var].apply(lambda x: d_coef.get(x))
conlogistic['number'] = conlogistic['name'].apply(lambda x: len([c for c in d1d2_array if x in c]))
#conlogistic['coef'] = logistic['name'].apply(lambda x: d_com.get(x))
conlogistic.to_csv(path + 'age/result/tra_summary_%s.csv' % (coe))

#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
#unconditional logistic
unconlogistic = pd.read_csv(path + 'age/result/unconlogistic_summary_%s.csv' % (coe),index_col=0)
unconlogistic['d1'] = unconlogistic['name'].apply(lambda x: float(x.split('-')[0]))
unconlogistic['d2'] = unconlogistic['name'].apply(lambda x: float(x.split('-')[1]))
unconlogistic['link_type'] = 'bi-dir'
#
for var in ['d1','d2']:
    unconlogistic['number_'+var] = unconlogistic[var].apply(lambda x: d_num.get(x))
    unconlogistic['coef_'+var] = unconlogistic[var].apply(lambda x: d_coef.get(x))
unconlogistic['coef'] = unconlogistic['coef_1']
unconlogistic.to_csv(path + 'age/result/com_summary_%s.csv' % (coe))