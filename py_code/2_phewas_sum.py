# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:29:34 2021

@author: Can Hou, Haowen Liu
"""

import pandas as pd
import os
import numpy as np
path = r'~/depression/'

phe = []
for dir_,_,files in os.walk(path+'age/result/phewas1'):
    for file in files:
        if 'level1' in file:
            path_file = os.path.join(dir_,file)
            csv = pd.read_csv(path_file,index_col=0)
        else:
            continue
        try:
            phe = pd.concat([csv,phe])
        except:
            phe = csv.copy()
phe.to_csv(path + 'age/result/cox_result_level1_del.csv')

phe = []
for dir_,_,files in os.walk(path+'age/result/phewas2'):
    for file in files:
        if 'level2' in file:
            path_file = os.path.join(dir_,file)
            csv = pd.read_csv(path_file,index_col=0)
        else:
            continue
        try:
            phe = pd.concat([csv,phe])
        except:
            phe = csv.copy()
phe.to_csv(path + 'age/result/cox_result_level2_del.csv')

level_1 =  pd.read_csv(path + 'age/result/cox_result_level1_del.csv')
level_2 =  pd.read_csv(path + 'age/result/cox_result_level2_del.csv')

def merge(L1, L2):
    L2['disease'] = L2['disease'].apply(lambda x: str(x).split('.')[0])
    L2_ = L2.drop_duplicates(subset='disease')
    merge_df = []
    for dis in L1['disease']:
        if dis not in [float(x) for x in L2_['disease']]:
            try:
                merge_df = pd.concat([merge_df, L1.loc[L1['disease']==dis]])
            except:
                merge_df = L1.loc[L1['disease']==dis]
    return merge_df

merge_ = merge(level_1, level_2)
level_2 =  pd.read_csv(path + 'age/result/cox_result_level2_del.csv')
all_df = pd.concat([level_2, merge_])
phe = all_df
phe_ = phe.loc[~phe['p'].isna()]
phe_ = phe_.sort_values(by=['p'])
phe_['order'] = np.arange(len(phe_))+1
phe_['q'] = (phe_['p']*len(phe_))/phe_['order']
phe_select = phe_.loc[phe_['q']<0.05]
phe_select = phe_select.loc[phe_select['number']>=200]
phe_select.to_csv(path + 'age/result/phewas_summary_L1L2.csv')

phecode_cate = pd.read_csv(path + 'originData/phecode_definitions1.2.csv')
phe_def = {i:j for i,j in phecode_cate[['phecode','phenotype']].values}
phe_cate = {i:j for i,j in phecode_cate[['phecode','category']].values}
phe = pd.merge(phe,phe_[['disease','q']],on=['disease'],how='left')
phe['phenotype'] = phe['disease'].apply(lambda x: phe_def[x])
phe['category'] = phe['disease'].apply(lambda x: phe_cate[x])

#HR
def new_format(x):
    try:
        x = x.split(' ')
        n = x[0].split('/')
        return '{0:,}/{1:,} {2:s}'.format(int(n[0]),float(n[1]),x[1])
    except:
        return np.NaN

for col in ['exp','unexp']:
    phe[col] = phe[col].apply(lambda x: new_format(x))
phe['ci'] = phe.apply(lambda row: '%.2f (%.2f-%.2f)' % (np.exp(row['coef']),np.exp(row['coef']-1.96*row['se']),
                                                        np.exp(row['coef']+1.96*row['se'])),axis=1)
for col in ['q','ci']:
    if col == 'q':
        phe[col] = phe.apply(lambda row: 'NA' if pd.isna(row['p']) else '%.2E' % (row[col]),axis=1)
    else:
        phe[col] = phe.apply(lambda row: 'NA' if pd.isna(row['p']) else row[col],axis=1)
phe = phe.sort_values(by=['category','disease'])
phe[['disease','phenotype','category','exp','unexp','ci','q','coef','se']].to_csv(path + 'age/result/phewas_table_L1L2.csv')

