# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:47:36 2021

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
from scipy.stats import binom_test
path = r'~/depression/'
def d1d2_selection(d1d2):
    
    df = df_matched_group
    d1d2_var = 'd1d2'
    inpatient_var = 'inpatient_level1'
    eligible_var = 'd_eligible'
    
    d1 = d1d2[0]
    d2 = d1d2[1]
    d1_d2 = '%s-%s' % (str(d1),str(d2))
    d2_d1 = '%s-%s' % (str(d2),str(d1))
    
    df['flag'] = df[eligible_var].apply(lambda x: d1 in x and d2 in x)
    dataset = df.loc[df['flag']==True]
    array = dataset[d1d2_var].values
    array_inpatient = dataset[inpatient_var].values
    
    length_full = len([x for x in array_inpatient if d1 in x.keys() and d2 in x.keys()])
    len_d1d2 = len([x for x in array if d1_d2 in x])
    len_d2d1 = len([x for x in array if d2_d1 in x])
    
    if len_d1d2 >= len_d2d1:
        p_value = binom_test([len_d1d2,length_full-len_d1d2],alternative='greater')
        return [d1,d2,d1_d2,len_d1d2,length_full,p_value]
    else:
        p_value = binom_test([len_d2d1,length_full-len_d2d1],alternative='greater')
        return [d2,d1,d2_d1,len_d2d1,length_full,p_value]

array = np.load(path + 'result/baseline_merged_main_group.npy',allow_pickle=True)
array_columns = np.load(path + 'result/baseline_merged_columns_main_group.npy',allow_pickle=True)
df_matched_group = pd.DataFrame(array,columns=array_columns)
trajactory_list = pd.read_csv(path + 'result/comorbidity_summary.csv', index_col=0)[['d1','d2']].values

print("totall length of trajactory_list %i" % (len(trajactory_list)))
result_final = []
np.random.seed(number)
np.random.shuffle(trajactory_list)
total_length = len(trajactory_list)
##
result_final = []
for i in range(len(trajactory_list)):
    pair = trajactory_list[i]
    save = '%s-%s' % (pair[0],pair[1])
    with open (path+'temp.cache','r') as f:
        save_exl = f.read()
    if save not in save_exl.split(','):
        with open (path+'temp.cache','a') as f:
            f.write(',%s' % (save))
        progress = (len(save_exl.split(','))+1)/len(trajactory_list)
        print('%i: %i complete (%.2f%% in binomial test)' % (number,len(save_exl.split(','))+1,progress*100))
        result_final.append(d1d2_selection(pair))
    else:
        continue
    
binomial_result = pd.DataFrame(result_final,columns=['d1','d2','name','length','N','p'])
binomial_result.to_csv(path + 'result/binomial/binomial_%i.csv' % (number))