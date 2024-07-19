# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:44:59 2021

@author: Can Hou, Haowen Liu
"""

import pandas as pd
import numpy as np
import time
import math

def exc_lst(d):
    lst = []
    exl_range = phecode_cate.loc[phecode_cate['phecode']==d]['phecode_exclude_range'].values[0]
    if pd.isna(exl_range):
        return set(lst)
    else:
        for range_ in exl_range.split(','):
            exl_lower, exl_higher = float(range_.split('-')[0]), float(range_.split('-')[1])
            exl_list_index = np.where(np.all([phecode_lst>=exl_lower, phecode_lst<=exl_higher], axis=0))[0]
            exl_list_temp = phecode_lst[exl_list_index]
            lst += [x for x in exl_list_temp]
        return set(lst)

def d1_d2(dataset):
    exposed = dataset.copy()
    id_ = 'eid'
    inpatient_variable = 'inpatient_level1'
    date_start_variable = 'dia_date'
    eligible_variable = 'd_eligible'

    array = exposed[[id_, date_start_variable, inpatient_variable, eligible_variable]].values
    d1d2_result = []
    total = len(array)
    time0 = time.time()
    for i in range(len(array)):
        if i%3000 == 0:
            print("Progress: %.1f%%" % (i/total*100))
            print("Time spent: %.1f mins" % ((time.time()-time0)/60))
        d1d2 = []
        dict_temp = array[i][2]
        eligible = array[i][-1]
        d_list = [x for x in dict_temp.keys() if x in eligible]
        length = len(d_list)
        if length <= 1:
            d1d2_result.append([])
            continue
        for j in range(length-1):
            for k in (range(j+1,length)):
                d1 = d_list[j]
                d2 = d_list[k]
                if dict_temp.get(d2) > dict_temp.get(d1):
                    d1d2.append("%s-%s" % (d1,d2))
        d1d2_result.append(d1d2)
    exposed['d1d2'] = d1d2_result
    return exposed

def deal_(item):
    if len(str(item).split('.')[1]) == 2:
        temp_item = math.floor(item*10)/10
    else:
        temp_item = item
    return temp_item

def inpatient_process(dict_, history):
    if type(dict_) is not dict:
        return {}
    new_dict = {}
    for d in dict_.keys():
        if pd.isna(d):
            continue
        if deal_(d) in history:
            continue
        if deal_(d) not in new_dict.keys():
            new_dict[deal_(d)] = dict_[d]
        else:
            new_dict[deal_(d)] = min(dict_[d], new_dict[deal_(d)])
    return new_dict

def deal(lst):
    new_lst = []
    for item in lst:
        if pd.isna(item):
            continue
        try:
            if len(str(item).split('.')[1]) == 2:
                temp_item = math.floor(item*10)/10
                new_lst.append(temp_item)
            else:
                new_lst.append(item)
        except:
            print(item)
    return new_lst

path = r'~/depression/'
#phecode
phecode_cate = pd.read_csv(path+'originData/phecode_definitions1.2.csv')
phecode_lst = np.array([x for x in phecode_cate.phecode.values])

#sex for exclusion
exc_sex = {}
exc_sex[0] = list(phecode_cate.loc[phecode_cate['sex']=='Female'].phecode.values)
exc_sex[1] = list(phecode_cate.loc[phecode_cate['sex']=='Male'].phecode.values)
#d list
phewas_summary = pd.read_csv(path + 'age/result/phewas_summary_L1L2.csv', index_col=0)
phewas_summary = phewas_summary.loc[phewas_summary['number']>=200]
disease_list = phewas_summary.loc[phewas_summary['coef']>0]['disease'].values

exc_dict = {i:exc_lst(i) for i in disease_list}
array = np.load(path + 'age/df_merged.npy',allow_pickle=True)
array_columns = np.load(path + 'age/df_merged_columns.npy', allow_pickle=True)
df_matched = pd.DataFrame(array,columns=array_columns)

#grouping
df_matched['group'] = df_matched['outcome'].apply(lambda x: 1 if x==1 else np.NaN)
df_matched = df_matched.dropna(subset=['group'])

#medical history
df_matched['history'] = df_matched['history'].apply(lambda x: [] if type(x) is float else x)
df_matched['history_level1'] = df_matched['history'].apply(lambda x: deal(x))
df_matched['inpatient_level1'] = df_matched.apply(lambda row: inpatient_process(row['inpatient'],row['history_level1']),axis=1)
df_matched['d_eligible'] = df_matched.apply(lambda row: [x for x in disease_list
                                            if len(set(row['history']).intersection(exc_dict[x]))==0 and
                                            x not in exc_sex[row['sex']]], axis=1)
df_matched_group = d1_d2(df_matched)
np.save(path + 'age/result/baseline_merged_main_group.npy', df_matched_group.values)
np.save(path + 'age/result/baseline_merged_columns_main_group.npy', df_matched_group.columns)
# ---------------------------------------------------------------------------------