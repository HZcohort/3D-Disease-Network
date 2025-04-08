# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:09:18 2021

@author: Can Hou, Haowen Liu
"""
import sys
import argparse
#------------------
parser = argparse.ArgumentParser(description='New_depression project')
parser.add_argument("--number", type=int)
parser.add_argument("--coe", type=float)
args = parser.parse_args()
number = args.number
coe = args.coe
#logistic
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression as LR
import time

path = r'~/depression/'

def defination(sick, history, inpatient):
    if float(sick) in history or float(sick) in inpatient:
        return 1
    else:
        return 0

def logistic_unconditional(d1d2, df_matched_group, illList):
    time1 = time.time()
    dataset = df_matched_group
    inpatient_variable = 'inpatient_level1'
    eligible_variable = 'd_eligible'
    temp = []
    
    d1 = float(d1d2.split('-')[0])
    d2 = float(d1d2.split('-')[1])
    
    delDiseaseList = illList.copy()
    try:
        delDiseaseList.remove(str(d1))
    except:
        delDiseaseList = delDiseaseList
    try:
        delDiseaseList.remove(str(d2))
    except:
        delDiseaseList = delDiseaseList
        
    dataset['flag'] = dataset[eligible_variable].apply(lambda x: d1 in x and d2 in x)
    dataset_d = dataset.loc[dataset['flag']==True]
    var_co_vars_lst_d = dataset_d[co_vars].var()
    co_vars_selected = var_co_vars_lst_d[var_co_vars_lst_d != 0].index
    var_covar_lst_d = dataset_d[delDiseaseList].var()
    delDiseaseList = var_covar_lst_d[var_covar_lst_d != 0].index

    dataset_d['d1'] = dataset_d[inpatient_variable].apply(lambda x: 1 if d1 in x.keys() else 0)    
    dataset_d['d2'] = dataset_d[inpatient_variable].apply(lambda x: 1 if d2 in x.keys() else 0)
    dataset_d['constant'] = 1

    model = LR(solver='saga', penalty='l1', random_state=42, C=coe**(-1), n_jobs=1)
    model.fit(np.asarray(dataset_d[['d2','constant'] + list(co_vars_selected) + list(delDiseaseList)], dtype=int),
                           np.asarray(dataset_d['d1'], dtype=int))
    ii = np.flatnonzero(model.coef_)
    ii_ = ii[np.where(ii>(np.asarray(dataset_d[['d2','constant'] + list(co_vars_selected)]).shape[1]-1))]
    covar_lst_all_d1 = dataset_d[['d2','constant'] + list(co_vars_selected) + list(delDiseaseList)].iloc[:,ii_].columns

    model1 = sm.Logit(np.asarray(dataset_d['d1'],dtype=int),
                                np.asarray(dataset_d[['d2','constant'] + list(co_vars_selected) + list(covar_lst_all_d1)], dtype=int))
    try:
        try:
            result = model1.fit(maxiter=300)
            temp += [d1d2, result.params[0], result.pvalues[0],'%.2f (%.2f-%.2f)' % (np.exp(result.params[0]),
                                                                                     np.exp(result.conf_int()[0][0]),
                                                                                     np.exp(result.conf_int()[0][1])), np.NaN]
        except:
            result = model1.fit(method='cg', maxiter=300)
            temp += [d1d2, result.params[0], result.pvalues[0],'%.2f (%.2f-%.2f)' % (np.exp(result.params[0]),
                                                                                     np.exp(result.conf_int()[0][0]),
                                                                                     np.exp(result.conf_int()[0][1])), np.NaN]
    except Exception as e:
        print(e)
        temp += [d1d2,np.NaN,np.NaN,np.NaN,e]
    time2 = time.time()
    print('Spent %0.2f s time' % (time2 - time1))
    return temp

array = np.load(path +'result/baseline_merged_main_group.npy', allow_pickle=True)
array_columns = np.load(path + 'result/baseline_merged_columns_main_group.npy', allow_pickle=True)

df_matched_group = pd.DataFrame(array,columns=array_columns)
binomial_comorbidity = pd.read_csv(path + 'result/binomial_comorbidity.csv', index_col=0)
have_binomial_comorbidity = pd.read_csv(path + 'result/have_binomial_comorbidity.csv', index_col=0)
all_binomial_comorbidity = pd.concat([binomial_comorbidity, have_binomial_comorbidity])

all_trajactory_list = binomial_comorbidity['name'].values

trajactory_list = binomial_comorbidity['name'].values
co_vars = ['age']
for var in ['civil','famIncome','education','sex']:
    temp = pd.get_dummies(df_matched_group[var],prefix=var)
    co_vars += [x for x in temp.columns[1::]]
    df_matched_group = pd.concat([df_matched_group,temp],axis=1)

illnessList = []
for i in range(len(all_trajactory_list)):
    illnessList.append(str(all_trajactory_list[i].split('-')[0]))
    illnessList.append(str(all_trajactory_list[i].split('-')[1]))
illnessList = list(set(illnessList))
for ill in illnessList:    
    df_matched_group[str(ill)] = df_matched_group.apply(lambda row: defination(ill, row['history_level1'], [value for key, value in enumerate(row['inpatient_level1'])]), axis=1)

new_dataset = df_matched_group[illnessList]
new_dataset['constant'] = 1
covar_lst_all_d = []
for i in range(len(new_dataset.columns)-1):
    try:
        vif = variance_inflation_factor(new_dataset.values, i)
        if vif < 5:
            covar_lst_all_d.append(new_dataset.columns[i])
    except:
        covar_lst_all_d.append(new_dataset.columns[i])

np.random.seed(number)
np.random.shuffle(trajactory_list)
total_length = len(trajactory_list)

have_pair_unconlo = pd.read_csv(path + 'result/have_unconlogistic.csv', index_col=0)
have_pair_unconlo = have_pair_unconlo.loc[~have_pair_unconlo['p'].isna()]
have_pair_unconlo_list = [x for x in have_pair_unconlo['name']]

result_final = []
for i in range(len(trajactory_list)):
    pair = trajactory_list[i]
    with open (path+'temp.cache','r') as f:
        pair_exl = f.read()
    if pair not in pair_exl.split(','):
        with open (path+'temp.cache','a') as f:
            f.write(',%s' % (pair))
        progress = (len(pair_exl.split(','))+1)/len(trajactory_list)
        print('%i: %.2f%% in uncondition logistic' % (number,progress*100))
        if pair in have_pair_unconlo_list:
            result_final.append(have_pair_unconlo.loc[have_pair_unconlo['name']==pair][['name','coef_1','p_1','OR_CI_1','note1']])
        else:
            result_final.append(logistic_unconditional(pair, df_matched_group, covar_lst_all_d))
    else:
        continue
logistic_result = pd.DataFrame(result_final,columns=['name','coef_1','p_1','OR_CI_1','note1'])
logistic_result.to_csv(path + 'result/unconlogistic/unconlogistic_%i.csv' % (number))
