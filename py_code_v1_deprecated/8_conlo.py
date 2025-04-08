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
import statsmodels.discrete.conditional_models as sm
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')

path = r'~/depression/'

def defination(sick, history, inpatient):
    if float(sick) in history or float(sick) in inpatient:
        return 1
    else:
        return 0

def d_match(dataset,time_var):
    var1 = 'sex'
    var2 = 'birth_date'
    var3 = 'famIncome'
    date_death = 'time_end'
    n = 2
    
    print('Start matching')
    one_year = 365.25*24*3600
    case = dataset.loc[~dataset[time_var].isna()]
    result = []
    iter_ = 0
    for j in case.index:
        time_ = case.loc[j, time_var]
        try:
            t = dataset.loc[(dataset[var1]==case.loc[j,var1]) & 
                             (np.abs(dataset[var2]-case.loc[j,var2])<=one_year) & 
                             (dataset[var3]==case.loc[j,var3]) & 
                             ~(dataset[date_death]<=time_) & 
                             ~(dataset[time_var]<=time_)].sample(n,random_state=0)
        except:
            t = dataset.loc[(dataset[var1]==case.loc[j,var1]) & 
                             (np.abs(dataset[var2]-case.loc[j,var2])<=one_year) & 
                             (dataset[var3]==case.loc[j,var3]) & 
                             ~(dataset[date_death]<=time_) & 
                             ~(dataset[time_var]<=time_)]
            
        result.append([case.loc[j,'eid'],1,time_,iter_])
        result += [[x,0,time_,iter_] for x in t.eid.values]
        iter_ += 1
    result = pd.DataFrame(result,columns=['eid','outcome_conlo','date_start_conlo','match_2_conlo'])
    result = pd.merge(result,dataset,on='eid',how='left')
    return result

def logistic_conditional(d1d2, df_matched_group, illList):
    dataset = df_matched_group.copy()
    inpatient_variable = 'inpatient_level1'
    eligible_variable = 'd_eligible'
    d1 = float(d1d2.split('-')[0])
    d2 = float(d1d2.split('-')[1])
    
    delDiseaseList = illList.copy()
    delDiseaseList.remove(str(d1)), delDiseaseList.remove(str(d2))

    dataset['flag'] = dataset[eligible_variable].apply(lambda x: d1 in x and d2 in x)
    dataset_d = dataset.loc[dataset['flag']==True]
    
    dataset_d['d2_time'] = dataset_d[inpatient_variable].apply(lambda x: x.get(d2,np.NaN))
    dataset_d['d1_time'] = dataset_d[inpatient_variable].apply(lambda x: x.get(d1,np.NaN))
    dataset_d_matched = d_match(dataset_d,'d2_time')
    dataset_d_matched['exposure'] = dataset_d_matched.apply(lambda row: 1 if 
                                                            row['d1_time']<row['date_start_conlo'] else 0,
                                                            axis=1)
 
    var_co_vars_lst_d = dataset_d_matched[co_vars].var()
    co_vars_selected = var_co_vars_lst_d[var_co_vars_lst_d != 0].index
    var_covar_lst_d = dataset_d_matched[delDiseaseList].var()
    delDiseaseList = var_covar_lst_d[var_covar_lst_d != 0].index
    
    model = sm.ConditionalLogit(np.asarray(dataset_d_matched['outcome_conlo'],dtype=int),
                        np.asarray(dataset_d_matched[['exposure'] + list(co_vars_selected) + list(delDiseaseList)],dtype=int),
                        groups=dataset_d_matched['match_2_conlo'].values)
    len_d_other, len_cov = len(delDiseaseList), 1 + len(co_vars_selected)
    result = model.fit_regularized(method='elastic_net',alpha=[0]*len_cov+[coe]*len_d_other)
    
    ii = np.flatnonzero(result.params)
    covar_lst_all = dataset_d_matched[['exposure'] + list(co_vars_selected) + list(delDiseaseList)].iloc[:,ii].columns
    covar_lst_all_d = [x for x in covar_lst_all if x in list(delDiseaseList)]
    
    new_dataset = dataset_d_matched[covar_lst_all_d]
    new_dataset['constant'] = 1
    covar_lst_all_d1 = []
    for i in range(len(new_dataset.columns)-1):
        try:
            vif = variance_inflation_factor(new_dataset.values, i)
            if vif < 5:
                covar_lst_all_d1.append(new_dataset.columns[i])
        except:
            covar_lst_all_d1.append(new_dataset.columns[i])
        
    model_refit = sm.ConditionalLogit(np.asarray(dataset_d_matched['outcome_conlo'],dtype=int),
                                      np.asarray(dataset_d_matched[['exposure'] + list(co_vars_selected) + list(covar_lst_all_d1)],dtype=int),
                                      groups=dataset_d_matched['match_2_conlo'].values)
    
    try:
        try:
            result_refit = model_refit.fit(maxiter=300)
            return [d1d2,result_refit.params[0],result_refit.pvalues[0],'%.2f (%.2f-%.2f)' % (np.exp(result_refit.params[0]),
                                                          np.exp(result_refit.conf_int()[0][0]),
                                                          np.exp(result_refit.conf_int()[0][1])),np.NaN]
        except:
            result_refit = model_refit.fit(method='cg', maxiter=300)
            return [d1d2,result_refit.params[0],result_refit.pvalues[0],'%.2f (%.2f-%.2f)' % (np.exp(result_refit.params[0]),
                                                          np.exp(result_refit.conf_int()[0][0]),
                                                          np.exp(result_refit.conf_int()[0][1])),np.NaN]
    except Exception as e:
        print(e)
        return [d1d2,np.NaN,np.NaN,np.NaN,e]

array = np.load(path +'result/baseline_merged_main_group.npy', allow_pickle=True)
array_columns = np.load(path + 'result/baseline_merged_columns_main_group.npy', allow_pickle=True)

df_matched_group = pd.DataFrame(array, columns=array_columns)
binomial_directional = pd.read_csv(path + 'result/binomial_directional.csv', index_col=0)
have_binomial_directional = pd.read_csv(path + 'result/have_binomial_directional.csv', index_col=0)

all_binomial_directional = pd.concat([binomial_directional, have_binomial_directional])
all_trajactory_list = all_binomial_directional['name'].values
trajactory_list = binomial_directional['name'].values

co_vars = []
for var in ['civil','education']:
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

np.random.seed(number)
np.random.shuffle(trajactory_list)
total_length = len(trajactory_list)

have_pair_conlo = pd.read_csv(path + 'result/have_conlogistic.csv', index_col=0)
have_pair_conlo = have_pair_conlo.loc[~have_pair_conlo['p'].isna()]
have_pair_conlo_list = [x for x in have_pair_conlo['name']]

result_final = []
for i in range(len(trajactory_list)):
    pair = trajactory_list[i]
    with open (path+'temp.cache','r') as f:
        pair_exl = f.read()
    if pair not in pair_exl.split(','):
        with open (path+'temp.cache','a') as f:
            f.write(',%s' % (pair))
        progress = (len(pair_exl.split(','))+1)/len(trajactory_list)
        print('%i: %.2f%% in condition logistic' % (number,progress*100))
        if pair in have_pair_conlo_list:
            result_final.append(have_pair_conlo.loc[have_pair_conlo['name']==pair][['name','coef','p','OR_CI','note']])
        else:
            result_final.append(logistic_conditional(pair, df_matched_group, illnessList))
    else:
        continue
logistic_result = pd.DataFrame(result_final,columns=['name','coef','p','OR_CI','note'])
logistic_result.to_csv(path + 'result/conlogistic/logistic_%i.csv' % (number))
