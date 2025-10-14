# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 23:27:55 2022

@author: Can Hou, Haowen Liu
"""

import argparse
#-----------------------------------------
parser = argparse.ArgumentParser(description='New_depression project')
parser.add_argument('--number', type=int)
args = parser.parse_args()
number = args.number
#-----------------------------------------
import pandas as pd
import numpy as np
import time
from lifelines import CoxPHFitter
cph = CoxPHFitter()
import warnings
warnings.filterwarnings("ignore")
from statsmodels.duration.hazard_regression import PHReg

path = r'~/depression/'

def range_d(x):
    """
    phecode merge
    examples：254.2 --> 254.29
    241.0 --> 241.99
    241.23 --> 241.23
    """
    x_str = str(x)
    if x_str.split('.')[1] == '0':
        return round(x+0.99,2)
    elif len(x_str.split('.')[1]) == 1:
        return round(x+0.09,2)
    else:
        return x
    
    
def cox(disease, dataset, date_start_variable, threshold):
    id_='eid'
    inpatient_variable = 'inpatient'
    exposure_variable = 'outcome'
    date_end_variable = 'time_end'
    history_variable = 'history'
    co_vars_all = co_vars + ['age'] ###
    result = [disease]
    match_variable = 'match_2'
    
    dataset_analysis = dataset[co_vars_all+['sex',id_,exposure_variable,date_start_variable,   
                                            date_end_variable, inpatient_variable, history_variable, match_variable]].copy()
    #
    exl_range = phecode_cate.loc[phecode_cate['phecode'] == disease]['phecode_exclude_range'].values[0]
    if pd.isna(exl_range):
        dataset_analysis['flag_exl'] = 0
    else:
        exl_list = []
        for range_ in exl_range.split(','):
            exl_lower,exl_higher = float(range_.split('-')[0]), float(range_.split('-')[1])
            exl_list_index = np.where(np.all([phecode_lst >= exl_lower, phecode_lst <= exl_higher], axis=0))[0]
            exl_list_temp = phecode_lst[exl_list_index]
            exl_list += [x for x in exl_list_temp]
        dataset_analysis['flag_exl'] = dataset_analysis[history_variable].apply(lambda x: 
                                                                                1 if np.any([i in x for i in exl_list]) 
                                                                                else 0)
            
    dataset_analysis['flag_exl'] = dataset_analysis.apply(lambda row: 1 if disease in exc_sex[row['sex']] 
                                                          else row['flag_exl'],axis=1)
    #
    dataset_analysis = dataset_analysis.loc[dataset_analysis['flag_exl']==0]
    #
    if len(dataset_analysis) == 0:
        result += ['Sex specific']
        return result

    #outcome
    disease_upper = range_d(disease)
    disease_lst = [x for x in phecode_lst if x>=disease and x<=disease_upper]
    dataset_analysis['d_time'] = dataset_analysis[inpatient_variable].apply(lambda x: 
                                                                            np.nanmin([x.get(j,np.NaN) for j in disease_lst]))
        
    dataset_analysis['outcome_'] = dataset_analysis['d_time'].apply(lambda x: 0 if pd.isna(x) else 1)
    length = len(dataset_analysis.loc[(dataset_analysis[exposure_variable]==1) & 
                                              (dataset_analysis['outcome_']==1)])

    result += [length]
    #
    dataset_analysis['date_final'] = np.nanmin(dataset_analysis[['d_time',
                                                                 date_end_variable]],axis=1)
    dataset_analysis['time'] = (dataset_analysis['date_final'] - 
                                dataset_analysis[date_start_variable])/(365.25*24*3600)
    dataset_analysis['time'] = dataset_analysis['time'].astype(float)
    #time at risk
    n_exp = len(dataset_analysis.loc[(dataset_analysis[exposure_variable]==1) & (dataset_analysis['outcome_']==1)])
    n_unexp = len(dataset_analysis.loc[(dataset_analysis[exposure_variable]==0) & (dataset_analysis['outcome_']==1)])
    time_exp = dataset_analysis.groupby(by=exposure_variable).sum()['time'].loc[1]/1000
    time_unexp = dataset_analysis.groupby(by=exposure_variable).sum()['time'].loc[0]/1000
    
    if length < threshold:
        result += ['less than threshold','%i/%.2f (%.2f)' % (n_exp,time_exp,n_exp/time_exp),
                   '%i/%.2f (%.2f)' % (n_unexp,time_unexp,n_unexp/time_unexp)]
        return result
    #variance
    for var in co_vars_all:
        if dataset_analysis[var].var() == 0:
            co_vars_all.remove(var)
    #
    match_id = dataset_analysis[dataset_analysis['outcome_']==1][match_variable]
    dataset_analysis = dataset_analysis[dataset_analysis[match_variable].isin(match_id)]
    try:
        model = PHReg(np.asarray(dataset_analysis['time'],dtype=np.float32),
                      np.asarray(dataset_analysis[[exposure_variable]+co_vars_all],dtype=np.float32),
                      status=np.asarray(dataset_analysis['outcome_'],dtype=np.int32), 
                      strata=np.asarray(dataset_analysis[match_variable],dtype=np.float32))
        model_result = model.fit(method='bfgs',maxiter=300,disp=1)
        if pd.isna(model_result.params[0]) or pd.isna(model_result.bse[0]):
            model = cph.fit(dataset_analysis[['time','outcome_',exposure_variable,match_variable]+co_vars_all],
                            fit_options=dict(step_size=0.2), duration_col='time', event_col='outcome_',strata=[match_variable])
            result_temp = model.summary.loc[exposure_variable]
            result += ['fitted_lifelines','%i/%.2f (%.2f)' % (n_exp,time_exp,n_exp/time_exp),
                        '%i/%.2f (%.2f)' % (n_unexp,time_unexp,n_unexp/time_unexp)]
            result += [x for x in result_temp[['coef','se(coef)','p']]]
        else:
            result += ['fitted','%i/%.2f (%.2f)' % (n_exp,time_exp,n_exp/time_exp),
                                '%i/%.2f (%.2f)' % (n_unexp,time_unexp,n_unexp/time_unexp)]
            result += [model_result.params[0],model_result.bse[0],model_result.pvalues[0]]
    except:
        try:
            model = cph.fit(dataset_analysis[['time','outcome_',exposure_variable,match_variable]+co_vars_all],
                             fit_options=dict(step_size=0.2), duration_col='time', event_col='outcome_',strata=[match_variable])
            result_temp = model.summary.loc[exposure_variable]
            result += ['fitted_lifelines','%i/%.2f (%.2f)' % (n_exp,time_exp,n_exp/time_exp),
                        '%i/%.2f (%.2f)' % (n_unexp,time_unexp,n_unexp/time_unexp)]
            result += [x for x in result_temp[['coef','se(coef)','p']]]
        except Exception as e:
            print(e)
            result += [e,'%i/%.2f (%.2f)' % (n_exp,time_exp,n_exp/time_exp),
                        '%i/%.2f (%.2f)' % (n_unexp,time_unexp,n_unexp/time_unexp)]
    return result
#-------------------------------------------------------------------------------------------------------------------------
values = np.load(path + 'age/df_merged.npy', allow_pickle=True)
columns = np.load(path + 'age/df_merged_columns.npy', allow_pickle=True)
df_matched = pd.DataFrame(values, columns=columns)


co_vars = []
for var in ['civil','famIncome','education']:
    temp = pd.get_dummies(df_matched[var], prefix=var)
    co_vars += [x for x in temp.columns[1::]]
    # co_vars += [x for x in temp.columns if var+'_1' not in x]
    df_matched = pd.concat([df_matched,temp],axis=1)
    
phecode_cate = pd.read_csv(path+ 'originData/phecode_definitions1.2.csv')
phecode_lst = np.array([x for x in phecode_cate.phecode.values])
exc_sex = {}
exc_sex[0] = list(phecode_cate.loc[phecode_cate['sex']=='Female'].phecode.values)
exc_sex[1] = list(phecode_cate.loc[phecode_cate['sex']=='Male'].phecode.values)
phecode_cate_ = phecode_cate.loc[(~phecode_cate['category'].isin(['symptoms','congenital anomalies','pregnancy complications'])) & 
                                (~phecode_cate['category'].isna())]

#level1phecode
phecode_cate_['level'] = phecode_cate_['phecode'].apply(lambda x: 1 if str(x).split('.')[1]=='0' 
                                          else 2 if len(str(x).split('.')[1])==1 
                                          else 3)

phecode_cate_2 = phecode_cate_.loc[phecode_cate_['level']==2]
phecode_cate_ = phecode_cate_.loc[phecode_cate_['level']==1] #level == 2

phecode_lst_2 = np.array([int(x) for x in phecode_cate_2.phecode.values])
phecode_lst_ = np.array([x for x in phecode_cate_.phecode.values if int(x) not in phecode_lst_2])
threshold_phewas = 200
#-------------------------------------------------------------------------------------
result_final = []
np.random.seed(number)
np.random.shuffle(phecode_lst_)
total_length = len(phecode_lst_)
result_final = []
for i in range(len(phecode_lst_)):
    d_ = phecode_lst_[i]
    with open (path+'age/temp.cache','r') as f:
        d_exl = f.read()
    if str(d_) not in d_exl.split(','):
        with open (path+'age/temp.cache','a') as f:
            f.write(',%s' % str(d_))
        progress = (len(d_exl.split(','))+1)/len(phecode_lst_)
        print('%i: %.2f%% in phewas1' % (number,progress*100))
        result_final.append(cox(d_,df_matched,'dia_date',threshold_phewas))
    else:
        continue
phe_result = pd.DataFrame(result_final, columns=['disease','number','describe','exp','unexp',
                                                 'coef','se','p'])
phe_result.to_csv(path + 'age/result/phewas1/cox_result_level1_del_%i.csv' % (number))