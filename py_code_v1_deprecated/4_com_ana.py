# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:41:36 2021

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
from scipy.stats import t
import math
import warnings
warnings.filterwarnings("ignore")
path = r'~/depression/'

def split(lst,num):
    n_split = 5
    n_total = len(lst)
    lower = math.floor(n_total/n_split*num)
    upper = math.floor(n_total/n_split*(num+1))
    return lst[lower:upper]

array = np.load(path + 'age/result/baseline_merged_main_group.npy', allow_pickle=True)
array_columns = np.load(path + 'age/result/baseline_merged_columns_main_group.npy', allow_pickle=True)

df_matched_group = pd.DataFrame(array,columns=array_columns)
threshold = 20

phewas_summary = pd.read_csv(path + 'age/result/phewas_summary_L1L2.csv',index_col=0)
disease_list = phewas_summary.loc[phewas_summary['coef']>0]['disease'].values 

d1d2_lst = []
for i in range(len(disease_list)-1):
    for j in range(i+1,len(disease_list)):
        d1d2_lst.append([disease_list[i],disease_list[j]])

d1d2_lst_sub = split(d1d2_lst, number)
len_sub = len(d1d2_lst_sub)
result = []
iter_ = 0

for d1,d2 in d1d2_lst_sub:
    iter_ += 1
    if iter_ % 50 == 0:
        print('%i %.2f%% in commorbidity analysis' % (number, iter_/len_sub*100))
    df_matched_group['flag'] = df_matched_group['d_eligible'].apply(lambda x: d1 in x and d2 in x)
    df_ = df_matched_group.loc[df_matched_group['flag']==True]
    n = len(df_)
    c = sum([d1 in x and d2 in x for x in df_['inpatient_level1'].values]) #d1d2
    if c<=threshold:
        result.append([d1,d2,c,n,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN])
        continue
    p1 = sum([d1 in x for x in df_['inpatient_level1'].values])
    p2 = sum([d2 in x for x in df_['inpatient_level1'].values])
    rr = (n*c)/(p1*p2) #
    theta = (1/c + 1/((p1*p2)/n) - 1/n - 1/n)**0.5 #RR
    #theta = ((n-p1*p2/n)/(p1*p2/n)/n + (n-c)/c/n)**0.5 #RR-
    t_ = abs(np.log(rr)/theta) #RRt
    p = (1-t.cdf(t_,n))*2 #RRP
    #rr_lower,rr_upper = rr*np.exp(-1.96*theta), rr*np.exp(1.96*theta)
    #rr_lower,rr_upper = rr*np.exp(-1.96*theta_), rr*np.exp(1.96*theta_)
    #phi = (c*(n-p1-p2+c)-(p1-c)*(p2-c))/(((p1*p2)*(n-p1)*(n-p2))**0.5)
    phi = (c*n-p1*p2)/(((p1*p2)*(n-p1)*(n-p2))**0.5) #phi
    z_phi = 0.5*np.log((1+phi)/(1-phi))
    z_phi_theta = (1/(n-3))**0.5
    z_phi_t = abs(z_phi/z_phi_theta)
    p_phi = (1-t.cdf(z_phi_t,n))*2
    #t_phi = abs((phi*(max(p1,p2)-2)**0.5)/((1-phi**2)**0.5)) #phip
    #p_phi = (1-t.cdf(t_phi,max(p1,p2)))*2 #p
    result.append([d1,d2,c,n,rr,theta,p,phi,p_phi])
result_df = pd.DataFrame(result,columns=['d1','d2','n_d1d2','N','RR','se','p_rr','phi','p_phi']) # amount 16110
result_df.to_csv(path + 'age/result/comorbidityResult/comorbidity_%i.csv' % (number))