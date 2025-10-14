# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 19:36:05 2025

@author: Can Hou
"""
import pandas as pd
import numpy as np
import DiseaseNetPy as dnt

n_cores = 30
interval_days = 60
matching_n = 5

path_processed = r'~/dnt_data/primary_age45/'
path_result = r'~/result/primary_age45/'

if __name__ == "__main__":
    #step 0
    col_dict = {
    'Participant ID': 'eid',          # Maps the participant identifier
    'Exposure': 'outcome',                  # Defines exposure status (0 or 1)
    'Sex': 'sex',                           # Indicates sex (1 for female, 0 for male)
    'Index date': 'dia_date',             # Start date of the study
    'End date': 'time_end',                 # End date of the study
    'Match ID': 'match_2'        # Identifier for matching group
    }
    vars_lst = ['age', 'civil', 'famIncome', 'education']  # List of covariates to be used
    # Initialize the data object with study design and phecode level
    data = dnt.DiseaseNetworkData(
        study_design='matched cohort',          # Type of study design
        phecode_level=2,                        # Level of phecode (1 or 2)
        date_fmt='%Y-%m-%d'                     # Date format in data files
    )
    # Load the phenotype CSV file into the data object
    data.phenotype_data(
        phenotype_data_path=rf'{path_processed}/dep_age_sample.csv',  # Path to phenotype data
        column_names=col_dict,                                   # Column mappings
        covariates=vars_lst,                                      # Covariates to include
        is_single_sex=True
    )
    #merge with medical records data
    #main or all diagnosis
    import os
    merge_path = rf'{path_processed}/medical_records'
    for file in os.listdir(merge_path):
        if 'icd9' in file:
            file_path = os.path.join(merge_path,file)
            data.merge_medical_records(medical_records_data_path=file_path,
                                       diagnosis_code='ICD-9-WHO',
                                       column_names={'Participant ID': 'eid', 
                                                     'Diagnosis code': 'diagnosis', 
                                                     'Date of diagnosis': 'x_indatuma'})
        elif 'icd10' in file:
            file_path = os.path.join(merge_path,file)
            data.merge_medical_records(medical_records_data_path=file_path,
                                       diagnosis_code='ICD-10-WHO',
                                       column_names={'Participant ID': 'eid', 
                                                     'Diagnosis code': 'diagnosis', 
                                                     'Date of diagnosis': 'x_indatuma'})
    data.save(rf'{path_processed}/dep')

    #data loading
    data = dnt.DiseaseNetworkData(phecode_level=2)
    data.load(rf'{path_processed}/dep')


    #step 1
    phewas_result = dnt.phewas(
        data=data,                                             # DiseaseNetworkData object
        n_threshold=250,                            # Minimum proportion of cases to include
        n_process=n_cores,                                          # Number of parallel processes
        system_exl=[                                           # Phecode systems to exclude
            'pregnancy complications','congenital anomalies', 'symptoms', 'others'],
        covariates=['age', 'civil', 'famIncome', 'education'],  # Covariates to adjust for
        lifelines_disable=False,                                # Disable lifelines for faster computation
        log_file=path_result+'/phewas.log'                # Path to log file
    )
    phewas_result = dnt.phewas_multipletests(phewas_result,correction='fdr_bh')
    phewas_result.to_csv(f'{path_result}/phewas.csv',index=False)

    #step 2
    #filter on phecode with HR>1
    phewas_result = pd.read_csv(rf'{path_result}/phewas.csv')
    phewas_result = phewas_result[phewas_result['phewas_coef'] > 0]
    phewas_result = dnt.phewas_multipletests(phewas_result,correction='fdr_bh')
    print('Phewas: ',len(phewas_result[phewas_result['phewas_p_significance']==True]))

    data.disease_pair(phewas_result=phewas_result,min_interval_days=interval_days,force=True)
    #data.save(rf'{path_processed}/dep_withtra')
    
    #step 3
    com_strength_result = dnt.comorbidity_strength(
        data=data,                                     # DiseaseNetworkData object
        n_threshold=25,                   # Minimum proportion for comorbidity
        n_process=n_cores,                                   # Number of parallel processes
        log_file=path_result+'/comorbidity.log'          # Path to log file
    )
    com_strength_result.to_csv(f'{path_result}/comorbidity_strength.csv',index=False)
    com_strength_result = com_strength_result[(com_strength_result['phi'] > 0) & (com_strength_result['RR'] > 1)]
    com_strength_result = dnt.comorbidity_strength_multipletests(df=com_strength_result,
                                                                 correction_phi='fdr_bh',correction_RR='fdr_bh',
                                                                 cutoff_phi=0.05,cutoff_RR=0.05)
    print('Comorbidity strength: ',len(com_strength_result[(com_strength_result['RR_p_significance']==True) & 
                                                       (com_strength_result['phi_p_significance']==True)]))
    
    #step 4
    comorbidity_result = dnt.comorbidity_network(
        data=data,                                       # DiseaseNetworkData object
        comorbidity_strength_result=com_strength_result, # Comorbidity strength results
        n_process=n_cores,                                   # Number of parallel processes
        covariates=['age', 'sex', 'civil', 'famIncome', 'education'],  # Covariates to adjust for
        method='RPCN',                                   # Analysis method ('CN', 'PCN_PCA', 'RPCN')
        log_file=path_result+'/uncond_logistic.log'         # Path to log file
    )
    comorbidity_result = dnt.comorbidity_multipletests(df=comorbidity_result,correction='fdr_bh',cutoff=0.05)
    print('Comorbidity network: ',len(comorbidity_result[comorbidity_result['comorbidity_p_significance']==True]))
    comorbidity_result.to_csv(f'{path_result}/comorbidity_RPCN.csv',index=False)
    
    #step 5
    binomial_result = dnt.binomial_test(
    data=data,                                        # DiseaseNetworkData object
    comorbidity_strength_result=com_strength_result,  # Comorbidity strength results
    n_process=1,                                      # Number of CPU cores (1 to disable multiprocessing)
    enforce_temporal_order=True,                      # Enforce temporal order in testing
    log_file=path_result+'/binomial.log'              # Path to log file
    )
    binomial_result = dnt.binomial_multipletests(df=binomial_result, correction='fdr_bh', cutoff=0.05)
    print('Binomial: ',len(binomial_result[binomial_result['binomial_p_significance']==True]))
    binomial_result.to_csv(f'{path_result}/binomial.csv',index=False)
    
    #step 6
    trajectory_result = dnt.disease_trajectory(
        data=data,                                       # DiseaseNetworkData object
        comorbidity_strength_result=com_strength_result, # Comorbidity strength results
        binomial_test_result=binomial_result,           # Binomial test results
        method='RPCN',                                   # Trajectory analysis method ('CN', 'PCN_PCA', 'RPCN')
        n_process=n_cores,                                     # Number of parallel processes
        matching_var_dict={'age': 1, 'sex': 'exact', 'famIncome': 'exact'},    # Matching variables and criteria
        matching_n=matching_n,                                    # Number of matched controls per case
        enforce_time_interval=True,                     # Enforce time interval in trajectory analysis
        covariates=['age', 'civil', 'education'],  # Covariates to adjust for
        log_file=path_result+'/cond_logistic.log'            # Path to log file
    )
    trajectory_result = dnt.trajectory_multipletests(df=trajectory_result,correction='fdr_bh',cutoff=0.05)
    print('Trajectory analysis: ',len(trajectory_result[trajectory_result['trajectory_p_significance']==True]))
    trajectory_result.to_csv(f'{path_result}/trajectory_match{matching_n}.csv',index=False)
