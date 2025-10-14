# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 19:36:05 2025

@author: Can Hou
"""
import pandas as pd
import numpy as np
import DiseaseNetPy as dnt

n_cores = 10
interval_days = 60
matching_n = 5

path_processed = r'~/UKB_validation/dnt_data'
path_result = r'~/UKB_validation/result'
path_result_sweden = r'~/result/primary_age45/'


if __name__ == "__main__":
    #step 0
    col_dict = {
    'Participant ID': 'new_index',          # Maps the participant identifier
    'Exposure': 'outcome',                  # Defines exposure status (0 or 1)
    'Sex': 'sex',                           # Indicates sex (1 for female, 0 for male)
    'Index date': 'date_start',             # Start date of the study
    'End date': 'time_end_chort',                 # End date of the study
    'Match ID': 'match_2'        # Identifier for matching group
    }
    vars_lst = ['age', 'social_cate', 'income', 'diet', 
                'BMI','drinking','smoking','physical']  # List of covariates to be used

    # Initialize the data object with study design and phecode level
    data = dnt.DiseaseNetworkData(
        study_design='matched cohort',          # Type of study design
        phecode_level=2,                        # Level of phecode (1 or 2)
        date_fmt='%Y-%m-%d'                     # Date format in data files
    )
    # Load the phenotype CSV file into the data object
    data.phenotype_data(
        phenotype_data_path=rf'{path_processed}/dep_matched_exl.csv',  # Path to phenotype data
        column_names=col_dict,                                   # Column mappings
        covariates=vars_lst                                      # Covariates to include
    )
    #merge with medical records data
    #ICD-9
    data.merge_medical_records(medical_records_data_path=rf'{path_processed}/icd9.csv',
                               diagnosis_code='ICD-9-WHO',
                               column_names={'Participant ID': 'new_index', 
                                             'Diagnosis code': 'diag_icd9', 
                                             'Date of diagnosis': 'dia_date'})
    data.merge_medical_records(medical_records_data_path=rf'{path_processed}/icd10.csv',
                               diagnosis_code='ICD-10-WHO',
                               column_names={'Participant ID': 'new_index', 
                                             'Diagnosis code': 'diag_icd10', 
                                             'Date of diagnosis': 'dia_date'})
    #save
    data.save(rf'{path_processed}/dep')
    
    #data loading
    data = dnt.DiseaseNetworkData(phecode_level=2)
    data.load(rf'{path_processed}/dep')
    
    #step 1 - phewas
    #phewas validation
    swenden_phewas_results = pd.read_csv(rf'{path_result_sweden}/phewas.csv')
    #swedish results filtering
    swenden_phewas_results = swenden_phewas_results[swenden_phewas_results['phewas_coef'] > 0]
    swenden_phewas_results = dnt.phewas_multipletests(swenden_phewas_results,correction='fdr_bh')
    phecode_lst = swenden_phewas_results[swenden_phewas_results['phewas_p_significance']==True]['phecode'].to_list()
    
    #step 1
    phewas_result = dnt.phewas(
        data=data,                                             # DiseaseNetworkData object
        n_threshold=1,                            # Minimum proportion of cases to include
        n_process=n_cores,                                          # Number of parallel processes
        phecode_inc=phecode_lst,
        covariates=['age', 'income','diet', 
                    'BMI','drinking','smoking','physical'],  # Covariates to adjust for
        lifelines_disable=False,                                # Disable lifelines for faster computation
        correction='none',
        log_file=path_result+'/phewas.log'                # Path to log file
    )
    phewas_result.to_csv(rf'{path_result}/phewas_validation.csv',index=False)
    
    #step 2
    #load phewas results
    phewas_result = pd.read_csv(rf'{path_result}/phewas_validation.csv')
    phecode_validated = phewas_result[phewas_result['phewas_p_significance']==True]['phecode'].to_list()
    data.disease_pair(phewas_result=phewas_result,min_interval_days=interval_days,force=True)

    #step 3
    com_swenden_result = pd.read_csv(rf'{path_result_sweden}/comorbidity_RPCN.csv')
    com_swenden_result = com_swenden_result[(com_swenden_result['comorbidity_p_significance']==True) &
                                            (com_swenden_result['comorbidity_beta']>0)]
    #select eligible
    com_swenden_result = com_swenden_result[(com_swenden_result['phecode_d1'].isin(phecode_validated)) & 
                                            (com_swenden_result['phecode_d2'].isin(phecode_validated))]
    com_swenden_result['phi_p_significance'] = True
    com_swenden_result['RR_p_significance'] = True

    #comorbidity network validation
    comorbidity_result = dnt.comorbidity_network(
    data=data,                                       # DiseaseNetworkData object
    comorbidity_strength_result = com_swenden_result, # Comorbidity strength results
    n_process=n_cores,                                   # Number of parallel processes
    covariates=['age', 'sex', 'income','social_cate','diet', 
                'BMI','drinking','smoking','physical'],  # Covariates to adjust for
    method='RPCN',                                   # Analysis method ('CN', 'PCN_PCA', 'RPCN')
    correction='none',
    log_file=path_result+'/uncond_logistic.log'         # Path to log file
    )
    comorbidity_result.to_csv(rf'{path_result}/comorbidity_RPCN_validation.csv',index=False)
    
    #step 4
    tra_sweden_result = pd.read_csv(rf'{path_result_sweden}/trajectory_match5.csv')
    tra_sweden_result = tra_sweden_result[(tra_sweden_result['trajectory_p_significance']==True) & 
                                          (tra_sweden_result['trajectory_beta']>=0)]
    tra_sweden_result = tra_sweden_result[(tra_sweden_result['phecode_d1'].isin(phecode_validated)) &
                                          (tra_sweden_result['phecode_d2'].isin(phecode_validated))]
    tra_sweden_result['binomial_p_significance'] = True
    
    trajectory_result = dnt.disease_trajectory(
    data=data,                                       # DiseaseNetworkData object
    comorbidity_strength_result=com_swenden_result, # Comorbidity strength results
    binomial_test_result=tra_sweden_result,           # Binomial test results
    method='RPCN',                                   # Trajectory analysis method ('CN', 'PCN_PCA', 'RPCN')
    n_process=n_cores,                                     # Number of parallel processes
    matching_var_dict={'age': 1, 'sex': 'exact'},    # Matching variables and criteria
    matching_n=matching_n,                                    # Number of matched controls per case
    enforce_time_interval=True,                     # Enforce time interval in trajectory analysis
    covariates=['income','social_cate','diet', 
                'BMI','drinking','smoking','physical'],  # Covariates to adjust for
    correction='none',
    log_file=path_result+'/cond_logistic.log'            # Path to log file
    )
    trajectory_result.to_csv(rf'{path_result}\trajectory_match{matching_n}_RPCN_validation.csv',index=False)

