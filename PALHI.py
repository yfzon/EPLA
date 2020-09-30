#!/usr/bin/env python
# encoding: utf-8
'''
@time: 2020/9/22 10:27
@authors: Fan Yang
@copywrite: Tencent
'''

import os
import gc
import time
import logging
import numpy as np
import pandas as pd
import argparse
import joblib
from xgboost.sklearn import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', default='.')
parser.add_argument('--llh_file', default='pred.csv')
parser.add_argument('--log_file', default='PALHI.log')
parser.add_argument('--model_file', default='palhi.pickle.dat')
args = parser.parse_args()

model_folder = args.model_folder
llh_file = os.path.join(model_folder, args.llh_file)
log_file = os.path.join(model_folder, args.log_file)
model_file = os.path.join(model_folder, args.model_file)

def set_log(logfileName='./dgLog.log', level=logging.INFO):
    logging.basicConfig(
        level=level, format='%(asctime)s: %(message)s', 
        handlers=[
            logging.FileHandler(logfileName, mode='w'),
            logging.StreamHandler()
        ])
    
def genPatientIdxDict(patient_ID):
    ''' generate patient->patches index dict
    '''
    patient_idx_dict = {}
    unique_patient, unique_patient_idx = np.unique(patient_ID, return_index=True)
    for p in unique_patient:
        patient_idx_dict[p] = np.where(patient_ID == p)[0]

    return patient_idx_dict, unique_patient_idx

def loadLikelihood_test(llh_file):
    '''
        read the likelihood list according to patch_name
        llh_file: likelihood file
                  Sample.ID,PScore,PROI_X,PROI_Y,x,y
                  Sample.ID: 'TCGA-AA-3510-01A-01-BS1_17858_8929_0.png'
    '''
    llh_tbl = pd.read_csv(llh_file, header=0, index_col=None)
    llh_tbl['patch_name'] = llh_tbl['Sample.ID'].apply(
        lambda x: x.split('.')[0])
    llh_tbl['Patient_ID'] = llh_tbl['patch_name'].apply(
        lambda x: '-'.join(str(x).split('-')[:3]) if x[:4]=='TCGA' else '-'.join(str(x).split('-')[2:5]))
    logging.info('We have {:} patients'.format(len(np.unique(llh_tbl['Patient_ID']))))

    test_llh_tbl = llh_tbl.sort_values(by=['Patient_ID'])

    logging.info('We have {:} patients'.format(len(np.unique(test_llh_tbl['Patient_ID']))))

    te_data = {'patient_ID':test_llh_tbl['Patient_ID'].values,
               'patch_name':test_llh_tbl['patch_name'].values,
               'likelihood':test_llh_tbl['PScore'].values}
    return te_data

def genLikelihoodHist(likelihood, patient_ID, num_bin, norm_hist = False):
    '''
        likelihood: (num_patch, )
        patient_ID: (num_patch, )
        num_bin: euqal size [0, 1]
        norm_hist: whether to normalize each hist
    return
        patient_hist: (num_unique_patient, num_bin)
        unique_patient_idx: (num_unique_patient, )
    '''
    bins = [-float('Inf')]
    bins.extend([i/num_bin for i in range(1, num_bin)])
    bins.append(float('Inf'))
    
    patient_idx_dict, unique_patient_idx = genPatientIdxDict(patient_ID)
    patient_hist = np.zeros((len(unique_patient_idx), num_bin))
    for i in range(len(unique_patient_idx)):
        idx = patient_idx_dict[patient_ID[unique_patient_idx[i]]]
        patient_hist[i,:] = np.histogram(likelihood[idx], bins = bins)[0]
        if norm_hist:
            patient_hist[i,:] = patient_hist[i,:] / np.sum(patient_hist[i,:])
    return patient_hist, unique_patient_idx

def genWsiDf_test(te_data, te_unique_patient_idx, te_pred_prob, te_pred_label):
    ''' columns: Sample.ID,  Patch.Num, WSI.Score, WSI.pred
    '''
    wsi_score = np.array(te_pred_prob)
    wsi_pred = np.array(te_pred_label)
    sample_ID = np.array(te_data['patient_ID'][te_unique_patient_idx])
    te_patch_num = np.zeros(len(te_unique_patient_idx), dtype=int)

    for i in range(len(te_unique_patient_idx)):
        idx = te_data['patient_ID'] == te_data['patient_ID'][te_unique_patient_idx[i]]
        te_patch_num[i] = np.sum(idx)

        
    patch_num = np.array(te_patch_num)

    wsi_pred_df = pd.DataFrame({'Sample.ID':sample_ID, 'Patch.Num':patch_num,
                                'WSI.Score':wsi_score, 'WSI.pred':wsi_pred})

    return wsi_pred_df

def PALHI_inference(te_data, clf, num_bin=200, norm_hist=False):
    ''' PAtch Likelihood HIstogram pipeline
        tr_data, te_data: dict with 'patient_ID', 'MSI_label', 'MSI_score', 'patch_name', 'likelihood'
        cls_model:
        num_bin:
        norm_hist:
    '''
    te_patient_hist, te_unique_patient_idx = genLikelihoodHist(te_data['likelihood'], te_data['patient_ID'],
                                                               num_bin, norm_hist)

    te_pred_label = clf.predict(te_patient_hist)
    te_pred_prob = clf.predict_proba(te_patient_hist)[:,1]

    gc.collect()
    
    wsi_pred_df =  genWsiDf_test(te_data, te_unique_patient_idx, te_pred_prob, te_pred_label)

    return wsi_pred_df

if __name__ == "__main__":
    ''' One file needed
        llh_file: .csv file, patch llh generated  (Sample.ID (patch png name), PScore)

    '''
    set_log(log_file)

    logging.info('lilelihood file: {:}'.format(os.path.basename(llh_file)))

    ''' PIPELINE START FROM HERE
    '''
    te_data = loadLikelihood_test(llh_file)

    since = time.time()
    clf = XGBClassifier()
    clf = joblib.load(os.path.join(model_folder, 'palhi.model'))

    wsi_pred_df = PALHI_inference(te_data, clf)

    wsi_pred_file, _ = os.path.splitext(log_file)
    wsi_pred_file = wsi_pred_file+'_wsi_pred.csv'
    wsi_pred_df.to_csv(wsi_pred_file, sep=',', index=False)

    time_elapsed = time.time() - since
    logging.info('Predicting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    logging.shutdown()
                