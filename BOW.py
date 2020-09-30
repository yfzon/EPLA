#!/usr/bin/env python
# encoding: utf-8
'''
@time: 2020/9/22 10:27
@authors: Fan Yang
@copywrite: Tencent
'''

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from sklearn.naive_bayes import MultinomialNB
import sklearn.feature_extraction.text as ft
from PALHI import genWsiDf_test, loadLikelihood_test, genPatientIdxDict

np.random.seed(1228)

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', default='.')
parser.add_argument('--llh_file', default='pred.csv')
parser.add_argument('--log_file', default='BOW.log')
parser.add_argument('--feature_file', default='bow_feature.pkl')
parser.add_argument('--tfidftransformer_file', default='bow_tfidftransformer.pkl')
parser.add_argument('--model_file', default='bow.pickle.dat')
args = parser.parse_args()

model_folder = args.model_folder
llh_file = os.path.join(model_folder, args.llh_file)
log_file = os.path.join(model_folder, args.log_file)
feature_path = os.path.join(model_folder,args.feature_file)
tfidftransformer_path = os.path.join(model_folder, args.tfidftransformer_file)


precision = 6

def set_log(logfileName='./dgLog.log', level=logging.INFO):
    logging.basicConfig(
        level=level, format='%(asctime)s: %(message)s', 
        handlers=[
            logging.FileHandler(logfileName, mode='w'),
            logging.StreamHandler()
        ])

def genBoW(data, precision):
    ''' data is a dict with 'patient_ID', 'patch_name', 'likelihood'
        precision: precision of BoW
    '''
    corpus_list = []
    sample_id_list = []
    patch_no_list = []
    
    patient_idx_dict, unique_patient_idx = genPatientIdxDict(data['patient_ID'])
    
    for i in range(len(unique_patient_idx)):
        pid = data['patient_ID'][unique_patient_idx[i]]
        llh = data['likelihood'][patient_idx_dict[pid]]
        llh = llh.tolist()
        words = ' '.join(["{0:.{1}f}".format(x, precision) for x in llh])
        wsi_patches = len(patient_idx_dict[pid])
        corpus_list.append(words)
        sample_id_list.append(pid)
        patch_no_list.append(wsi_patches)

    return unique_patient_idx, corpus_list, sample_id_list, patch_no_list

def BOW(te_data, cv, tf, precision, model):
    # generate corpus
    te_unique_patient_idx, corpus_te, sample_id_te, patch_no_te = genBoW(te_data, precision)
    
    test_tfmat = cv.transform(corpus_te)
    test_x = tf.transform(test_tfmat)

    logging.info("Results on testing set")

    te_pred_label = model.predict(test_x)
    te_pred_prob = model.predict_proba(test_x)[:, 1]

    wsi_pred_df =  genWsiDf_test(te_data, te_unique_patient_idx, te_pred_prob, te_pred_label)

    return wsi_pred_df
    

if __name__ == "__main__":
    set_log(log_file)
    logging.info(model_folder)
    logging.info('lilelihood file: {:}'.format(os.path.basename(llh_file)))

    ''' PIPELINE START FROM HERE
    '''
    te_data = loadLikelihood_test(llh_file)

    since = time.time()
    model = joblib.load(os.path.join(model_folder, 'bow.model'))
    # load vector feature
    cv = ft.CountVectorizer(decode_error="replace", vocabulary=pkl.load(open(feature_path, "rb")))
    tf = pkl.load(open(tfidftransformer_path, "rb"))

    wsi_pred_df = BOW(te_data, cv, tf, precision, model)

    # save wsi pred
    wsi_pred_file, _ = os.path.splitext(log_file)
    wsi_pred_file = wsi_pred_file+'_wsi_pred.csv'
    wsi_pred_df.to_csv(wsi_pred_file, sep=',', index=False)

    time_elapsed = time.time() - since
    logging.info('Predicting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    logging.shutdown()    
                