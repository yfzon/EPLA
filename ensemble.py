#!/usr/bin/env python
# encoding: utf-8
'''
@time: 2020/9/22 10:27
@authors: Fan Yang
@copywrite: Tencent
'''

from __future__ import print_function, division
import os
import time
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--PALHI_tbl', default='PALHI_wsi_pred.csv')
parser.add_argument('--BOW_tbl', default='BOW_wsi_pred.csv')
parser.add_argument('--model_folder', default='.')

args = parser.parse_args()

model_folder = args.model_folder
PALHI_csv = os.path.join(model_folder, args.PALHI_tbl)
BOW_csv = os.path.join(model_folder, args.BOW_tbl)


if __name__ == "__main__":

    since = time.time()

    modelCSVs = [PALHI_csv, BOW_csv]
    modelPredDFs = pd.read_csv(modelCSVs[0])

    modelPredDFs['Sample.ID'] = modelPredDFs['Sample.ID'].apply(lambda x: str(os.path.basename(x))[:12])
        
    for idx, predCSV in enumerate(modelCSVs):
        if idx > 0:
            modelPredDFs = modelPredDFs.merge(
                pd.read_csv(predCSV), how='outer', on=['Sample.ID'])

    colNames = list(modelPredDFs)
    colOfIns = [x for x in colNames if x[:9] == 'WSI.Score']
    print(np.array(colOfIns))

    weights = [0.5, 0.5]
    youden_criterion = 0.5 #could be custom

    modelPredScores = modelPredDFs[colOfIns].apply(
         lambda x: np.inner(x, np.array(weights)), axis=1)

    modelPredDFs['WSI.Score'] = modelPredScores
    modelPredDFs['WSI.pred'] = modelPredDFs['WSI.Score'].apply(lambda x: 1 if x >= youden_criterion else 0)

    print(modelPredDFs.head(10))
    modelPredDFs.to_csv(os.path.join(
        model_folder, 'EPLA_output.csv'), encoding='utf-8', index=False)

    time_elapsed = time.time() - since
    print('Predicting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))