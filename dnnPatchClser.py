#!/usr/bin/env python
# encoding: utf-8
'''
@time: 2020/9/22 10:27
@authors: Fan Yang
@copywrite: Tencent
'''

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms

import os
import time
import logging
import argparse
import pandas as pd
from glob import glob
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', default='.')
parser.add_argument('--test_data_search_path', default='.')
parser.add_argument('--gt_table', default='gt_tbl.csv')
parser.add_argument('--model_name', default='dnnPatchClser.pt')

args = parser.parse_args()

model_folder = args.model_folder
test_data_search_path = args.test_data_search_path
gt_table = args.gt_table
model_path = os.path.join(model_folder, args.model_name)

NUM_WORKERS = 1
taskConfig = dict()
taskConfig['batch_size'] = 4

os.environ['TORCH_HOME'] = ''

def set_log(logfileName='./dgLog.log', level=logging.INFO):
    logging.basicConfig(
        level=level, format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(logfileName),
            logging.StreamHandler()
        ])

torch.manual_seed(823)
np.random.seed(823)


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

if not os.path.exists(model_folder):
    os.mkdir(model_folder)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TCGADataSet():
    def __init__(self):

        gt_tbl_csv = os.path.join(model_folder, gt_table)

        gt_tbl_updated = pd.read_csv(gt_tbl_csv)

        self.gt_tbl_updated = gt_tbl_updated

    def prepPatchGTTbl(self):
        ''' gt_tbl_patch:
            Patient_ID: image path
            Sample.ID: common column to merge two tables
        '''
        test_image_fullpath = glob(
            test_data_search_path.rstrip(os.sep) + os.sep + '*.png')
        print(test_image_fullpath)
        print('Prepare all patches from te data folder ...')
        gt_tbl_patch = pd.DataFrame(
            {'Patch.ID': np.unique(np.array(test_image_fullpath))})

        gt_tbl_patch['Sample.ID'] = gt_tbl_patch['Patch.ID'].apply(lambda x: str(os.path.basename(x))[
                                                                             : 15])
        print('Extract sample ID from patch ID, as following...')
        print(gt_tbl_patch.head(3))
        gt_tbl_patch = gt_tbl_patch.merge(self.gt_tbl_updated, how='inner', on=[
            'Sample.ID'])
        print(gt_tbl_patch.head(10))
        print(gt_tbl_patch.tail(10))

        return gt_tbl_patch

tcgaDS = TCGADataSet()
gt_tbl_patch = tcgaDS.prepPatchGTTbl()

class DatasetMSI(torch.utils.data.Dataset):

    def __init__(self, summarize_tbl, transform=None):
        ''' dataset class for loader and iterator, noted the format of image is NCWH
        '''
        self.data = summarize_tbl
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)

        image_name = self.data.iloc[index, :]['Patch.ID']

        with open(image_name, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        image_base_name = str(os.path.basename(image_name))
        return image, image_base_name

def predict(model, dataloaders):
    '''return predict_DF_set with Sample.ID, PScore, x, y'''
    since = time.time()
    was_training = model.training
    model.eval()
    predict_DF_set = pd.DataFrame()

    logging.info('start predict on')
    for i, (inputs, patch_name) in enumerate(dataloaders['test']):

        logging.debug(inputs)
        logging.debug('inputs {:} size = {:}'.format(
            type(inputs), inputs.shape))

        logging.debug(patch_name)

        inputs = inputs.to(device)

        outputs = model(inputs)
        score = torch.sigmoid(outputs)
        score = score[:, 1]

        pp = score.data.cpu().detach().numpy()

        predict_DF_set = predict_DF_set.append(pd.DataFrame({
            'Sample.ID': list(patch_name), 'PScore': pp}))

    print('Have {:>10} predictions...'.format(len(predict_DF_set)))
    print(predict_DF_set.head(10))

    predict_DF_set = predict_DF_set.drop_duplicates(
        subset='Sample.ID', keep='first')

    xy_tuple_list = list(predict_DF_set['Sample.ID'].apply(
        lambda x: (str(x).split('_')[1], str(x).split('_')[2])))
    x_list, y_list = zip(*xy_tuple_list)
    predict_DF_set['x'], predict_DF_set['y'] = list(x_list), list(y_list)

    print('Have {:} predictions...'.format(len(predict_DF_set)))

    time_elapsed = time.time() - since
    print('Predicting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(predict_DF_set.head(10))
    model.train(mode=was_training)
    logging.info('Done')
    return predict_DF_set

def build_model_on(device):
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    return model_ft

def prepIO():

    image_datasets = {'test': DatasetMSI(gt_tbl_patch, data_transforms['test'])}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], num_workers=NUM_WORKERS,
                                                  shuffle=True, batch_size=taskConfig['batch_size'])
                   for x in ['test']}

    return dataloaders, dataset_sizes

if __name__ == '__main__':

    dataloaders, dataset_sizes = prepIO()

    model_ft = build_model_on(device)

    with open(model_path, 'rb') as f:
        best_model_wts = torch.load(f, map_location='cpu')

    model_ft.load_state_dict(best_model_wts['model_state_dict'])

    assert model_folder != ''

    result_pd = os.path.join(model_folder, 'pred.csv')

    predict_DF_set = predict(model_ft, dataloaders)
    predict_DF_set.to_csv(result_pd, encoding='utf-8', index=False)


