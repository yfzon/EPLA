# Development and interpretation of a pathomics-based model for the prediction of microsatellite instability in Colorectal Cancer

This package provides an implementation of the prediction of microsatellite instability in whole slide imaging of Colorectal Cancer patients using deep learning

## Citation
Cao R, Yang F, Ma SC, Liu L, Zhao Y, Li Y, Wu DH, Wang T, Lu WJ, Cai WJ, Zhu HB, Guo XJ, Lu YW, Kuang JJ, Huan WJ, Tang WM, Huang K, Huang J, Yao J, Dong ZY. Development and interpretation of a pathomics-based model for the prediction of microsatellite instability in Colorectal Cancer. Theranostics 2020; 10(24):11080-11091. doi:10.7150/thno.49864. Available from http://www.thno.org/v10p11080.htm 

## Setup

### Dependencies

Processing pipelines are implemented in python.

#### Python 3.6+
* torch 1.1.0
* torchvision 0.2.1
* numpy 1.15.2
* pandas 1.0.3
* xgboost 0.90
* pillow 5.3.0
* sklearn 0.23.1
* logging 0.5.1.2
* joblib 0.15.1

* pickle 4.0

#### Requirements to run the algorithm and average test time
* 0.5118 s/patient using Nvidia GPUs (P40) 
* 20.9291 s/patient using regular CPU machines

## Data

Data are in png format, and the patient info table is in csv format. Results will be saved in csv format.

### Input patient info table format

A csv file in the following formate is needed for prediction:

patient_ID  |
---|
TCGA-AA-3812-01 |
TCGA-AA-A00E-01 |
TCGA-AA-A01Q-01 |
TCGA-AA-A02R-01 |

If you use other kind of names, please change related codes.
There are also some cropped patches in the folder to help you go through the whole process.

### Model checkpoints

All models and feature matrix needed for processing the interpretation is stored in this fold.
* dnnPatchClser.pt
* bow.model
* palhi.model
* bow_feature.pkl
* bow_tfidftransformer.pkl


## Calculate the MSI likelihood of each patch (Step 1)

Prediction on the patch level code will make prediction on each patch. To run the code:
```
python dnnPatchClser.py --model_folder=. --test_data_search_path=. --gt_table=gt_tbl.csv --model_name=dnnPatchClser.pt
```
The "model_folder" is the current folder, the model and codes can be store in the same folder. 
The "test_data_search_path" is the folder path that stores image data. 
The "gt_table" is the patient info table and the format is shown above.
The "model_name" should be the checkpoint of the patch-level model.
A file named "pred.csv" will be saved containing the patch-level prediction results. Use this file as the input of next step.

## Prediction on the patient level (Step 2)

This prediction code will predict MSI probability for each patient.
### PALHI (Step 2.1)
```
python PALHI.py --model_folder=. --llh_file=pred.csv --log_file=PALHI.log --model_file=palhi.pickle.dat
```
The precess will be logged in 'log_file'. The model needed is provided in this github and should be stored in the "model_folder". A file named "PALHI_wsi_pred.csv" will be saved containing the patch-level prediction results. Use this file as the input of Step3.

### BOW (Step 2.2)

```
python BOW.py --model_folder=. --llh_file=pred.csv --log_file=BOW.log --feature_file=bow_feature.pkl --tfidftransformer_file=bow_tfidftransformer.pkl --model_file=bow.pickle.dat
```
The precess will be logged in 'log_file'. The model needed is provided in this github and should be stored in the "model_folder". A file named "BOW_wsi_pred.csv" will be saved containing the patch-level prediction results. Use this file as the input of Step3.

### Ensemble (Step 2.3)

```
python ensemble.py --model_folder=. --PALHI_tbl=PALHI_wsi_pred.csv --BOW_tbl=BOW_wsi_pred.csv 

```
The input "PALHI_tbl" and "BOW_tbl" is generated in Step1 and Step2. A file named "EPLA_output.csv" is generated as the final output.

5 values are generated:

Value| Explaination              
--- | ---
Sample.ID   | The input patient ID      
WSI.Score_x | MSI probability from PALHI
WSI.pred_x  | MSI status from BOW       
WSI.Score_y | MSI probability from BOW  
WSI.pred_y  | MSI status from BOW       
WSI.Score   | EPLA final MSI probability
WSI.pred    | EPLA final MSI status     

The threshold is 0.5 by default and can be reset using optimal threshold.

## Disclaimer
This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.
