import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
import shap
import xgboost
import matplotlib.pyplot as plt
import statistics as st
from bayes_opt import BayesianOptimization
from pyirr import intraclass_correlation
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from feature_utils.feature_classfication import data_normalization, data_normalization_apply_cohort1_to_all, lgb_evaluate_lgbm, lgb_evaluate_svm

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from functools import partial
from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve, auc
import sys
import argparse

import csv
import os

def write_experiment_results_to_csv(file_path, data_dict):
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)

def f1_scores(label_f1, pred_f1, pos_index):
    """
    Calculate f1 score（for bootstrap）
    :param data: data[:, 0] = pred, data[:, 1] = label, data[:, 2] = index
    :return: f1 score
    """
    conf_mat = confusion_matrix(y_true=label_f1, y_pred=pred_f1)

    tp = conf_mat[pos_index][pos_index]
    fp = np.sum(conf_mat[:, pos_index]) - tp
    fn = np.sum(conf_mat[pos_index]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_cls = 2 * precision * recall / (precision + recall)
    return f1_cls

def update_value_ep(x):
    if x < 0.1:
        return 0
    elif 0.1 <= x <= 0.5:
        return 0.5
    else:
        return 1
    
def update_value_age(x):
    """
    年龄的encoding：0-3，3-6，6-9，9-12，12-15，15-18，18+, [0, 0.15, 0.3, 0.45, 0.6, 0.75, 1]
    """
    if x < 3:
        return 0
    elif 3 <= x < 6:
        return 0.15
    elif 6 <= x < 9:
        return 0.3
    elif 9 <= x < 12:
        return 0.45
    elif 12 <= x < 15:
        return 0.6
    elif 15 <= x < 18:
        return 0.75
    else:
        return 1
    
def update_value_dsmn(x):
    """
    转移pattern：C-D-（0）；C-D+（0.25）；C+D-（0.5）；C+NA（0.75）；C+D+（1）
    """
    if x ==0:
        return 0
    elif x==0.25:
        return 1.0
    elif x==0.375:
        return 0.75
    elif x==0.5:
        return 0.5
    elif x==0.75:
        return 0.25
    else:
        return 0
    
def update_value_CD(x):
    """
    转移pattern：C-D+（0.25
    """
    if x ==2:
        return 1
    elif x == 1:
        return 0
    else:
        return 0

class DummyFile(object):
    def write(self, x): pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LGBM_clnc_g34')

    parser.add_argument('--radiomics_file', default='/Users/wangpengcheng/Documents/MB_AI/new_g34_radiomics_old.txt')#default='/Users/wangpengcheng/Documents/MB_AI/new_3cls_radiomics.txt')
    parser.add_argument('--radiomics_file_c4', default='/Users/wangpengcheng/Documents/MB_AI/new_g34_radiomics_cohort4_new.txt')#default='/Users/wangpengcheng/Documents/MB_AI/new_3cls_radiomics_cohort4.txt')
    parser.add_argument('--clnc_list', nargs='*', type=str, required=True)
    parser.add_argument('--csv_result', default='/Users/wangpengcheng/Downloads/MB_LGBM_CLNC_PIPE_G34_FINAL/results.csv')
    parser.add_argument('--print', action='store_true', default=False)
    parser.add_argument('--compare', default=None)
    parser.add_argument('--split_ep', action='store_true', default=False)
    parser.add_argument('--split_age', action='store_true', default=False)
    parser.add_argument('--tranf_dsmn', action='store_true', default=False)
    parser.add_argument('--no_age', action='store_true', default=False)
    

    args = parser.parse_args()
    
    os.makedirs(args.csv_result.replace('results.csv', ''), exist_ok=True)
    
    original_stdout = sys.stdout 
    if not args.print:
        sys.stdout = DummyFile()
    
    three_cls_path = args.radiomics_file
    three_cls_path_c4 = args.radiomics_file_c4
    data = pd.read_csv(three_cls_path, sep='\t')
    data_c4 = pd.read_csv(three_cls_path_c4, sep='\t')
    
    # selected features
    wnt_shh_g34_feature_names = ['T1E_log-sigma-3-mm-3D_glrlm_LongRunLowGrayLevelEmphasis_x',
                                 'T1E_log-sigma-1-mm-3D_glrlm_ShortRunHighGrayLevelEmphasis_x',
                                 'T1E_log-sigma-3-mm-3D_glcm_Autocorrelation_x',
                                 'T2_wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis_x',
                                 'T2_wavelet-HHL_glcm_ClusterShade_x',
                                 'T2_wavelet-HHL_firstorder_Skewness_x', 
                                 'T2_wavelet-HHL_glszm_GrayLevelNonUniformityNormalized_y', 
                                 'T1E_lbp-3D-m2_firstorder_TotalEnergy_y', 
                                 'T2_wavelet-LLH_glszm_ZoneEntropy_x', 
                                 'T1E_lbp-3D-k_firstorder_Kurtosis_y']

    clinical_feature_names = args.clnc_list
    
    id_val = data[data['Cohort']!=1]['id'].tolist()
    id_new = data_c4['id'].tolist()
    print('Val id: ', len(id_val))
    print('New id: ', len(id_new))
    #['Location','FourthIn','Enhance','Enhance_Percent','Margin','Water','Cyst','Hydro','EpendymalDissemination','3rdVIRDissemination','SuperLMD','InfraLMD','LMDPattern','Ependymal3Dissemination','LMDDissemination','AllDissemination']
    clnc_all = ['Location','FourthIn','Enhance','Enhance_Percent','Margin','Water','Cyst','Hydro','EpendymalDissemination','3rdVIRDissemination','SuperLMD','InfraLMD','LMDPattern','Ependymal3Dissemination','LMDDissemination','AllDissemination']
    
    if not args.split_age:
        if not args.compare:
            sel_columns = ['Cohort','Subtypes_three','fold','Age'] + clinical_feature_names + wnt_shh_g34_feature_names
        else:
            if 'Age' in args.compare:
                sel_columns_basic = ['Cohort','Subtypes_three','fold','Age']
            else:
                sel_columns_basic = ['Cohort','Subtypes_three','fold']
            if 'Clnc' in args.compare:
                sel_columns = sel_columns_basic + clnc_all
            elif 'Radio' in args.compare:
                sel_columns = sel_columns_basic + wnt_shh_g34_feature_names
            elif 'ClRa' in args.compare:
                sel_columns = sel_columns_basic + clnc_all + wnt_shh_g34_feature_names
            else:
                sel_columns = ['Cohort','Subtypes_three','fold','Age'] + clinical_feature_names + wnt_shh_g34_feature_names
    else:
        if not args.compare:
            sel_columns = ['Cohort','Subtypes_three','fold','AgeOg'] + clinical_feature_names + wnt_shh_g34_feature_names
        else:
            if 'Age' in args.compare:
                sel_columns_basic = ['Cohort','Subtypes_three','fold','AgeOg']
            else:
                sel_columns_basic = ['Cohort','Subtypes_three','fold']
            if 'Clnc' in args.compare:
                sel_columns = sel_columns_basic + clnc_all
            elif 'Radio' in args.compare:
                sel_columns = sel_columns_basic + wnt_shh_g34_feature_names
            elif 'ClRa' in args.compare:
                sel_columns = sel_columns_basic + clnc_all + wnt_shh_g34_feature_names
            else:
                sel_columns = ['Cohort','Subtypes_three','fold','AgeOg'] + clinical_feature_names + wnt_shh_g34_feature_names
                
    if args.no_age:
        if 'Age' in sel_columns:
            sel_columns.remove('Age')
        if 'AgeOg' in sel_columns:
            sel_columns.remove('AgeOg')

    x = data[sel_columns]
    x.iloc[:, 4:] = x.iloc[:, 4:].astype(float)
    print('total features and labels shape: ', x.shape)
    
    x_new = data_c4[sel_columns]
    print(sel_columns, x_new.shape)
    x_new.iloc[:, 3:] = x_new.iloc[:, 3:].astype(float)
    print('total features and labels shape: ', x.shape)
    
    # update values
    if args.split_ep and 'Enhance_Percent' in args.clnc_list:
        x['Enhance_Percent'] = x['Enhance_Percent'].apply(update_value_ep)
        x.iloc[:, 4:] = x.iloc[:, 4:].astype(float)
    elif args.split_ep and args.compare and  'Cl' in args.compare:
        x['Enhance_Percent'] = x['Enhance_Percent'].apply(update_value_ep)
        x.iloc[:, 4:] = x.iloc[:, 4:].astype(float)
        
    if args.split_age and 'AgeOg' in sel_columns:
        x['AgeOg'] = x['AgeOg'].apply(update_value_age)
        x.iloc[:, 4:] = x.iloc[:, 4:].astype(float)
        
    if args.tranf_dsmn:
        for dsmn in ['EpendymalDissemination','3rdVIRDissemination','SuperLMD','InfraLMD','LMDPattern']:
            if dsmn in sel_columns:
                x[dsmn] = x[dsmn].apply(update_value_dsmn)
                x.iloc[:, 4:] = x.iloc[:, 4:].astype(float)
    
    # split base dataset and external validation dataset
    df_val = x[(x['Cohort'] == 2) | (x['Cohort'] == 3)]
    df_val.reset_index(inplace=True)
    X_val = df_val.iloc[:, 4:]

    feature_names = X_val.columns.tolist()
    X_val = np.array(X_val)
    y_val = df_val['Subtypes_three']
    y_val = np.array(y_val)
    
    df_new = x_new
    X_new = df_new.iloc[:, 3:]

    feature_names = X_new.columns.tolist()
    print(feature_names)
    X_new = np.array(X_new)
    y_new = df_new['Subtypes_three']
    y_new = np.array(y_new)

    print('X-val: ', X_val.shape)
    print('X-new: ', X_new.shape)

    df_X = x[x['Cohort'] == 1]
    df_X.reset_index(inplace=True)

    folds_indices = {}

    for fold in df_X['fold'].unique():
        test_index = df_X[df_X['fold'] == fold].index
        train_index = df_X[df_X['fold'] != fold].index
        print(len(test_index), len(train_index))
        folds_indices[fold] = {'train_index': train_index, 'test_index': test_index}

    X = df_X.iloc[:, 4:]
    y = df_X['Subtypes_three']
    print('X: ', X.shape)
    print('Fold index has been determined!')
    
    params_best = []
    for i, values in enumerate(folds_indices.values()):
        train_index, test_index = values['train_index'], values['test_index']
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        y_train = np.array(y_train).astype(int)
        y_test = np.array(y_test).astype(int)

        nm = SMOTE(random_state=42)
        X_res, y_res = nm.fit_resample(X_train, y_train)

        print('size of data partitions')
        print('...............INPUT..............OUTPUT..........')
        print('Train      : ', X_train.shape, '         ', y_train.shape, np.sum(y_train))
        print('Test       : ', X_test.shape, '         ', y_test.shape, np.sum(y_test))
        print('Validation : ', X_val.shape, '         ', y_val.shape, np.sum(y_val))

        # bayesian optimization of hyperparameters for lg_boost
        gp_params = {"alpha": 1e-4}
        seed = 1

        # use partial to pass train, val and test data
        partial_lgb_evaluate_lgbm = partial(lgb_evaluate_lgbm, X_train=X_res, y_train=y_res, X_val=X_val, y_val=y_val,
                                            X_test=X_test, y_test=y_test, seed=1, multi_cls=False)
        lgb_bo = BayesianOptimization(partial_lgb_evaluate_lgbm, {'max_depth': (1, 10),
                                                     'num_leaves': (2,10),
                                                     'learning_rate':(0.01, 0.4),
                                                     'max_bin':(200, 800),
                                                     'colsample_bytree': (0.01, 1.0),
                                                     'reg_alpha':(1,10),
                                                     'reg_lambda':(0.01,1),
                                                     'subsample':(0.01,1),
                                                     'min_child_samples' : (5, 30),
                                                     'min_child_weight':(1, 5),
                                                     'n_estimators' :(500,2200),
                                                     'scale_pos_weight': (1,7),})

        # Optimally needs quite a few more initiation points and number of iterations
        lgb_bo.maximize(init_points=200, n_iter=100, acq='ei',  **gp_params)

        fold_max = lgb_bo.max['params']
        fold_params = {'max_depth': int(fold_max['max_depth']),
                      'num_leaves': int(fold_max['num_leaves']),
                      'learning_rate': fold_max['learning_rate'],
                      'max_bin': int(fold_max['max_bin']),
                      'colsample_bytree': fold_max['colsample_bytree'],
                      'reg_alpha': fold_max['reg_alpha'],
                      'reg_lambda': fold_max['reg_lambda'],
                      'subsample': fold_max['subsample'],
                      'min_child_samples': int(fold_max['min_child_samples']),
                      'min_child_weight': fold_max['min_child_weight'],
                      'n_estimators': int(fold_max['n_estimators']),
                      'scale_pos_weight': fold_max['scale_pos_weight']
                      }
        params_best.append(fold_params)
        
    test_folds = []
    test_gt_folds = []
    val_folds = []
    true_folds = []
    pred_folds = []
    cm_folds = []
    f1_test_folds = []
    auc_test_folds = []
    new_folds = []
    thresholds = []

    for i, values in enumerate(folds_indices.values()):
        train_index, test_index = values['train_index'], values['test_index']
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        y_train = np.array(y_train).astype(int)
        y_test = np.array(y_test).astype(int)

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        y_train = np.array(y_train).astype(int)
        y_test = np.array(y_test).astype(int)

        nm = SMOTE(random_state=42)
        X_res, y_res = nm.fit_resample(X_train, y_train)

        params_cur = params_best[i]
        modellgb = LGBMClassifier()
        modellgb.set_params(**params_cur)
        clf_lg = modellgb.fit(X_res, y_res)
        p_lg_train = clf_lg.predict_proba(X_train)[:,1]
        p_lg_test = clf_lg.predict_proba(X_test)[:,1]
        p_lg_val = clf_lg.predict_proba(X_val)[:,1]
        p_lg_new = clf_lg.predict_proba(X_new)[:,1]
        
        val_folds.append(p_lg_val)
        test_folds.append(p_lg_test)
        test_gt_folds.append(y_test)
        new_folds.append(p_lg_new)
        
        arg_thr = 0
        arg_f1score = 0
        for threshold in np.arange(0.1,0.99,0.001):
            l_lg_train = (p_lg_train > threshold).astype('int')
            l_lg_test = (p_lg_test > threshold).astype('int')
            l_lg_val = (p_lg_val > threshold).astype('int')
            l_lg_new = (p_lg_new > threshold).astype('int')

            f1_train = f1_score(y_train, l_lg_train, average='weighted')
            f1_test = f1_score(y_test, l_lg_test, average='weighted')
            f1_val = f1_score(y_val, l_lg_val, average='weighted')
            f1_new = f1_score(y_new, l_lg_new, average='weighted')
            
            if f1_test >= arg_f1score:
                arg_f1score = f1_test
                arg_thr = threshold
            
        f1_test_folds.append(arg_f1score)
        thresholds.append(arg_thr)

        true_folds.append(y_test)
        pred_folds.append((p_lg_test > arg_thr).astype('int'))

        lg_auc_train = roc_auc_score(y_train, p_lg_train)#, multi_class='ovr')
        lg_auc_test = roc_auc_score(y_test, p_lg_test)#, multi_class='ovr')
        lg_auc_val = roc_auc_score(y_val, p_lg_val)#, multi_class='ovr')
        lg_auc_new = roc_auc_score(y_new, p_lg_new)#, multi_class='ovr')
        auc_test_folds.append(lg_auc_test)

        cm_test = confusion_matrix(y_test, (p_lg_test > arg_thr).astype('int'))
        cm_folds.append(cm_test)

    all_val_folds = np.stack(val_folds)
    all_new_folds = np.stack(new_folds)
    
    mean_folds = np.mean(all_val_folds, axis=0)
    mean_new_folds = np.mean(all_new_folds, axis=0)
    
    #mean_thr = np.mean(thresholds)
    weights = np.array(auc_test_folds) 
    weights = np.maximum(weights - 0.55, 0)
    mean_thr = np.average(np.array(thresholds), weights=weights)
    
    if not args.compare:
            experiment_name = '_'.join(args.clnc_list)
    else:
        experiment_name = args.compare

    if args.split_ep and 'Enhance_Percent' in args.clnc_list:
        experiment_name = experiment_name + '_split_ep'
    elif args.split_ep and args.compare and 'Cl' in args.compare:
         experiment_name = experiment_name + '_split_ep'

    if args.split_age and 'AgeOg' in sel_columns:
        experiment_name = experiment_name + '_split_age'
    elif args.no_age:
        experiment_name = experiment_name + '_no_age'

    if args.tranf_dsmn:
        for dsmn in ['EpendymalDissemination','3rdVIRDissemination','SuperLMD','InfraLMD','LMDPattern']:
            if dsmn in sel_columns:
                experiment_name = experiment_name + '_tranf_dsmn'
                break
                
    from datetime import datetime

    # 获取当前时间
    now = datetime.now()

    # 格式化时间为 MMDDHHMM 格式
    timestamp = now.strftime("%m%d%H%M")

    with open(args.csv_result.replace('results.csv', experiment_name + '_' + timestamp + '.txt'), 'w') as f:
        sys.stdout = f
        print('-------------Features---------------')
        print(sel_columns)

        print('-------------Testing----------------')
        all_test_folds = np.concatenate(test_folds)
        gt_test = np.concatenate(true_folds)
        print('All_test_folds: ', all_test_folds.shape)

        lg_auc_test = roc_auc_score(gt_test, all_test_folds)
        print('Test AUC : ', np.round(lg_auc_test, decimals=4))
        print('Test AUC: ', np.round(np.mean(auc_test_folds), decimals=4), '(mean fold)')
        print('Test AUC: ', np.round(auc_test_folds, decimals=4), '(per fold)')
        # Binarize the true labels to use with roc_auc_score

        f1_test = f1_score(gt_test, np.concatenate(pred_folds), average='weighted')
        print('Test F1 score: ', np.round(f1_test, decimals=4))
        print('Test F1 score: ', np.round(f1_test_folds, decimals=4), '(per fold)')
        test_f1_per_subtype = [np.round(f1_scores(gt_test, np.concatenate(pred_folds), pos_index=0), decimals=4),
                              np.round(f1_scores(gt_test, np.concatenate(pred_folds), pos_index=1), decimals=4)]
        print('Test F1 score: ', test_f1_per_subtype, '(per subtype, [G3, G4])')

        cm_test = confusion_matrix(gt_test, np.concatenate(pred_folds))
        print('Test Confusion Matrix: ')
        print(cm_test)
        #disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_val)
        #disp1.plot()
        print('Threshold:', np.round(thresholds, decimals=4).tolist())

        print('-----------Validation--------------')
        all_val_folds = np.stack(val_folds)
        mean_folds = np.mean(all_val_folds, axis=0)
        print('All_val_folds: ', all_val_folds.shape)
        print('All_mean_val_folds:', mean_folds.shape)

        lg_auc_val = roc_auc_score(y_val, mean_folds)
        print('Validation AUC : ', np.round(lg_auc_val, decimals=4))
        # Binarize the true labels to use with roc_auc_score

        f1_mean_val = f1_score(y_val, (mean_folds > mean_thr).astype('int'), average='weighted')
        print('Validation f1 score: ', np.round(f1_mean_val, decimals=4))
        val_f1_per_subtype = [np.round(f1_scores(y_val, (mean_folds > mean_thr).astype('int'), pos_index=0), decimals=4),
                             np.round(f1_scores(y_val, (mean_folds > mean_thr).astype('int'), pos_index=1), decimals=4)]
        print('Validation f1 score: ', val_f1_per_subtype, '(per subtype, [G3, G4])')

        cm_val = confusion_matrix(y_val, (mean_folds > mean_thr).astype('int'))
        print('Validation Confusion Matrix: ')
        print(cm_val)
        #disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_val)
        #disp1.plot()
        print('Threshold:', np.round(mean_thr, decimals=4))
        id_val = np.array(id_val)
        print('False Positive: (True: G3, Pred: G4): ')
        print(id_val[np.where((y_val == 0) & ((mean_folds > mean_thr).astype('int') == 1))[0]].tolist())
        print('False Negative: (True: G4, Pred: G3): ')
        print(id_val[np.where((y_val == 1) & ((mean_folds > mean_thr).astype('int') == 0))[0] ].tolist())
        
        print('-----------Golden--------------')
        all_new_folds = np.stack(new_folds)
        mean_new_folds = np.mean(all_new_folds, axis=0)
        print('All_new_folds: ', all_new_folds.shape)
        print('All_mean_new_folds:', mean_new_folds.shape)

        lg_auc_new = roc_auc_score(y_new, mean_new_folds)
        print('Golden AUC : ', np.round(lg_auc_new, decimals=4))
        # Binarize the true labels to use with roc_auc_score

        f1_mean_new = f1_score(y_new, (mean_new_folds > mean_thr).astype('int'), average='weighted')
        print('Golden f1 score: ', np.round(f1_mean_new, decimals=4))
        new_f1_per_subtype = [np.round(f1_scores(y_new, (mean_new_folds > mean_thr).astype('int'), pos_index=0), decimals=4),
                             np.round(f1_scores(y_new, (mean_new_folds > mean_thr).astype('int'), pos_index=1), decimals=4)]
        print('Golden f1 score: ', new_f1_per_subtype, '(per subtype, [G3, G4])')

        cm_new = confusion_matrix(y_new, (mean_new_folds > mean_thr).astype('int'))
        print('Golden Confusion Matrix: ')
        print(cm_new)
        print('Threshold:', np.round(mean_thr, decimals=4))
        id_new = np.array(id_new)
        print('False Positive: (True: G3, Pred: G4): ')
        print(id_new[np.where((y_new == 0) & ((mean_new_folds > mean_thr).astype('int') == 1))[0]].tolist())
        print('False Negative: (True: G4, Pred: G3): ')
        print(id_new[np.where((y_new == 1) & ((mean_new_folds > mean_thr).astype('int') == 0))[0]].tolist())

        print('-------------Params-----------------')
        for i in range(3):
            print(params_best[i])
        
        experiment_data = {
            'Experiment': experiment_name,
            'Test AUC Overall': np.round(lg_auc_test, decimals=4),
            'Test AUC Mean Fold': np.round(np.mean(auc_test_folds), decimals=4),
            'Test AUC Per Fold': np.round(auc_test_folds, decimals=4).tolist(),
            'Test F1 Overall': np.round(f1_test, decimals=4),
            'Test F1 Per Fold': np.round(f1_test_folds, decimals=4).tolist(),
            'Test F1 Per Subtype': test_f1_per_subtype,
            'Thresholds': np.round(thresholds, decimals=4).tolist(),
            'Mean Threshold': np.round(mean_thr, decimals=4),
            'Validation AUC Overall': np.round(lg_auc_val, decimals=4),
            'Validation F1 Overall': np.round(f1_mean_val, decimals=4),
            'Validation F1 Per Subtype': val_f1_per_subtype,
            'Golden AUC Overall': np.round(lg_auc_new, decimals=4),
            'Golden F1 Overall': np.round(f1_mean_new, decimals=4),
            'Golden F1 Per Subtype': new_f1_per_subtype,
            'details file': experiment_name + '_' + timestamp + '.txt'
        }

        # 写入数据
        write_experiment_results_to_csv(args.csv_result, experiment_data)

    
    