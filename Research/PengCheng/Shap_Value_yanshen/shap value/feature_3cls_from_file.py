"""
本代码是为了从之前已经训练过的模型保存的txt文件中读取特征名称和模型参数而创建的
"""
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


def extract_features_and_params(file_path):
    """
    提取特征列表和参数字典列表。
    
    :param file_path: txt文件的路径
    :return: features_from_file - 特征列表
             params_from_file - 参数字典列表
    """
    import ast

    with open(file_path, 'r') as file:
        content = file.readlines()

    # 提取特征列表
    features_from_file = ast.literal_eval(content[1].strip())

    # 提取参数字典
    params_content = []
    params_flag = False
    for line in content:
        if '--Params--' in line:
            params_flag = True
            continue
        if params_flag:
            if line.strip():  # 忽略空行
                params_content.append(line.strip())

    # 解析参数字典
    params_from_file = []
    for line in params_content:
        try:
            param_dict = ast.literal_eval(line)
            params_from_file.append(param_dict)
        except (ValueError, SyntaxError):
            continue

    return features_from_file, params_from_file


class DummyFile(object):
    def write(self, x): pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LGBM_clnc')

    # 设置一些参数变量，这些变量可以方便地从外部输入
    parser.add_argument('--radiomics_file', # 大数据集的radiomics特征路径：cohort1, cohort2, cohort3
                        default='/Users/huyanshen/Desktop/AI-Medicine/Research/Shap_Value_yanshen/new_3cls_radiomics_new.txt')
    parser.add_argument('--radiomics_file_c4', # 黄金验证数据集的radiomics特征路径：cohort4
                        default='/Users/huyanshen/Desktop/AI-Medicine/Research/Shap_Value_yanshen/new_3cls_radiomics_cohort4_new.txt')
    parser.add_argument('--result_file',  required=True) # 读取特征名称和LGBM分类器参数的结果文件的路径
    parser.add_argument('--result_file_save',  required=True) # 保存metrics以及其他的一些参数的路径
    parser.add_argument('--csv_result', # 保存关键指标的csv文件路径
                        default='/Users/huyanshen/Desktop/AI-Medicine/Research/Shap_Value_yanshen/results.csv')
    parser.add_argument('--print', default=True) # 运行时是否打印详细信息
    

    args = parser.parse_args()
    
    original_stdout = sys.stdout 
    if not args.print:
        sys.stdout = DummyFile()
    
    # 读取特征文件为dataframe
    three_cls_path = args.radiomics_file
    three_cls_path_c4 = args.radiomics_file_c4
    data = pd.read_csv(three_cls_path, sep='\t')
    data_c4 = pd.read_csv(three_cls_path_c4, sep='\t')
    
    # 从txt文件中读取特征名称和模型参数
    file_path = args.result_file
    sel_columns_from_file, params_from_file = extract_features_and_params(file_path)
    
    sel_columns = sel_columns_from_file

    # 参数预处理，我们使用cohort1作为训练集进行三折训练，cohort2,3,4都是作为外部测试验证集
    x = data[sel_columns]
    x.iloc[:, 4:] = x.iloc[:, 4:].astype(float)
    print(sel_columns, x.shape)
    
    x_new = data_c4[sel_columns]
    id_new = data_c4['id'].tolist()
    print(sel_columns, x_new.shape)
    x_new.iloc[:, 3:] = x_new.iloc[:, 3:].astype(float)
    print('total features and labels shape: ', x.shape)
    
    # split base dataset and external validation dataset
    df_val = x[(x['Cohort'] == 2) | (x['Cohort'] == 3)]
    df_val.reset_index(inplace=True)
    X_val = df_val.iloc[:, 4:]

    feature_names = X_val.columns.tolist()
    print(feature_names)
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

    # 训练集
    df_X = x[x['Cohort'] == 1]
    df_X.reset_index(inplace=True)

    folds_indices = {}

    # 从文件中读取已经划分好的fold情况
    for fold in df_X['fold'].unique():
        test_index = df_X[df_X['fold'] == fold].index
        train_index = df_X[df_X['fold'] != fold].index
        print(len(test_index), len(train_index))
        folds_indices[fold] = {'train_index': train_index, 'test_index': test_index}

    X = df_X.iloc[:, 4:]
    y = df_X['Subtypes_three']
    print('X: ', X.shape)
    print('Fold index has been determined!')
    
    # params from file
    params_best = params_from_file
        
    test_folds = []
    test_gt_folds = []
    val_folds = []
    true_folds = []
    pred_folds = []
    cm_folds = []
    f1_test_folds = []
    auc_test_folds = []
    new_folds = []

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

        # 这里需要注意的是采用了数据增广，所以最后用于拟合数据的训练集并不只是之前划分的X_train和y_train
        nm = SMOTE(random_state=42)
        X_res, y_res = nm.fit_resample(X_train, y_train)

        # 模型的参数以及拟合
        params_cur = params_best[i]
        modellgb = LGBMClassifier()
        modellgb.set_params(**params_cur)
        clf_lg = modellgb.fit(X_res, y_res)
        
        # 模型预测出的概率
        p_lg_train = clf_lg.predict_proba(X_res)  # [:,1]
        p_lg_test = clf_lg.predict_proba(X_test)  # [:,1]
        p_lg_val = clf_lg.predict_proba(X_val)  # [:,1]
        p_lg_new = clf_lg.predict_proba(X_new)  # [:,1]

        val_folds.append(p_lg_val)
        test_folds.append(p_lg_test)
        test_gt_folds.append(y_test)
        new_folds.append(p_lg_new)

        # 模型预测出的类别
        l_lg_train = clf_lg.predict(X_res)
        l_lg_test = clf_lg.predict(X_test)
        l_lg_val = clf_lg.predict(X_val)
        l_lg_new = clf_lg.predict(X_new)

        f1_train = f1_score(y_res, l_lg_train, average='weighted')
        f1_test = f1_score(y_test, l_lg_test, average='weighted')
        f1_val = f1_score(y_val, l_lg_val, average='weighted')
        f1_new = f1_score(y_new, l_lg_new, average='weighted')
        f1_test_folds.append(f1_test)

        true_folds.append(y_test)
        pred_folds.append(l_lg_test)

        lg_auc_train = roc_auc_score(y_res, p_lg_train, multi_class='ovr')
        lg_auc_test = roc_auc_score(y_test, p_lg_test, multi_class='ovr')
        lg_auc_val = roc_auc_score(y_val, p_lg_val, multi_class='ovr')
        lg_auc_new = roc_auc_score(y_new, p_lg_new, multi_class='ovr')
        auc_test_folds.append(lg_auc_test)

        cm_test = confusion_matrix(y_test, l_lg_test)
        cm_folds.append(cm_test)

    all_val_folds = np.stack(val_folds)
    all_new_folds = np.stack(new_folds)
    
    mean_folds = np.mean(all_val_folds, axis=0)
    mean_new_folds = np.mean(all_new_folds, axis=0)
    
    # 打印关键参数并存入路径
    with open(args.result_file_save, 'w') as f:
        sys.stdout = f
        print('-------------Features---------------')
        print(sel_columns)

        print('-------------Testing----------------')
        all_test_folds = np.vstack(test_folds)
        gt_test = np.concatenate(test_gt_folds)
        print('All_test_folds: ', all_test_folds.shape)

        lg_auc_test = roc_auc_score(gt_test, all_test_folds, multi_class='ovr')
        print('Test AUC : ', np.round(lg_auc_test, decimals=4))
        print('Test AUC: ', np.round(auc_test_folds, decimals=4), '(per fold)')
        # Binarize the true labels to use with roc_auc_score
        y_test_binarized = label_binarize(gt_test, classes=[0, 1, 2])
        aucs_test = []
        for i in range(3):
            auc_test = roc_auc_score(y_test_binarized[:, i], np.array(all_test_folds)[:, i])
            aucs_test.append(auc_test)
        print('Test AUC : ', np.round(aucs_test, decimals=4), '(per subtype, [WNT, G3/G4, SHH])')

        f1_test = f1_score(gt_test, np.argmax(all_test_folds, axis=1), average='weighted')
        print('Test F1 score: ', np.round(f1_test, decimals=4))
        print('Test F1 score: ', np.round(f1_test_folds, decimals=4), '(per fold)')
        test_f1_per_subtype = [np.round(f1_scores(gt_test, np.argmax(all_test_folds, axis=1), pos_index=0), decimals=4),
                              np.round(f1_scores(gt_test, np.argmax(all_test_folds, axis=1), pos_index=1), decimals=4),
                              np.round(f1_scores(gt_test, np.argmax(all_test_folds, axis=1), pos_index=2), decimals=4)]
        print('Test F1 score: ', test_f1_per_subtype, '(per subtype, [WNT, G3/G4, SHH])')

        cm_test = confusion_matrix(gt_test, np.argmax(all_test_folds, axis=1))
        print('Test Confusion Matrix: ')
        print(cm_test)
        #disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_val)
        #disp1.plot()

        print('-----------Validation--------------')
        all_val_folds = np.stack(val_folds)
        mean_folds = np.mean(all_val_folds, axis=0)
        print('All_val_folds: ', all_val_folds.shape)
        print('All_mean_val_folds:', mean_folds.shape)

        lg_auc_val = roc_auc_score(y_val, mean_folds, multi_class='ovr')
        print('Validation AUC : ', np.round(lg_auc_val, decimals=4))
        # Binarize the true labels to use with roc_auc_score
        y_val_binarized = label_binarize(y_val, classes=[0, 1, 2])
        aucs_val = []
        for i in range(3):
            auc_val = roc_auc_score(y_val_binarized[:, i], np.array(mean_folds)[:, i])
            aucs_val.append(auc_val)
        print('Validation AUC : ', np.round(aucs_val, decimals=4), '(per subtype, [WNT, G3/G4, SHH])')

        f1_mean_val = f1_score(y_val, np.argmax(mean_folds, axis=1), average='weighted')
        print('Validation f1 score: ', np.round(f1_mean_val, decimals=4))
        val_f1_per_subtype = [np.round(f1_scores(y_val, np.argmax(mean_folds, axis=1), pos_index=0), decimals=4),
                             np.round(f1_scores(y_val, np.argmax(mean_folds, axis=1), pos_index=1), decimals=4),
                             np.round(f1_scores(y_val, np.argmax(mean_folds, axis=1), pos_index=2), decimals=4)]
        print('Validation f1 score: ', val_f1_per_subtype, '(per subtype, [WNT, G3/G4, SHH])')

        cm_val = confusion_matrix(y_val, np.argmax(mean_folds, axis=1))
        print('Validation Confusion Matrix: ')
        print(cm_val)
        #disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_val)
        #disp1.plot()
        
        print('-----------Golden--------------')
        all_new_folds = np.stack(new_folds)
        mean_new_folds = np.mean(all_new_folds, axis=0)
        print('All_new_folds: ', all_new_folds.shape)
        print('All_mean_new_folds:', mean_new_folds.shape)

        lg_auc_new = roc_auc_score(y_new, mean_new_folds, multi_class='ovr')
        print('Golden AUC : ', np.round(lg_auc_new, decimals=4))
        # Binarize the true labels to use with roc_auc_score
        y_new_binarized = label_binarize(y_new, classes=[0, 1, 2])
        aucs_new = []
        for i in range(3):
            auc_new = roc_auc_score(y_new_binarized[:, i], np.array(mean_new_folds)[:, i])
            aucs_new.append(auc_new)
        print('Golden AUC : ', np.round(aucs_new, decimals=4), '(per subtype, [WNT, G3/G4, SHH])')

        f1_mean_new = f1_score(y_new, np.argmax(mean_new_folds, axis=1), average='weighted')
        print('Golden f1 score: ', np.round(f1_mean_new, decimals=4))
        new_f1_per_subtype = [np.round(f1_scores(y_new, np.argmax(mean_new_folds, axis=1), pos_index=0), decimals=4),
                             np.round(f1_scores(y_new, np.argmax(mean_new_folds, axis=1), pos_index=1), decimals=4),
                             np.round(f1_scores(y_new, np.argmax(mean_new_folds, axis=1), pos_index=2), decimals=4)]
        print('Golden f1 score: ', new_f1_per_subtype, '(per subtype, [WNT, G3/G4, SHH])')

        cm_new = confusion_matrix(y_new, np.argmax(mean_new_folds, axis=1))
        print('Golden Confusion Matrix: ')
        print(cm_new)
        print('---------------Golden Probability------------------')
        print('id', '\t', 'True', '\t', 'Pred', '\t', 'WNT', '\t', 'G34', '\t', 'SHH')
        for i in range(len(y_new)):
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(id_new[i], y_new[i], np.argmax(mean_new_folds, axis=1)[i], mean_new_folds[i][0], mean_new_folds[i][1],mean_new_folds[i][2]))
        print('------------True Golden WNT----------------')
        for i in range(len(y_new)):
            if y_new[i] ==0 and np.argmax(mean_new_folds, axis=1)[i] == 0:
                print('{}\t{}\t{}'.format(id_new[i], y_new[i], np.argmax(mean_new_folds, axis=1)[i]))
        

        print('-------------Params-----------------')
        for i in range(3):
            print(params_best[i])
            
        experiment_data = {
            'Experiment': args.result_file_save.split('/')[-1],
            'Test AUC Overall': np.round(lg_auc_test, decimals=4),
            'Test AUC Per Fold': np.round(auc_test_folds, decimals=4).tolist(),
            'Test AUC Per Subtype': np.round(aucs_test, decimals=4),
            'Test F1 Overall': np.round(f1_test, decimals=4),
            'Test F1 Per Fold': np.round(f1_test_folds, decimals=4).tolist(),
            'Test F1 Per Subtype': test_f1_per_subtype,
            'Validation AUC Overall': np.round(lg_auc_val, decimals=4),
            'Validation AUC Per Subtype': np.round(aucs_val, decimals=4).tolist(),
            'Validation F1 Overall': np.round(f1_mean_val, decimals=4),
            'Validation F1 Per Subtype': val_f1_per_subtype,
            'Golden AUC Overall': np.round(lg_auc_new, decimals=4),
            'Golden AUC Per Subtype': np.round(aucs_new, decimals=4).tolist(),
            'Golden F1 Overall': np.round(f1_mean_new, decimals=4),
            'Golden F1 Per Subtype': new_f1_per_subtype,
            'details file': args.result_file_save.split('/')[-1]
        }

        # 写入数据
        write_experiment_results_to_csv(args.csv_result, experiment_data)

    
    