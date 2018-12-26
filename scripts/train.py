from catboost import CatBoostClassifier, FeaturesData
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')
import re
from scripts.utils import * 
from tqdm import tqdm
import pickle
import feather

def get_pseudo_idx():
    threshold = 0.91
    obj_class = 90
    sub = pd.read_csv('../data/sample_submission.csv.zip')
    cols = sub.columns[1:-1]
    sub_86_s = ['pred16r4qq_0.csv', 'pred16r4qq_1.csv', 'pred16r4qq_2.csv', 'pred16r4qq_3.csv']
    best_sub_dp = get_raw_from_yuval_sub(sub_86_s, np.array([0, 1, 2, 3]), cols)
    pseudo_idx = best_sub_dp['class_' + str(obj_class)] > threshold
    return pseudo_idx

def load_data_ex_gal(exp_name, data_type, pseudo_idx):
    fn_s = np.load('../fi/' + exp_name + '_fn_s_' + data_type + '.npy')
    fn_s = [el.replace('/', '_') for el in fn_s]
    X = load_arr(fn_s, 'train')
    X_pseudo = load_arr_with_idx(fn_s, 'test', pseudo_idx)
    return X, X_pseudo

def run_ex_gal(exp_name, data_type):
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    le = load_pickle('../others/label_encoder.pkl')
    y = le.transform(np.load('../others/train_target.npy'))
    distmod_mask = np.load('../others/distmod_mask.npy')
    W = np.load('../others/W.npy')
    pseudo_idx = np.load('../others/pseudo_idx.npy')
    class_names = [99,95,92,90,88,67,65,64,62,53,52,42,16,15,6]
    obj_class = 90

    W_tr = np.zeros(14)
    W_tr[le.transform(class_names[1:])] = W[1:]
    real_weight = get_real_weight(y, W_tr)
    ex_gal_labels = np.where(np.bincount(y[distmod_mask]) != 0)[0]
    ex_gal_label_map = np.zeros(np.max(ex_gal_labels) + 1, dtype = np.int32)
    ex_gal_label_map[ex_gal_labels] = np.arange(ex_gal_labels.shape[0])
    ex_gal_index = ((tes_m['hostgal_specz'].isnull()) & (~tes_m['distmod'].isnull())).values
    
    X, X_pseudo = load_data_ex_gal(exp_name, data_type, pseudo_idx)
    y_pseudo = np.full(pseudo_idx.sum(), ex_gal_label_map[le.transform([obj_class])][0])
    params = {
    'iterations': 10000,
    'learning_rate' :0.1, 
    'depth' : 3,
    'loss_function' : 'MultiClass', 
    'colsample_bylevel' : 0.7,
    'random_seed' : 0,
    'class_weights' : real_weight[ex_gal_labels]/real_weight[ex_gal_labels].sum()
    }
    iterations = load_pickle('../fi/' + exp_name + '_rounds.pkl')
    iteration = iterations[data_type]
    params['iterations'] = iteration
    print('iteration: ' + str(params['iterations']))
    
    orig_size = np.bincount(ex_gal_label_map[y[distmod_mask]])[ex_gal_label_map[le.transform([obj_class])][0]]
    whole_data = np.concatenate((X[distmod_mask], X_pseudo), axis = 0)
    whole_labels = np.concatenate((ex_gal_label_map[y[distmod_mask]], y_pseudo), axis = 0)
    after_size = np.bincount(whole_labels)[ex_gal_label_map[le.transform([obj_class])][0]]
    sample_weight = np.ones(whole_labels.shape[0])
    sample_weight[whole_labels == ex_gal_label_map[le.transform([obj_class])][0]] = orig_size/after_size
    model = CatBoostClassifier(**params)
    model.fit(whole_data, whole_labels, sample_weight = sample_weight)
    model.save_model('../models/' + exp_name + '_' + data_type + '.cbm')

def run_gal(exp_name, data_type):
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    le = load_pickle('../others/label_encoder.pkl')
    y = le.transform(np.load('../others/train_target.npy'))
    distmod_mask = np.load('../others/distmod_mask.npy')
    W = np.load('../others/W.npy')
    pseudo_idx = np.load('../others/pseudo_idx.npy')
    class_names = [99,95,92,90,88,67,65,64,62,53,52,42,16,15,6]
    obj_class = 90

    W_tr = np.zeros(14)
    W_tr[le.transform(class_names[1:])] = W[1:]
    real_weight = get_real_weight(y, W_tr)
    gal_labels = np.where(np.bincount(y[~distmod_mask]) != 0)[0]
    gal_label_map = np.zeros(np.max(gal_labels) + 1, dtype = np.int32)
    gal_label_map[gal_labels] = np.arange(gal_labels.shape[0])

    fn_s = np.load('../fi/' + exp_name + '_fn_s_' + data_type + '.npy')
    fn_s = [el.replace('/', '_') for el in fn_s]
    X = load_arr(fn_s, 'train')
    params = {
    'iterations': 10000,
    'learning_rate' :0.1, 
    'depth' : 3,
    'loss_function' : 'MultiClass', 
    'colsample_bylevel' : 0.7,
    'random_seed' : 0,
    'class_weights' : real_weight[gal_labels]
    }
    iterations = load_pickle('../fi/' + exp_name + '_rounds.pkl')
    iteration = iterations[data_type]
    params['iterations'] = iteration
    print('iteration: ' + str(params['iterations']))

    whole_data = X[~distmod_mask]
    whole_labels = gal_label_map[y[~distmod_mask]]
    model = CatBoostClassifier(**params)
    model.fit(whole_data, whole_labels)
    model.save_model('../models/' + exp_name + '_' + data_type + '.cbm')

def run(exp_name, data_type):
    if data_type == 'gal':
        run_gal(exp_name, data_type)
    elif data_type == 'ex_gal' or data_type == 'ex_gal_spec':
        run_ex_gal(exp_name, data_type)
    else:
        raise Error
        
def main():
    pseudo_idx = get_pseudo_idx()
    np.save('../others/pseudo_idx.npy', pseudo_idx)
    
    exp_names = ['exp_46_2', 'exp_44_2', 'exp_40_4', 'exp_46_1', 'exp_45_3', 'exp_46_4', 'exp_43_3']
    data_types = ['ex_gal', 'ex_gal_spec', 'gal']
    
    for exp_name in tqdm(exp_names):
        for data_type in data_types:
            run(exp_name, data_type)
            
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()