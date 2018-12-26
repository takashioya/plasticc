import pandas as pd
import numpy as np
import pickle
import re
import feather
import gc
import os
import sys
sys.path.append('..')
from scripts.utils import *
from catboost import CatBoostClassifier, FeaturesData
from tqdm import tqdm


def run(exp_name, data_type):
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    le = load_pickle('../others/label_encoder.pkl')
    y = le.transform(np.load('../others/train_target.npy'))
    distmod_mask = np.load('../others/distmod_mask.npy')
    ex_gal_labels = np.where(np.bincount(y[distmod_mask]) != 0)[0]
    gal_labels = np.where(np.bincount(y[~distmod_mask]) != 0)[0]
    ex_gal_index = ((tes_m['hostgal_specz'].isnull()) & (~tes_m['distmod'].isnull())).values
    ex_gal_spec_index = ((~tes_m['hostgal_specz'].isnull()) & (~tes_m['distmod'].isnull())).values
    gal_index = (tes_m['distmod'].isnull()).values

    fn_s = np.load('../fi/' + exp_name + '_fn_s_' + data_type + '.npy')
    fn_s = [el.replace('/', '_') for el in fn_s]
    X_test = load_arr(fn_s, 'test')

    model = CatBoostClassifier()
    model.load_model('../models/' + exp_name + '_' + data_type + '.cbm')

    if data_type == 'ex_gal':
        real_test_data = FeaturesData(X_test.astype(np.float32)[ex_gal_index])
        y_pred_ex_gal = model.predict_proba(real_test_data)
        ex_gal_pred = np.zeros((y_pred_ex_gal.shape[0], 14))
        ex_gal_pred[:, ex_gal_labels] = y_pred_ex_gal
        np.save('../preds/' + data_type + '_pred_' + exp_name + '.npy', ex_gal_pred)
    elif data_type == 'ex_gal_spec':
        real_test_data = FeaturesData(X_test.astype(np.float32)[ex_gal_spec_index])
        y_pred_ex_gal_spec = model.predict_proba(real_test_data)
        ex_gal_spec_pred = np.zeros((y_pred_ex_gal_spec.shape[0], 14))
        ex_gal_spec_pred[:, ex_gal_labels] = y_pred_ex_gal_spec
        np.save('../preds/' + data_type + '_pred_' + exp_name + '.npy', ex_gal_spec_pred)
    elif data_type == 'gal':
        real_test_data = FeaturesData(X_test.astype(np.float32)[gal_index])
        y_pred_gal = model.predict_proba(real_test_data)
        gal_pred = np.zeros((y_pred_gal.shape[0], 14))
        gal_pred[:, gal_labels] = y_pred_gal
        np.save('../preds/' + data_type + '_pred_' + exp_name + '.npy', gal_pred)
    else: 
        raise Error
    gc.collect()


def main():
    exp_names = ['exp_46_2', 'exp_44_2', 'exp_40_4', 'exp_46_1', 'exp_45_3', 'exp_46_4', 'exp_43_3']
    data_types = ['ex_gal', 'ex_gal_spec', 'gal']
    
    for exp_name in tqdm(exp_names):
        for data_type in tqdm(data_types):
            run(exp_name, data_type)
            
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()
