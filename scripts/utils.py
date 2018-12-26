import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def save_pickle(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, mode='rb') as f:
        obj = pickle.load(f)
    return obj

def write_to_fn_tree(column_names, f_type_1, f_type_2, path='../others/fn_tree.pkl'):
    fn_tree = load_pickle(path)
    if f_type_2 not in list(fn_tree[f_type_1].keys()):
        fn_tree[f_type_1][f_type_2] = column_names
    else:
        for column_name in column_names:
            if column_name not in fn_tree[f_type_1][f_type_2]:
                fn_tree[f_type_1][f_type_2] += [column_name]
    save_pickle(fn_tree, path)

def initialize_feature_name_tree(path, destruct=False):
    if not os.path.exists(path) or destruct:
        fn_tree = {}
        f_types_1 = ['meta', 'ts', 'nyanp']
        f_types_2 = [['main'], ['main'], ['main']]
        for i, f_type_1 in enumerate(f_types_1):
            fn_tree[f_type_1] = {}
            for f_type_2 in f_types_2[i]:
                fn_tree[f_type_1][f_type_2] = []
        save_pickle(fn_tree, path)
             
def save_df_as_npy(df, f_type_1, f_type_2, data_type, path='../features/'):
    if data_type == 'train':
        assert (df.shape[0] == 7848)
    elif data_type == 'test':
        assert(df.shape[0] == 3492890)
    else:
        raise Error
    column_names = list(df.columns)
    for column_name in tqdm(column_names):
        np.save(os.path.join(path, column_name) + '_' + data_type + '.npy', df[column_name].values)
    write_to_fn_tree(column_names, f_type_1, f_type_2)
            
def load_arr(feature_names, data_type, path='../features/'):
    if data_type == 'train':
        X = np.zeros((7848, len(feature_names)), dtype = np.float32)
    elif data_type == 'test':
        X = np.zeros((3492890, len(feature_names)), dtype = np.float32)
    else:
        raise Error

    for i in tqdm(range(len(feature_names))):
        X[:, i] = np.load(os.path.join(path, feature_names[i]) + '_' + data_type + '.npy')
    return X

def load_dataframe_npy(feature_names, data_type, path='../features/'):
    df = pd.DataFrame()
    for feature_name in feature_names:
        feature = np.load(os.path.join(path, feature_name) + '_' + data_type + '.npy')
        df[feature_name] = feature
    return df

def encode(train, column_name):
    encoded = pd.merge(train[[column_name]], pd.DataFrame(train[column_name].unique(), columns=[column_name])\
                       .reset_index().dropna(), how='left', on=column_name)[['index']].rename(
        columns={'index': column_name})
    return encoded

def calc_sample_weight(real_weight, y):
    sample_weight = np.zeros(y.shape[0])
    for i in range(14):
        sample_weight[y == i] = real_weight[i]
    return sample_weight

def save_df_as_npy_without_fn_tree(df, data_type, path='../features/'):
    if data_type == 'train':
        assert (df.shape[0] == 7848)
    elif data_type == 'test':
        assert(df.shape[0] == 3492890)
    else:
        raise Error
    column_names = list(df.columns)
    for column_name in tqdm(column_names):
        np.save(os.path.join(path, column_name) + '_' + data_type + '.npy', df[column_name].values)
        
def get_skew_kurt_from_df_ignored(df, col_name_agg, by_hoge):
    gp = df.groupby(by_hoge)[col_name_agg].agg(['mean', 'std']).reset_index().rename(columns = {'mean' : 'mu', 'std' : 's'})
    merged = pd.merge(df, gp, how = 'left', on = by_hoge)
    merged['z'] = (merged[col_name_agg] - merged['mu'])/merged['s']
    merged['z_pow_3'] = merged['z'] ** 3
    merged['z_pow_4'] = merged['z'] ** 4
    gp_z_pow_sum = merged.groupby(by_hoge)[['z_pow_3', 'z_pow_4']].sum()
    gp = df.groupby(by_hoge).size().reset_index().rename(columns = {0 : 'n'})
    assert(gp.shape[0] == gp_z_pow_sum.shape[0])
    gp['z_pow_3_sum'] = gp_z_pow_sum['z_pow_3'].values
    gp['z_pow_4_sum'] = gp_z_pow_sum['z_pow_4'].values
    gp['skew'] = (gp['n']/((gp['n'] - 1) * (gp['n'] - 2))) * gp['z_pow_3_sum']
    gp['kurt'] = ((gp['n'] * (gp['n'] + 1))/((gp['n'] - 1) * (gp['n'] - 2) * (gp['n'] - 3))) * gp['z_pow_4_sum'] \
    - (3 * ((gp['n'] - 1) ** 2))/((gp['n'] - 2) * (gp['n'] - 3))
    return gp[['skew', 'kurt', 'object_id']].set_index(by_hoge)

def get_skew_kurt_from_df(df, col_name_agg, by_hoge):
    gp = df.groupby(by_hoge)[col_name_agg].agg(['mean', 'std']).reset_index().rename(columns = {'mean' : 'mu', 'std' : 's'})
    merged = pd.merge(df, gp, how = 'left', on = by_hoge)
    merged['z'] = (merged[col_name_agg] - merged['mu'])/merged['s']
    merged['z_pow_3'] = merged['z'] ** 3
    merged['z_pow_4'] = merged['z'] ** 4
    gp_z_pow_sum = merged.groupby(by_hoge)[['z_pow_3', 'z_pow_4']].sum()
    gp = df.groupby(by_hoge).size().reset_index().rename(columns = {0 : 'n'})
    assert(gp.shape[0] == gp_z_pow_sum.shape[0])
    gp['z_pow_3_sum'] = gp_z_pow_sum['z_pow_3'].values
    gp['z_pow_4_sum'] = gp_z_pow_sum['z_pow_4'].values
    gp['skew'] = (gp['n']/((gp['n'] - 1) * (gp['n'] - 2))) * gp['z_pow_3_sum']
    gp['kurt'] = ((gp['n'] * (gp['n'] + 1))/((gp['n'] - 1) * (gp['n'] - 2) * (gp['n'] - 3))) * gp['z_pow_4_sum'] \
    - (3 * ((gp['n'] - 1) ** 2))/((gp['n'] - 2) * (gp['n'] - 3))
    gp_unstack = gp.set_index(['object_id', 'passband']).drop(['n', 'z_pow_3_sum', 'z_pow_4_sum'], axis = 1).unstack(level = -1)
    return gp_unstack

def get_raw_from_yuval_sub(filenames, folds, cols):
    subs = [pd.read_csv('../sub/' + el).sort_values('object_id').reset_index(drop = True) for el in list(np.array(filenames)[folds])]
    avg = sum([el[cols] for el in subs])/len(folds)
    assert(np.all(np.isclose(avg.sum(axis = 1), 1)))
    return avg

def get_real_weight(y, W_tr):
    bin_count = np.bincount(y)
    inverse_sum = (1/bin_count).sum()
    class_weight = (1/bin_count)/inverse_sum
    real_weight = class_weight * W_tr
    real_weight = real_weight/real_weight.sum()
    return real_weight

def load_arr_with_idx(feature_names, data_type, idx, path = '../features/'):
    X = np.zeros((idx.sum(), len(feature_names)), dtype = np.float32)
    for i in tqdm(range(len(feature_names))):
        X[:, i] = np.load(os.path.join(path, feature_names[i]) + '_' + data_type + '.npy')[idx]
    return X

def get_raw_from_mamas_sub(sub):
    sub_dropped = sub.drop(['object_id', 'class_99'], axis = 1)
    raw = sub_dropped.divide(sub_dropped.sum(axis = 1), axis = 0)
    assert(np.all(np.isclose(raw.sum(axis = 1), 1)))
    return raw

def get_raw_from_nyanp_sub(sub):
    sub_sorted = sub.sort_values(['object_id']).drop(['object_id', 'class_99'], axis = 1).reset_index(drop = True)
    assert(np.all(np.isclose(sub_sorted.sum(axis = 1), 1)))
    return sub_sorted