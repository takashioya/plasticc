import sys
sys.path.append('..')
from scripts.utils import * 
import re
from itertools import chain
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_and_save_spectrum_features(data_type):
    fn_tree = load_pickle('../others/fn_tree.pkl')
    fn_s = []
    fn_s += list(chain.from_iterable(list(fn_tree['meta'].values())))
    fn_s += list(chain.from_iterable(list(fn_tree['ts'].values())))
    
    fn_passband = [el for el in fn_s if 'passband' in el and 'normed' not in el]
    assert(len(fn_passband) % 6 == 0)
    fn_replaced = [re.sub('_passband_' + '\d', '', el) for el in fn_passband]
    unique_label = encode(pd.DataFrame(fn_replaced), 0).values.ravel()
    unique_indices = np.array([np.where(unique_label == i)[0] for i in range(int(unique_label.max() + 1))])
    assert(unique_indices.shape[1] == 6)
    p_groups = np.array(fn_passband)[unique_indices]
    counter = 0
    for p_group in tqdm(p_groups):
        assert(np.unique([re.sub('_passband_' + '\d', '', el) for el in p_group]).shape[0] == 1)
        wth_p_name = [re.sub('_passband_' + '\d', '', el) for el in p_group][0]
        gp_names = [gp_name for gp_name in list(fn_tree['ts'].keys()) if np.all(np.in1d(p_group, fn_tree['ts'][gp_name]))]
        assert(len(gp_names) == 1)
        gp_name = gp_names[0]
        
        stats = ['mean', 'sum', 'median', 'min', 'max', 'var']
        funcs = [np.mean, np.sum, np.nanmedian, np.min, np.max, np.var]
    
        stats_col_names = [wth_p_name + '_over_p_' + el for el in stats]
        normed_col_names = [el + '_normed' for el in p_group]
        if not np.all([os.path.exists('../features/' + el + '_' + data_type + '.npy') for el in stats_col_names + normed_col_names]):
            counter += 1
            p_df = load_dataframe_npy(p_group, data_type = data_type)

            np.warnings.filterwarnings('ignore')
            stats_df = pd.concat([pd.Series(func(p_df, axis = 1)) for func in funcs], axis = 1).astype(np.float32)
            stats_df.columns = stats_col_names

            normed_df = p_df.div(stats_df[wth_p_name + '_over_p_sum'], axis = 0).astype(np.float32)
            normed_df.columns = normed_col_names
            save_df_as_npy(normed_df, 'ts', gp_name + '_normed', data_type, path='../features/')
            save_df_as_npy(stats_df, 'ts', gp_name + '_pstats', data_type, path='../features/')
    print(str(counter) + ' type of objects are saved')


def get_and_save_spectrum_features_nyanp(data_type):
    fn_tree = load_pickle('../others/fn_tree.pkl')
    fn_s = []
    fn_s += list(chain.from_iterable(list(fn_tree['nyanp'].values())))

    fn_passband = [el for el in fn_s if 'ch' in el and 'normed' not in el and el.count('ch') == 1 \
                   and 'deltadiff' not in el and 'change' not in el and 'chisq' not in el]
    assert(len(fn_passband) % 6 == 0)
    fn_replaced = [re.sub('ch' + '\d', '', el) for el in fn_passband]
    unique_label = encode(pd.DataFrame(fn_replaced), 0).values.ravel()
    unique_indices = [np.where(unique_label == i)[0] for i in range(int(unique_label.max() + 1))]
    shapes = np.array([el.shape[0] for el in unique_indices])
    unique_indices = np.array(list(np.array(unique_indices)[shapes == 6]))
    assert(unique_indices.shape[1] == 6)
    p_groups = np.array(fn_passband)[unique_indices]
    counter = 0
    for p_group in tqdm(p_groups):
        assert(np.unique([re.sub('ch' + '\d', '', el) for el in p_group]).shape[0] == 1)
        wth_p_name = [re.sub('ch' + '\d', '', el) for el in p_group][0]
        gp_names = [gp_name for gp_name in list(fn_tree['nyanp'].keys()) if np.all(np.in1d(p_group, fn_tree['nyanp'][gp_name]))]
        assert(len(gp_names) == 1)
        gp_name = gp_names[0]
        
        stats = ['mean', 'sum', 'median', 'min', 'max', 'var']
        funcs = [np.mean, np.sum, np.nanmedian, np.min, np.max, np.var]
    
        stats_col_names = [wth_p_name + '_over_p_' + el for el in stats]
        normed_col_names = [el + '_normed' for el in p_group]
        if not np.all([os.path.exists('../features/' + el + '_' + data_type + '.npy') for el in stats_col_names + normed_col_names]):
            counter += 1
            p_df = load_dataframe_npy(p_group, data_type = data_type)

            np.warnings.filterwarnings('ignore')
            stats_df = pd.concat([pd.Series(func(p_df, axis = 1)) for func in funcs], axis = 1).astype(np.float32)
            stats_df.columns = stats_col_names

            normed_df = p_df.div(stats_df[wth_p_name + '_over_p_sum'], axis = 0).astype(np.float32)
            normed_df.columns = normed_col_names
            save_df_as_npy(normed_df, 'nyanp', 'processed', data_type, path='../features/')
            save_df_as_npy(stats_df, 'nyanp', 'processed', data_type, path='../features/')
    print(str(counter) + ' type of objects are saved')