import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('..')
from scripts.utils import * 

def get_ddf_flux_sorted_diff_stats(tes_m, tes):
    num_group = 3
    threshold = 200
    col_name_lst = ['mean', 'var']
    final_col_name = 'ddf_flux_sorted_diff'
    for_agg = ['mean', 'var']
    stats_col = 'flux'
    
    ddf_object = tes_m[tes_m['ddf'] == 1]['object_id'].unique()
    ddf_tes = tes[tes['object_id'].isin(ddf_object)].copy()
    
    shifted = ddf_tes.groupby(['object_id', 'passband'])['mjd'].shift(1)
    diff = ddf_tes['mjd'] - shifted
    ddf_tes['diff'] = diff
    ddf_tes['is_gap'] = (ddf_tes['diff'] > threshold).astype('int')
    ddf_tes['period'] = ddf_tes.groupby(['object_id', 'passband'])['is_gap'].cumsum()
    
    gp = ddf_tes.groupby(['object_id', 'passband', 'period'])[stats_col].agg(for_agg).reset_index()
    
    gp_nums = []
    for col_name in tqdm(col_name_lst):
        gp_sorted = gp.sort_values(by = ['object_id', 'passband', col_name])
        gp_sorted['order'] = np.tile(np.arange(num_group), int(gp.shape[0]/num_group))
        gp_num = gp_sorted.drop(['period'], axis = 1).set_index(['object_id', 'passband', 'order'])[[col_name]].unstack(level = [1, 2])

        for passband in range(6):
            for i in range(num_group - 1):
                gp_num[col_name, passband, str(i) + '_' + str(i + 1) + '_diff'] = \
                gp_num[col_name][passband][i + 1] - gp_num[col_name][passband][i]
            gp_num[col_name, passband, 'min_max_diff'] = \
            gp_num[col_name][passband][num_group - 1] - gp_num[col_name][passband][0]
    
        col_names = [final_col_name + '_' + col_name + '_passband_' + str(col[1]) + '_order_' + str(col[2]) for col in gp_num.columns.values]
        gp_num.columns = col_names
        gp_nums.append(gp_num.copy())
    gp_num_all = pd.concat(gp_nums, axis = 1)
    merged = pd.merge(tes_m[['object_id']], gp_num_all.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_ddf_flux_detected_sorted_diff_stats(tes_m, tes):
    num_group = 3
    threshold = 200
    col_name_lst = ['mean', 'var']
    final_col_name = 'ddf_flux_detected_sorted_diff'
    for_agg = ['mean', 'var']
    stats_col = 'flux'
    
    ddf_object = tes_m[tes_m['ddf'] == 1]['object_id'].unique()
    ddf_tes = tes[tes['object_id'].isin(ddf_object)].copy()
    
    shifted = ddf_tes.groupby(['object_id', 'passband'])['mjd'].shift(1)
    diff = ddf_tes['mjd'] - shifted
    ddf_tes['diff'] = diff
    ddf_tes['is_gap'] = (ddf_tes['diff'] > threshold).astype('int')
    ddf_tes['period'] = ddf_tes.groupby(['object_id', 'passband'])['is_gap'].cumsum()
    
    ddf_tes = ddf_tes[ddf_tes['detected'] == 1]
    
    gp = ddf_tes.groupby(['object_id', 'passband', 'period'])[stats_col].agg(for_agg).reset_index()
    
    gp_nums = []
    for col_name in tqdm(col_name_lst):
        gp_resampled = gp.set_index(['object_id', 'passband', 'period']).unstack(level = [1, 2]).stack(level = [1, 2],  dropna = False).reset_index()
        gp_sorted = gp_resampled.sort_values(by = ['object_id', 'passband', col_name])
        gp_sorted['order'] = np.tile(np.arange(num_group), int(gp_sorted.shape[0]/num_group))
        gp_num = gp_sorted.drop(['period'], axis = 1).set_index(['object_id', 'passband', 'order'])[[col_name]].unstack(level = [1, 2])

        for passband in range(6):
            for i in range(num_group - 1):
                gp_num[col_name, passband, str(i) + '_' + str(i + 1) + '_diff'] = \
                gp_num[col_name][passband][i + 1] - gp_num[col_name][passband][i]
            gp_num[col_name, passband, 'min_max_diff'] = \
            gp_num[col_name][passband][num_group - 1] - gp_num[col_name][passband][0]
    
        col_names = [final_col_name + '_' + col_name + '_passband_' + str(col[1]) + '_order_' + str(col[2]) for col in gp_num.columns.values]
        gp_num.columns = col_names
        gp_nums.append(gp_num.copy())
    gp_num_all = pd.concat(gp_nums, axis = 1)
    merged = pd.merge(tes_m[['object_id']], gp_num_all.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_ddf_diff_flux_sorted_diff_stats(tes_m, tes):
    num_group = 3
    threshold = 200
    col_name_lst = ['mean', 'var']
    final_col_name = 'ddf_diff_flux_sorted_diff'
    for_agg = ['mean', 'var']
    stats_col = 'flux_diff'
    
    ddf_object = tes_m[tes_m['ddf'] == 1]['object_id'].unique()
    ddf_tes = tes[tes['object_id'].isin(ddf_object)].copy()
    
    shifted = ddf_tes.groupby(['object_id', 'passband'])['mjd'].shift(1)
    diff = ddf_tes['mjd'] - shifted
    ddf_tes['diff'] = diff
    ddf_tes['is_gap'] = (ddf_tes['diff'] > threshold).astype('int')
    ddf_tes['period'] = ddf_tes.groupby(['object_id', 'passband'])['is_gap'].cumsum()
    
    shifted = ddf_tes.groupby(['object_id', 'passband'])['flux'].shift(1)
    diff = ddf_tes['flux'] - shifted
    ddf_tes['flux_diff'] = diff
    
    gp = ddf_tes.groupby(['object_id', 'passband', 'period'])[stats_col].agg(for_agg).reset_index()
    
    gp_nums = []
    for col_name in tqdm(col_name_lst):
        gp_sorted = gp.sort_values(by = ['object_id', 'passband', col_name])
        gp_sorted['order'] = np.tile(np.arange(num_group), int(gp.shape[0]/num_group))
        gp_num = gp_sorted.drop(['period'], axis = 1).set_index(['object_id', 'passband', 'order'])[[col_name]].unstack(level = [1, 2])

        for passband in range(6):
            for i in range(num_group - 1):
                gp_num[col_name, passband, str(i) + '_' + str(i + 1) + '_diff'] = \
                gp_num[col_name][passband][i + 1] - gp_num[col_name][passband][i]
            gp_num[col_name, passband, 'min_max_diff'] = \
            gp_num[col_name][passband][num_group - 1] - gp_num[col_name][passband][0]
    
        col_names = [final_col_name + '_' + col_name + '_passband_' + str(col[1]) + '_order_' + str(col[2]) for col in gp_num.columns.values]
        gp_num.columns = col_names
        gp_nums.append(gp_num.copy())
    gp_num_all = pd.concat(gp_nums, axis = 1)
    merged = pd.merge(tes_m[['object_id']], gp_num_all.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_ddf_diff_flux_detected_sorted_diff_stats(tes_m, tes):
    num_group = 3
    threshold = 200
    col_name_lst = ['mean', 'var']
    final_col_name = 'ddf_diff_flux_detected_sorted_diff'
    for_agg = ['mean', 'var']
    stats_col = 'flux_diff'
    
    ddf_object = tes_m[tes_m['ddf'] == 1]['object_id'].unique()
    ddf_tes = tes[tes['object_id'].isin(ddf_object)].copy()
    
    shifted = ddf_tes.groupby(['object_id', 'passband'])['mjd'].shift(1)
    diff = ddf_tes['mjd'] - shifted
    ddf_tes['diff'] = diff
    ddf_tes['is_gap'] = (ddf_tes['diff'] > threshold).astype('int')
    ddf_tes['period'] = ddf_tes.groupby(['object_id', 'passband'])['is_gap'].cumsum()
    
    shifted = ddf_tes.groupby(['object_id', 'passband'])['flux'].shift(1)
    diff = ddf_tes['flux'] - shifted
    ddf_tes['flux_diff'] = diff
    
    ddf_tes = ddf_tes[ddf_tes['detected'] == 1]
    
    gp = ddf_tes.groupby(['object_id', 'passband', 'period'])[stats_col].agg(for_agg).reset_index()
    
    gp_nums = []
    for col_name in tqdm(col_name_lst):
        gp_resampled = gp.set_index(['object_id', 'passband', 'period']).unstack(level = [1, 2]).stack(level = [1, 2],  dropna = False).reset_index()
        gp_sorted = gp_resampled.sort_values(by = ['object_id', 'passband', col_name])
        gp_sorted['order'] = np.tile(np.arange(num_group), int(gp_sorted.shape[0]/num_group))
        gp_num = gp_sorted.drop(['period'], axis = 1).set_index(['object_id', 'passband', 'order'])[[col_name]].unstack(level = [1, 2])

        for passband in range(6):
            for i in range(num_group - 1):
                gp_num[col_name, passband, str(i) + '_' + str(i + 1) + '_diff'] = \
                gp_num[col_name][passband][i + 1] - gp_num[col_name][passband][i]
            gp_num[col_name, passband, 'min_max_diff'] = \
            gp_num[col_name][passband][num_group - 1] - gp_num[col_name][passband][0]
    
        col_names = [final_col_name + '_' + col_name + '_passband_' + str(col[1]) + '_order_' + str(col[2]) for col in gp_num.columns.values]
        gp_num.columns = col_names
        gp_nums.append(gp_num.copy())
    gp_num_all = pd.concat(gp_nums, axis = 1)
    merged = pd.merge(tes_m[['object_id']], gp_num_all.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mjd_stats_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = tes.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mjd_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = tes_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mjd_stats_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    gp = tes_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mjd_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    gp = tes_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_stats_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = tes.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = tes_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_stats_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    gp = tes_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    gp = tes_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_err_stats_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = tes.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_err_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = tes_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_err_stats_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_err_diff'] = tes_cp['flux_err'] - shifted['flux_err']
    gp = tes_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_err_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_err_diff'] = tes_det['flux_err'] - shifted['flux_err']
    gp = tes_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_curve_angle_stats_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    gp = tes_cp.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_curve_angle_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    gp = tes_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_curve_angle_stats_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_curve_angle_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tes_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_abs_curve_angle_stats_p_ignored(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['abs_curve_angle'] = np.abs(np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff']))
    
    gp = tes_cp.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_abs_curve_angle_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['abs_curve_angle'] = np.abs(np.arctan(tes_det['mjd_diff']/tes_det['flux_diff']))
    
    gp = tes_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_n_sigma_stats_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    gp = tes_cp.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_n_sigma_stats_detected_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    gp = tes_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_n_sigma_stats_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_n_sigma_stats_detected_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    gp = tes_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mm_scaled_flux_stats_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    gp = tes_cp.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1]
    
    gp = tes_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_stats_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_stats_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tes_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_stats_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    
    gp = tr_cp.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_stats_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    gp = tr_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_stats_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    
    shifted = tr_cp.groupby(['object_id']).shift(1)
    tr_cp[col_name_agg + '_diff'] = tr_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tr_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_stats_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    shifted = tr_det.groupby(['object_id']).shift(1)
    tr_det[col_name_agg + '_diff'] = tr_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tr_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_min_corrected_stats_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    
    gp = tr_cp.groupby(['object_id'])[col_name_agg].agg(stats_agg).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_min_corrected_stats_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    gp = tr_det.groupby(['object_id'])[col_name_agg].agg(stats_agg)\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_min_corrected_stats_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    
    shifted = tr_cp.groupby(['object_id']).shift(1)
    tr_cp[col_name_agg + '_diff'] = tr_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tr_cp.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_min_corrected_stats_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    shifted = tr_det.groupby(['object_id']).shift(1)
    tr_det[col_name_agg + '_diff'] = tr_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tr_det.groupby(['object_id'])[col_name_agg + '_diff'].agg(stats_agg)\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mjd_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = get_skew_kurt_from_df_ignored(tes, col_name_agg, ['object_id']).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mjd_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mjd_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mjd_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mjd'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = get_skew_kurt_from_df_ignored(tes, col_name_agg, ['object_id']).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_err_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = get_skew_kurt_from_df_ignored(tes, col_name_agg, ['object_id']).rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_err_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_err_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_err_diff'] = tes_cp['flux_err'] - shifted['flux_err']
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_err_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'flux_err'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_err_diff'] = tes_det['flux_err'] - shifted['flux_err']
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_curve_angle_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_curve_angle_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_curve_angle_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_curve_angle_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_abs_curve_angle_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['abs_curve_angle'] = np.abs(np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff']))
    
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_abs_curve_angle_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['abs_curve_angle'] = np.abs(np.arctan(tes_det['mjd_diff']/tes_det['flux_diff']))
    
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_n_sigma_skew_kurt_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_n_sigma_skew_kurt_detected_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_n_sigma_skew_kurt_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_n_sigma_skew_kurt_detected_p_ignored(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mm_scaled_flux_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1]
    
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_skew_kurt_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df_ignored(tes_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_skew_kurt_detected_p_ignored(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    
    shifted = tes_det.groupby(['object_id']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = get_skew_kurt_from_df_ignored(tes_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_skew_kurt_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    
    gp = get_skew_kurt_from_df_ignored(tr_cp, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_skew_kurt_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    gp = get_skew_kurt_from_df_ignored(tr_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_skew_kurt_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    
    shifted = tr_cp.groupby(['object_id']).shift(1)
    tr_cp[col_name_agg + '_diff'] = tr_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df_ignored(tr_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_skew_kurt_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    shifted = tr_det.groupby(['object_id']).shift(1)
    tr_det[col_name_agg + '_diff'] = tr_det[col_name_agg] - shifted[col_name_agg]
    
    gp = get_skew_kurt_from_df_ignored(tr_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    
    gp = get_skew_kurt_from_df_ignored(tr_cp, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_min_corrected_skew_kurt_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    gp = get_skew_kurt_from_df_ignored(tr_det, col_name_agg, ['object_id'])\
    .rename(columns = lambda x : col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    
    shifted = tr_cp.groupby(['object_id']).shift(1)
    tr_cp[col_name_agg + '_diff'] = tr_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df_ignored(tr_cp, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_min_corrected_skew_kurt_detected_p_ignored(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    shifted = tr_det.groupby(['object_id']).shift(1)
    tr_det[col_name_agg + '_diff'] = tr_det[col_name_agg] - shifted[col_name_agg]
    
    gp = get_skew_kurt_from_df_ignored(tr_det, col_name_agg + '_diff', ['object_id'])\
    .rename(columns = lambda x : 'diff_' + col_name_agg + '_' + x + '_p_ignored_detected')
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)