import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import sys
import gc
sys.path.append('..')
from scripts.utils import * 
from joblib import Parallel, delayed
from itertools import chain

def save_hostgal_photoz_to_distmod_lgb(tr_m, tes_m):
    tr_wna = tr_m[~tr_m['distmod'].isnull()]
    tes_wna = tes_m[~tes_m['distmod'].isnull()]
    X = np.concatenate((tr_wna[['hostgal_photoz']].values, tes_wna[['hostgal_photoz']].values), axis = 0)
    y = np.concatenate((tr_wna['distmod'].values, tes_wna['distmod'].values), axis = 0)

    params = {
    'boosting_type': 'gbdt',
    'objective':     'regression',
    'metric':        'rmse',
    'learning_rate': 0.1,
    'num_leaves':    16,
    'max_depth':     -1, 
    'feature_fraction': 1, 
    'bagging_freq':     1,
    'bagging_fraction': 1,
    'lambda_l1':       1, 
    'lambda_l2':       1, 
    'verbosity':       1, 
    }

    i = 5
    params['feature_fraction_seed'] = i+1
    params['bagging_seed'] = (i+1)**2
    
    dtr = lgb.Dataset(X, y)
    dte = dtr.create_valid(X[:2], y[:2])
    gbm = lgb.train(params, dtr, num_boost_round = 400, valid_sets = [dte])
    save_pickle(gbm, '../models/exp_6_3_1.pkl')

def get_specz_dist(tr_m, gbm):
    tr_m_cp = tr_m.copy()
    nan_mask = (tr_m_cp['hostgal_specz'] != 0) & (~tr_m_cp['hostgal_specz'].isnull())
    tr_m_cp.loc[nan_mask, 'specz_dist'] = gbm.predict(tr_m_cp[nan_mask][['hostgal_specz']])
    specz_dist = tr_m_cp[['specz_dist']].astype(np.float32)
    return specz_dist
    
def get_specz_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr, data_type):
    col_name_agg = 'specz_dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']
    
    specz_dist = load_dataframe_npy(['specz_dist'], data_type)
    tr_m_cp = tr_m.copy()
    tr_m_cp['specz_dist'] = specz_dist

    merged = pd.merge(tr_cp, tr_m_cp[['object_id', 'specz_dist']], on  = 'object_id', how = 'left')
    merged['log_specz_dist_squared_shifted_flux'] = (2 * (merged['specz_dist'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_specz_dist_squared_shifted_flux']
    
    gp = tr_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_specz_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr, data_type):
    col_name_agg = 'specz_dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']
    
    specz_dist = load_dataframe_npy(['specz_dist'], data_type)
    tr_m_cp = tr_m.copy()
    tr_m_cp['specz_dist'] = specz_dist

    merged = pd.merge(tr_cp, tr_m_cp[['object_id', 'specz_dist']], on  = 'object_id', how = 'left')
    merged['log_specz_dist_squared_shifted_flux'] = (2 * (merged['specz_dist'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_specz_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    gp = tr_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_specz_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr, data_type):
    col_name_agg = 'specz_dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']
    
    specz_dist = load_dataframe_npy(['specz_dist'], data_type)
    tr_m_cp = tr_m.copy()
    tr_m_cp['specz_dist'] = specz_dist

    merged = pd.merge(tr_cp, tr_m_cp[['object_id', 'specz_dist']], on  = 'object_id', how = 'left')
    merged['log_specz_dist_squared_shifted_flux'] = (2 * (merged['specz_dist'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_specz_dist_squared_shifted_flux']
    
    shifted = tr_cp.groupby(['object_id', 'passband']).shift(1)
    tr_cp[col_name_agg + '_diff'] = tr_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tr_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_specz_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr, data_type):
    col_name_agg = 'specz_dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']
    
    specz_dist = load_dataframe_npy(['specz_dist'], data_type)
    tr_m_cp = tr_m.copy()
    tr_m_cp['specz_dist'] = specz_dist

    merged = pd.merge(tr_cp, tr_m_cp[['object_id', 'specz_dist']], on  = 'object_id', how = 'left')
    merged['log_specz_dist_squared_shifted_flux'] = (2 * (merged['specz_dist'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_specz_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    shifted = tr_det.groupby(['object_id', 'passband']).shift(1)
    tr_det[col_name_agg + '_diff'] = tr_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tr_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mjd_flux_nan_masks(tr_m, tr):
    new_df = pd.DataFrame(np.repeat(tr_m['object_id'], 6), columns = ['object_id']).reset_index(drop = True).reset_index()
    new_df['passband'] = np.repeat(np.arange(6)[np.newaxis, ...], tr_m.shape[0], axis = 0).ravel()
    merged = pd.merge(tr, new_df, how = 'left', on = ['object_id', 'passband']).rename(columns = {'index' : 'ob_p'})
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    unstack = merged[['ob_p', 'mjd', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
    mjd_uns = unstack['mjd'].values
    flux_uns = unstack['flux'].values
    nan_masks = ~np.isnan(mjd_uns)
    return mjd_uns, flux_uns, nan_masks

def process_01(mjd, flux):
    n_s = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    x = np.full(2 * len(n_s), np.nan, dtype = np.float32)
    idxmax = np.argmax(flux)
    top_flux = flux[idxmax]
    top_mjd = mjd[idxmax]
    for j in range(len(n_s)):
        n = n_s[j]
        n_percent_flux = top_flux * n/100
        diff = np.diff((flux > n_percent_flux).astype('int'))
        for_interps = np.where(diff)[0] 
        interp_mjds = []
        for k in range(len(for_interps)):
            idx = for_interps[k]
            mjd_0 = mjd[idx]
            mjd_1 = mjd[idx + 1]
            flux_0 = flux[idx]
            flux_1 = flux[idx + 1]
            spc_mjd = (mjd_1 * (flux_0 - n_percent_flux) + mjd_0 * (n_percent_flux - flux_1))/(flux_0 - flux_1)
            interp_mjds.append(spc_mjd)
        diffs = (top_mjd - interp_mjds)
        pos_diffs = diffs[diffs > 0]
        neg_diffs = diffs[diffs < 0]
        if pos_diffs.size != 0:
            left_diff = pos_diffs.min()
            x[j * 2 + 0] = left_diff
        if neg_diffs.size != 0:
            right_diff = neg_diffs.max()
            x[j * 2 + 1] = np.abs(right_diff)
    return x

def proc_X_01(tr_m, X):
    n_s = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    X_reshaped = X.reshape(tr_m.shape[0], 6 * 2 * len(n_s))
    f_tr = pd.DataFrame(X_reshaped)
    cols = []
    for i in range(6):
        cols_b = ['days_from_peak_to_' + str(n) + '_percent_flux_backward_passband_' + str(i) for n in n_s]
        cols_f = ['days_from_peak_to_' + str(n) + '_percent_flux_forward_passband_' + str(i) for n in n_s] 
        cols += list(chain.from_iterable([[el1, el2] for el1, el2 in zip(cols_b, cols_f)]))
    f_tr.columns = cols
    return f_tr

def get_days_from_peak_to_n_percent_flux(tr_m, tr):
    mjd_uns, flux_uns, nan_masks = get_mjd_flux_nan_masks(tr_m, tr)
    X = np.array(Parallel(n_jobs=8, verbose=10)( [delayed(process_01)(mjd[nan_mask], flux[nan_mask]) \
                                              for mjd, flux, nan_mask in zip(mjd_uns, flux_uns, nan_masks)] ))
    f_tr = proc_X_01(tr_m, X)
    return f_tr

def get_mjd_flux_nan_masks_min_corrected(tr_m, tr):
    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']
    
    new_df = pd.DataFrame(np.repeat(tr_m['object_id'], 6), columns = ['object_id']).reset_index(drop = True).reset_index()
    new_df['passband'] = np.repeat(np.arange(6)[np.newaxis, ...], tr_m.shape[0], axis = 0).ravel()
    merged = pd.merge(tr_cp, new_df, how = 'left', on = ['object_id', 'passband']).rename(columns = {'index' : 'ob_p'})
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    unstack = merged[['ob_p', 'mjd', 'shifted_flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
    mjd_uns = unstack['mjd'].values
    flux_uns = unstack['shifted_flux'].values
    nan_masks = ~np.isnan(mjd_uns)
    return mjd_uns, flux_uns, nan_masks

def process_02(mjd, flux):
    n_s = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    x = np.full(2 * len(n_s), np.nan, dtype = np.float32)
    idxmax = np.argmax(flux)
    top_flux = flux[idxmax]
    top_mjd = mjd[idxmax]
    for j in range(len(n_s)):
        n = n_s[j]
        n_percent_flux = top_flux * n/100
        diff = np.diff((flux > n_percent_flux).astype('int'))
        for_interps = np.where(diff)[0] 
        interp_mjds = []
        for k in range(len(for_interps)):
            idx = for_interps[k]
            mjd_0 = mjd[idx]
            mjd_1 = mjd[idx + 1]
            flux_0 = flux[idx]
            flux_1 = flux[idx + 1]
            spc_mjd = (mjd_1 * (flux_0 - n_percent_flux) + mjd_0 * (n_percent_flux - flux_1))/(flux_0 - flux_1)
            interp_mjds.append(spc_mjd)
        diffs = (top_mjd - interp_mjds)
        pos_diffs = diffs[diffs > 0]
        neg_diffs = diffs[diffs < 0]
        if pos_diffs.size != 0:
            left_diff = pos_diffs.min()
            x[j * 2 + 0] = left_diff
        if neg_diffs.size != 0:
            right_diff = neg_diffs.max()
            x[j * 2 + 1] = np.abs(right_diff)
    return x

def proc_X_02(tr_m, X):
    n_s = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    X_reshaped = X.reshape(tr_m.shape[0], 6 * 2 * len(n_s))
    f_tr = pd.DataFrame(X_reshaped)
    cols = []
    for i in range(6):
        cols_b = ['days_from_peak_to_' + str(n) + '_percent_flux_backward_min_corrected_passband_' + str(i) for n in n_s]
        cols_f = ['days_from_peak_to_' + str(n) + '_percent_flux_forward_min_corrected_passband_' + str(i) for n in n_s] 
        cols += list(chain.from_iterable([[el1, el2] for el1, el2 in zip(cols_b, cols_f)]))
    f_tr.columns = cols
    return f_tr

def get_days_from_peak_to_n_percent_flux_min_corrected(tr_m, tr):
    mjd_uns, flux_uns, nan_masks = get_mjd_flux_nan_masks_min_corrected(tr_m, tr)
    X = np.array(Parallel(n_jobs=8, verbose=10)( [delayed(process_02)(mjd[nan_mask], flux[nan_mask]) \
                                              for mjd, flux, nan_mask in zip(mjd_uns, flux_uns, nan_masks)] ))
    f_tr = proc_X_02(tr_m, X)
    return f_tr

def process_03(mjd, flux):
    n_s = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 250]
    n_s += [-el for el in n_s]
    x = np.full(len(n_s), np.nan, dtype = np.float32)
    idxmax = np.argmax(flux)
    top_flux = flux[idxmax]
    top_mjd = mjd[idxmax]
    for j in range(len(n_s)):
        n = n_s[j]
        after_n_days = top_mjd + n
        
        diff = np.diff((mjd > after_n_days).astype('int'))
        for_interps = np.where(diff)[0]
        if not for_interps:
            continue
        idx = for_interps[0]
        mjd_0 = mjd[idx]
        mjd_1 = mjd[idx + 1]
        flux_0 = flux[idx]
        flux_1 = flux[idx + 1]
        interp_flux = (flux_1 * (after_n_days - mjd_0) + flux_0 * (mjd_1 - after_n_days))/(mjd_1 - mjd_0)
        ratio = interp_flux/top_flux
        x[j] = ratio
    return x

def proc_X_03(tr_m, X):
    n_s = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 250]
    n_s += [-el for el in n_s]
    X_reshaped = X.reshape(tr_m.shape[0], 6 * len(n_s))
    f_tr = pd.DataFrame(X_reshaped)
    cols = []
    for i in range(6):
        cols += ['from_peak_to_percent_flux_' + str(n) + '_days_passband_' + str(i) for n in n_s]
    f_tr.columns = cols
    return f_tr

def get_from_peak_to_percent_flux_n_days(tr_m, tr):
    mjd_uns, flux_uns, nan_masks = get_mjd_flux_nan_masks(tr_m, tr)
    X = np.array(Parallel(n_jobs=8, verbose=10)( [delayed(process_03)(mjd[nan_mask], flux[nan_mask]) \
                                              for mjd, flux, nan_mask in zip(mjd_uns, flux_uns, nan_masks)] ))
    f_tr = proc_X_03(tr_m, X)
    return f_tr

def process_04(mjd, flux):
    n_s = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 250]
    n_s += [-el for el in n_s]
    x = np.full(len(n_s), np.nan, dtype = np.float32)
    idxmax = np.argmax(flux)
    top_flux = flux[idxmax]
    top_mjd = mjd[idxmax]
    for j in range(len(n_s)):
        n = n_s[j]
        after_n_days = top_mjd + n
        
        diff = np.diff((mjd > after_n_days).astype('int'))
        for_interps = np.where(diff)[0]
        if not for_interps:
            continue
        idx = for_interps[0]
        mjd_0 = mjd[idx]
        mjd_1 = mjd[idx + 1]
        flux_0 = flux[idx]
        flux_1 = flux[idx + 1]
        interp_flux = (flux_1 * (after_n_days - mjd_0) + flux_0 * (mjd_1 - after_n_days))/(mjd_1 - mjd_0)
        ratio = interp_flux/top_flux
        x[j] = ratio
    return x

def proc_X_04(tr_m, X):
    n_s = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 250]
    n_s += [-el for el in n_s]
    X_reshaped = X.reshape(tr_m.shape[0], 6 * len(n_s))
    f_tr = pd.DataFrame(X_reshaped)
    cols = []
    for i in range(6):
        cols += ['from_peak_to_percent_flux_' + str(n) + '_days_min_corrected_passband_' + str(i) for n in n_s]
    f_tr.columns = cols
    return f_tr

def get_from_peak_to_percent_flux_n_days_min_corrected(tr_m, tr):
    mjd_uns, flux_uns, nan_masks = get_mjd_flux_nan_masks(tr_m, tr)
    X = np.array(Parallel(n_jobs=8, verbose=10)( [delayed(process_04)(mjd[nan_mask], flux[nan_mask]) \
                                              for mjd, flux, nan_mask in zip(mjd_uns, flux_uns, nan_masks)] ))
    f_tr = proc_X_04(tr_m, X)
    return f_tr

def get_color_change_passband(data_type):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'std']
    funcs = [np.mean, np.sum, np.median, np.min, np.max, np.std]
    length = 256
    
    curve_tr = np.load('../curve/mm_scaled_mjd_flux_normalized_per_object_min_corrected_' + str(length) + '_' + data_type + '.npy')
    color_tr = curve_tr/(np.expand_dims(curve_tr.sum(axis = 1), axis = 1) + 1e-10)
    f_tr = pd.DataFrame()
    for stats, func in tqdm(zip(stats_agg, funcs)):
        col_names = ['color_change_' + stats + '_passband_' + str(i) for i in range(6)]
        f = func(color_tr, axis = 2)
        for i, col_name in enumerate(col_names):
            f_tr[col_name] = f[:, i]
    return f_tr

def get_diff_color_change_passband(data_type):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'std']
    funcs = [np.mean, np.sum, np.median, np.min, np.max, np.std]
    length = 256
    
    curve_tr = np.load('../curve/mm_scaled_mjd_flux_normalized_per_object_min_corrected_' + str(length) + '_' + data_type + '.npy')
    color_tr = curve_tr/(np.expand_dims(curve_tr.sum(axis = 1), axis = 1) + 1e-10)
    color_tr_diff = np.diff(color_tr, axis = 2)
    f_tr = pd.DataFrame()
    for stats, func in tqdm(zip(stats_agg, funcs)):
        col_names = ['diff_color_change_' + stats + '_passband_' + str(i) for i in range(6)]
        f = func(color_tr_diff, axis = 2)
        for i, col_name in enumerate(col_names):
            f_tr[col_name] = f[:, i]
    return f_tr

def get_mm_scaled_mjd_flux_normalized_per_object_min_corrected(tr_m, tr, length):
    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    
    new_df = pd.DataFrame(np.repeat(tr_m['object_id'], 6), columns = ['object_id']).reset_index(drop = True).reset_index()
    new_df['passband'] = np.repeat(np.arange(6)[np.newaxis, ...], tr_m.shape[0], axis = 0).ravel()
    tr_cp = pd.merge(tr, new_df, how = 'left', on = ['object_id', 'passband']).rename(columns = {'index' : 'ob_p'})
    tr_cp = pd.merge(tr_cp, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = (tr_cp['flux'] - tr_cp['min']).astype(np.float32)

    gp_mjd = tr_cp.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()

    merged = pd.merge(tr_cp, gp_mjd, how = 'left', on = 'object_id')

    merged['mm_scaled_mjd'] = (length - 1) * (merged['mjd'] - merged['mjd_min'])/(merged['mjd_max'] - merged['mjd_min'])

    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()

    unstack = merged[['ob_p', 'mm_scaled_mjd', 'shifted_flux', 'cc']].set_index(['ob_p', 'cc']).unstack()

    mjd_uns = unstack['mm_scaled_mjd'].values[..., np.newaxis]
    flux_uns = unstack['shifted_flux'].values[..., np.newaxis]
    mjd_flux = np.concatenate((mjd_uns, flux_uns), axis = 2)
    nan_masks = ~np.isnan(mjd_flux)[:, :, 0]

    x = np.arange(length)
    X = np.zeros((mjd_flux.shape[0], x.shape[0]))
    for i in tqdm(range(mjd_flux.shape[0])):
        intp = np.interp(x, mjd_flux[i][:, 0][nan_masks[i]], mjd_flux[i][:, 1][nan_masks[i]])
        X[i] = intp
    X_reshaped = X.reshape(tr_m.shape[0], 6, length).astype(np.float32)
    X_per_object = X_reshaped.reshape(X_reshaped.shape[0], length * 6)
    del X, merged; gc.collect()
    
    res = (X_reshaped)/\
    (1e-2 + X_per_object.std(axis = 1)[..., np.newaxis, np.newaxis])
    return res