import numpy as np
import pandas as pd
from tqdm import tqdm

def get_metadata(tr_m):
    if 'target' in tr_m.columns:
        return tr_m.drop(['target'], axis = 1)
    else: 
        return tr_m
    
def get_num_points_passband(tes_m, tes):
    gp_size = tes.groupby(['object_id', 'passband']).size()
    gp_size = gp_size.unstack(level = -1).rename(columns = lambda el : 'num_points_passband_' + str(el)).reset_index()
    merged = pd.merge(tes_m[['object_id']], gp_size, how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_num_detected(tes_m, tes):
    gp = tes.groupby('object_id')['detected'].sum().reset_index().rename(columns = {'detected' : 'num_detected'})
    merged = pd.merge(tes_m[['object_id']], gp, how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_num_detected_passband(tes_m, tes):
    gp = tes.groupby(['object_id', 'passband'])['detected'].sum().unstack(level = -1)
    gp = gp.rename(columns = lambda el : 'num_detected_passband_' + str(el)).reset_index()
    merged = pd.merge(tes_m[['object_id']], gp, on = 'object_id', how = 'left').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_ratio_detected(tes_m, tes):
    gp = tes.groupby('object_id')['detected'].mean().reset_index().rename(columns = {'detected' : 'ratio_detected'})
    merged = pd.merge(tes_m[['object_id']], gp, how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_ratio_detected_passband(tes_m, tes):
    gp = tes.groupby(['object_id', 'passband'])['detected'].mean().unstack(level = -1)
    gp = gp.rename(columns = lambda el : 'ratio_detected_passband_' + str(el)).reset_index()
    merged = pd.merge(tes_m[['object_id']], gp, on = 'object_id', how = 'left').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mjd_stats_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = tes.groupby(['object_id', 'passband'])['mjd'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['mjd_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mjd_stats_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = tes_det.groupby(['object_id', 'passband'])['mjd'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['mjd_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mjd_stats_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    gp = tes_cp.groupby(['object_id', 'passband'])['mjd_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_mjd_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mjd_stats_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    gp = tes_det.groupby(['object_id', 'passband'])['mjd_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_mjd_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_stats_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = tes.groupby(['object_id', 'passband'])['flux'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_stats_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = tes_det.groupby(['object_id', 'passband'])['flux'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_stats_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    gp = tes_cp.groupby(['object_id', 'passband'])['flux_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_stats_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    gp = tes_det.groupby(['object_id', 'passband'])['flux_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_err_stats_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = tes.groupby(['object_id', 'passband'])['flux_err'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_err_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_err_stats_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = tes_det.groupby(['object_id', 'passband'])['flux_err'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_err_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_err_stats_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_err_diff'] = tes_cp['flux_err'] - shifted['flux_err']
    gp = tes_cp.groupby(['object_id', 'passband'])['flux_err_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_err_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_err_stats_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_err_diff'] = tes_det['flux_err'] - shifted['flux_err']
    gp = tes_det.groupby(['object_id', 'passband'])['flux_err_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_err_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_curve_angle_stats_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_curve_angle_stats_detected_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_curve_angle_stats_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_curve_angle_stats_detected_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_abs_curve_angle_stats_passband(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['abs_curve_angle'] = np.abs(np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff']))
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_abs_curve_angle_stats_detected_passband(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['abs_curve_angle'] = np.abs(np.arctan(tes_det['mjd_diff']/tes_det['flux_diff']))
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_n_sigma_stats_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_flux_n_sigma_stats_detected_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_n_sigma_stats_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_flux_n_sigma_stats_detected_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mm_scaled_flux_stats_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_stats_detected_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1]
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_stats_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_stats_detected_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_n_sigma_stats_passband(tes_m, tes, n):
    col_name_agg = 'mm_scaled_flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    tes_cp[col_name_agg] = tes_cp['mm_scaled_flux'] + n * tes_cp['flux_err']/(tes_cp['max'] - tes_cp['min'])
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_n_sigma_stats_detected_passband(tes_m, tes, n):
    col_name_agg = 'mm_scaled_flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['mm_scaled_flux'] + n * tes_det['flux_err']/(tes_det['max'] - tes_det['min'])
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_n_sigma_stats_passband(tes_m, tes, n):
    col_name_agg = 'mm_scaled_flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    tes_cp[col_name_agg] = tes_cp['mm_scaled_flux'] + n * tes_cp['flux_err']/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_n_sigma_stats_detected_passband(tes_m, tes, n):
    col_name_agg = 'mm_scaled_flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['mm_scaled_flux'] + n * tes_det['flux_err']/(tes_det['max'] - tes_det['min'])
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_hist_passband(tes_m, tes, num_split):
    col_name_agg = 'mm_scaled_flux'
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])

    ranges = [(i, i + 1/num_split) for i in np.arange(0, 1, 1/num_split)]
    mergeds = []

    for i in tqdm(range(num_split)):
        spc_range = ranges[i]
        range_col_name = col_name_agg + '_from_' + str(spc_range[0]) + '_to_' + str(spc_range[1])
        tes_cp[range_col_name] = ((tes_cp[col_name_agg] >= spc_range[0]) & (tes_cp[col_name_agg] <= spc_range[1])).astype(np.uint8)
        gp = tes_cp.groupby(['object_id', 'passband'])[range_col_name].agg(['mean', 'sum']).unstack(level = -1)
    
        names_1 = gp.columns.get_level_values(0)
        names_2 = gp.columns.get_level_values(1)
        gp.columns = [range_col_name + '_hist_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
        merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
        mergeds.append(merged)
    concated = pd.concat(mergeds, axis = 1)
    return concated.astype(np.float32)

def get_mm_scaled_flux_hist_detected_passband(tes_m, tes, num_split):
    col_name_agg = 'mm_scaled_flux'
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    tes_cp = tes_cp[tes_cp['detected'] == 1].copy()

    ranges = [(i, i + 1/num_split) for i in np.arange(0, 1, 1/num_split)]
    mergeds = []

    for i in tqdm(range(num_split)):
        spc_range = ranges[i]
        range_col_name = col_name_agg + '_from_' + str(spc_range[0]) + '_to_' + str(spc_range[1])
        tes_cp[range_col_name] = ((tes_cp[col_name_agg] >= spc_range[0]) & (tes_cp[col_name_agg] <= spc_range[1])).astype(np.uint8)
        gp = tes_cp.groupby(['object_id', 'passband'])[range_col_name].agg(['mean', 'sum']).unstack(level = -1)
    
        names_1 = gp.columns.get_level_values(0)
        names_2 = gp.columns.get_level_values(1)
        gp.columns = [range_col_name + '_hist_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
        merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
        mergeds.append(merged)
    concated = pd.concat(mergeds, axis = 1)
    return concated.astype(np.float32)

def get_dist_squared_shifted_flux_stats_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    
    gp = tr_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_stats_detected_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    gp = tr_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_stats_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    
    shifted = tr_cp.groupby(['object_id', 'passband']).shift(1)
    tr_cp[col_name_agg + '_diff'] = tr_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tr_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_stats_detected_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    shifted = tr_det.groupby(['object_id', 'passband']).shift(1)
    tr_det[col_name_agg + '_diff'] = tr_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tr_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    
    gp = tr_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    gp = tr_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    
    shifted = tr_cp.groupby(['object_id', 'passband']).shift(1)
    tr_cp[col_name_agg + '_diff'] = tr_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tr_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux_min_corrected'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    flux_min.loc[flux_min['min'] > 0, 'min'] = 0 
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp[col_name_agg] = 10 ** merged['log_dist_squared_shifted_flux']
    tr_det = tr_cp[tr_cp['detected'] == 1].copy()
    
    shifted = tr_det.groupby(['object_id', 'passband']).shift(1)
    tr_det[col_name_agg + '_diff'] = tr_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tr_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mm_scaled_flux_curve_angle_stats_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['mm_scaled_flux_diff'] = tes_cp['mm_scaled_flux'] - shifted['mm_scaled_flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp[col_name_agg] = np.arctan(tes_cp['mjd_diff']/tes_cp['mm_scaled_flux_diff'])
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_curve_angle_stats_detected_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['mm_scaled_flux_diff'] = tes_det['mm_scaled_flux'] - shifted['mm_scaled_flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det[col_name_agg] = np.arctan(tes_det['mjd_diff']/tes_det['mm_scaled_flux_diff'])
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_curve_angle_stats_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['mm_scaled_flux_diff'] = tes_cp['mm_scaled_flux'] - shifted['mm_scaled_flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp[col_name_agg] = np.arctan(tes_cp['mjd_diff']/tes_cp['mm_scaled_flux_diff'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_diff_mm_scaled_flux_curve_angle_stats_detected_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['mm_scaled_flux_diff'] = tes_det['mm_scaled_flux'] - shifted['mm_scaled_flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det[col_name_agg] = np.arctan(tes_det['mjd_diff']/tes_det['mm_scaled_flux_diff'])
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg + '_diff'].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)

def get_mm_scaled_flux_abs_curve_angle_stats_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux_abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['mm_scaled_flux_diff'] = tes_cp['mm_scaled_flux'] - shifted['mm_scaled_flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp[col_name_agg] = np.abs(np.arctan(tes_cp['mjd_diff']/tes_cp['mm_scaled_flux_diff']))
    
    gp = tes_cp.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)

def get_mm_scaled_flux_abs_curve_angle_stats_detected_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux_abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['mm_scaled_flux_diff'] = tes_det['mm_scaled_flux'] - shifted['mm_scaled_flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det[col_name_agg] = np.abs(np.arctan(tes_det['mjd_diff']/tes_det['mm_scaled_flux_diff']))
    
    gp = tes_det.groupby(['object_id', 'passband'])[col_name_agg].agg(stats_agg).unstack(level = -1)
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)