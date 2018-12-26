import sys
sys.path.append('..')
from scripts.utils import *
import numpy as np
import pandas as pd


def get_mjd_skew_kurt_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = get_skew_kurt_from_df(tes, 'mjd', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['mjd_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_mjd_skew_kurt_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = get_skew_kurt_from_df(tes_det, 'mjd', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['mjd_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_mjd_skew_kurt_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    
    gp = get_skew_kurt_from_df(tes_cp, 'mjd_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_mjd_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_mjd_skew_kurt_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    
    gp = get_skew_kurt_from_df(tes_det, 'mjd_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_mjd_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_flux_skew_kurt_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = get_skew_kurt_from_df(tes, 'flux', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_flux_skew_kurt_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = get_skew_kurt_from_df(tes_det, 'flux', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_flux_skew_kurt_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    gp = get_skew_kurt_from_df(tes_cp, 'flux_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_flux_skew_kurt_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    gp = get_skew_kurt_from_df(tes_det, 'flux_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_flux_err_skew_kurt_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    gp = get_skew_kurt_from_df(tes, 'flux_err', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_err_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_flux_err_skew_kurt_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1]
    gp = get_skew_kurt_from_df(tes_det, 'flux_err', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['flux_err_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_flux_err_skew_kurt_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_err_diff'] = tes_cp['flux_err'] - shifted['flux_err']
    gp = get_skew_kurt_from_df(tes_cp, 'flux_err_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_err_' + el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_flux_err_skew_kurt_detected_passband(tes_m, tes):
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_err_diff'] = tes_det['flux_err'] - shifted['flux_err']
    gp = get_skew_kurt_from_df(tes_det, 'flux_err_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_flux_err_' + el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_curve_angle_skew_kurt_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    gp = get_skew_kurt_from_df(tes_cp, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_curve_angle_skew_kurt_detected_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    gp = get_skew_kurt_from_df(tes_det, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_curve_angle_skew_kurt_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['curve_angle'] = np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df(tes_cp, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_curve_angle_skew_kurt_detected_passband(tes_m, tes):
    col_name_agg = 'curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['curve_angle'] = np.arctan(tes_det['mjd_diff']/tes_det['flux_diff'])
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = get_skew_kurt_from_df(tes_det, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_abs_curve_angle_skew_kurt_passband(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp['flux_diff'] = tes_cp['flux'] - shifted['flux']
    tes_cp['mjd_diff'] = tes_cp['mjd'] - shifted['mjd']
    tes_cp['abs_curve_angle'] = np.abs(np.arctan(tes_cp['mjd_diff']/tes_cp['flux_diff']))
    
    gp = get_skew_kurt_from_df(tes_cp, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_abs_curve_angle_skew_kurt_detected_passband(tes_m, tes):
    col_name_agg = 'abs_curve_angle'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det['flux_diff'] = tes_det['flux'] - shifted['flux']
    tes_det['mjd_diff'] = tes_det['mjd'] - shifted['mjd']
    tes_det['abs_curve_angle'] = np.abs(np.arctan(tes_det['mjd_diff']/tes_det['flux_diff']))
    
    gp = get_skew_kurt_from_df(tes_det, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_flux_n_sigma_skew_kurt_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    gp = get_skew_kurt_from_df(tes_cp, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_flux_n_sigma_skew_kurt_detected_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    gp = get_skew_kurt_from_df(tes_det, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_flux_n_sigma_skew_kurt_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_cp = tes.copy()
    tes_cp[col_name_agg] = tes_cp['flux'] + n * tes_cp['flux_err']
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df(tes_cp, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_flux_n_sigma_skew_kurt_detected_passband(tes_m, tes, n):
    col_name_agg = 'flux_' + str(n) + '_sigma'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    tes_det = tes[tes['detected'] == 1].copy()
    tes_det[col_name_agg] = tes_det['flux'] + n * tes_det['flux_err']
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = get_skew_kurt_from_df(tes_det, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_mm_scaled_flux_skew_kurt_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    gp = get_skew_kurt_from_df(tes_cp, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)


def get_mm_scaled_flux_skew_kurt_detected_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1]
    
    gp = get_skew_kurt_from_df(tes_det, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)


def get_diff_mm_scaled_flux_skew_kurt_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    tes_cp = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    tes_cp['mm_scaled_flux'] = (tes_cp['flux'] - tes_cp['min'])/(tes_cp['max'] - tes_cp['min'])
    
    shifted = tes_cp.groupby(['object_id', 'passband']).shift(1)
    tes_cp[col_name_agg + '_diff'] = tes_cp[col_name_agg] - shifted[col_name_agg] 
    
    gp = get_skew_kurt_from_df(tes_cp, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)


def get_diff_mm_scaled_flux_skew_kurt_detected_passband(tes_m, tes):
    col_name_agg = 'mm_scaled_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']
    
    flux_min_max = tes.groupby(['object_id', 'passband'])['flux'].agg(['min', 'max']).reset_index()
    merged = pd.merge(tes, flux_min_max, how = 'left', on = ['object_id', 'passband'])
    merged['mm_scaled_flux'] = (merged['flux'] - merged['min'])/(merged['max'] - merged['min'])
    tes_det = merged[merged['detected'] == 1].copy()
    
    shifted = tes_det.groupby(['object_id', 'passband']).shift(1)
    tes_det[col_name_agg + '_diff'] = tes_det[col_name_agg] - shifted[col_name_agg]
    
    gp = get_skew_kurt_from_df(tes_det, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tes_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    
    return merged.astype(np.float32)


def get_dist_squared_shifted_flux_skew_kurt_passband(tr_m, tr):
    col_name_agg = 'dist_squared_shifted_flux'
    stats_agg = ['mean', 'sum', 'median', 'min', 'max', 'var']

    flux_min = tr.groupby(['object_id', 'passband'])['flux'].agg(['min']).reset_index()
    tr_cp = pd.merge(tr, flux_min, how = 'left', on = ['object_id', 'passband'])
    tr_cp['shifted_flux'] = tr_cp['flux'] - tr_cp['min']

    merged = pd.merge(tr_cp, tr_m[['object_id', 'distmod']], on  = 'object_id', how = 'left')
    merged['log_dist_squared_shifted_flux'] = (2 * (merged['distmod'] + 5)/5 + np.log10(merged['shifted_flux']))
    merged[merged == -np.inf] = np.NaN
    tr_cp['dist_squared_shifted_flux'] = 10 ** merged['log_dist_squared_shifted_flux']
    
    gp = get_skew_kurt_from_df(tr_cp, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_dist_squared_shifted_flux_skew_kurt_detected_passband(tr_m, tr):
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
    
    gp = get_skew_kurt_from_df(tr_det, col_name_agg, ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = [col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_dist_squared_shifted_flux_skew_kurt_passband(tr_m, tr):
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
    
    gp = get_skew_kurt_from_df(tr_cp, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)


def get_diff_dist_squared_shifted_flux_skew_kurt_detected_passband(tr_m, tr):
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
    
    gp = get_skew_kurt_from_df(tr_det, col_name_agg + '_diff', ['object_id', 'passband'])
    names_1 = gp.columns.get_level_values(0)
    names_2 = gp.columns.get_level_values(1)
    gp.columns = ['diff_' + col_name_agg + '_' +  el1 + '_passband_' + str(el2) + '_detected' for (el1, el2) in zip(names_1, names_2)]
    merged = pd.merge(tr_m[['object_id']], gp.reset_index(), how = 'left', on = 'object_id').drop(['object_id'], axis = 1)
    return merged.astype(np.float32)