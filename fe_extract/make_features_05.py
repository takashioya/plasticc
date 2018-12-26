import matplotlib
matplotlib.use('agg')
import feets
import GPy
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    return mjd_uns.astype(np.float32), flux_uns, nan_masks


def process_01(X, y):
    length = 1094.0653999999995
    sc = StandardScaler()
    y_sc = sc.fit_transform(y)
    model = GPy.models.GPRegression(X, y_sc)
    model.optimize()
    X_reg = np.arange(X.min() - length, X.max() + length, length/100)[..., np.newaxis]
    y_reg = model.predict(X_reg)[0]
    arg_max = np.argmax(y_reg)
    
    if (arg_max + 50 > y_reg.shape[0]) or (arg_max - 50 < 0):
        return np.full(100, np.nan, dtype = np.float32)
    
    y_reg = y_reg[arg_max - 50:arg_max + 50].ravel()
    y_reg = y_reg/y_reg.max()
    return y_reg


def proc_X_01(tr_m, X):
    feature_names = ['gp_fitted_' + str(i) for i in range(100)]
    X_reshaped = X.reshape(tr_m.shape[0], 6 * 100)
    f_tr = pd.DataFrame(X_reshaped)
    cols = []
    for i in range(6):
        cols += [el + '_passband_' + str(i) for el in feature_names]
    f_tr.columns = cols
    return f_tr


def get_gp_fitted(tr_m, tr):
    warnings.filterwarnings('ignore')
    mjd_uns, flux_uns, nan_masks = get_mjd_flux_nan_masks(tr_m, tr)
    X = np.zeros((mjd_uns.shape[0], 100), dtype = np.float32)
    for i in tqdm(range(mjd_uns.shape[0])):
        mjd = mjd_uns[i][nan_masks[i]][..., np.newaxis]
        flux = flux_uns[i][nan_masks[i]][..., np.newaxis]
        X[i] = process_01(mjd, flux)
    f_tr = proc_X_01(tr_m, X)
    return f_tr


def get_mjd_flux_flux_err_nan_masks(tr_m, tr):
    new_df = pd.DataFrame(np.repeat(tr_m['object_id'], 6), columns = ['object_id']).reset_index(drop = True).reset_index()
    new_df['passband'] = np.repeat(np.arange(6)[np.newaxis, ...], tr_m.shape[0], axis = 0).ravel()
    merged = pd.merge(tr, new_df, how = 'left', on = ['object_id', 'passband']).rename(columns = {'index' : 'ob_p'})
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    unstack = merged[['ob_p', 'mjd', 'flux', 'flux_err', 'cc']].set_index(['ob_p', 'cc']).unstack()
    mjd_uns = unstack['mjd'].values
    flux_uns = unstack['flux'].values
    flux_err_uns = unstack['flux_err'].values
    nan_masks = ~np.isnan(mjd_uns)
    return mjd_uns.astype(np.float32), flux_uns.astype(np.float32), flux_err_uns.astype(np.float32), nan_masks


def process_02(lc):
    if lc[0].shape[0] < 20:
        return np.full(63, np.nan, dtype = np.float32)
    fs = feets.FeatureSpace(data = ['time', 'magnitude', 'error'])
    try:
        _, values = fs.extract(*lc)
    except:
        return np.full(63, np.nan, dtype = np.float32)
    return values.astype(np.float32)


def proc_X_02(tr_m, X):
    feature_names = ['Amplitude',
     'AndersonDarling',
     'Autocor_length',
     'Beyond1Std',
     'CAR_mean',
     'CAR_sigma',
     'CAR_tau',
     'Con',
     'Eta_e',
     'FluxPercentileRatioMid20',
     'FluxPercentileRatioMid35',
     'FluxPercentileRatioMid50',
     'FluxPercentileRatioMid65',
     'FluxPercentileRatioMid80',
     'Freq1_harmonics_amplitude_0',
     'Freq1_harmonics_amplitude_1',
     'Freq1_harmonics_amplitude_2',
     'Freq1_harmonics_amplitude_3',
     'Freq1_harmonics_rel_phase_0',
     'Freq1_harmonics_rel_phase_1',
     'Freq1_harmonics_rel_phase_2',
     'Freq1_harmonics_rel_phase_3',
     'Freq2_harmonics_amplitude_0',
     'Freq2_harmonics_amplitude_1',
     'Freq2_harmonics_amplitude_2',
     'Freq2_harmonics_amplitude_3',
     'Freq2_harmonics_rel_phase_0',
     'Freq2_harmonics_rel_phase_1',
     'Freq2_harmonics_rel_phase_2',
     'Freq2_harmonics_rel_phase_3',
     'Freq3_harmonics_amplitude_0',
     'Freq3_harmonics_amplitude_1',
     'Freq3_harmonics_amplitude_2',
     'Freq3_harmonics_amplitude_3',
     'Freq3_harmonics_rel_phase_0',
     'Freq3_harmonics_rel_phase_1',
     'Freq3_harmonics_rel_phase_2',
     'Freq3_harmonics_rel_phase_3',
     'Gskew',
     'LinearTrend',
     'MaxSlope',
     'Mean',
     'Meanvariance',
     'MedianAbsDev',
     'MedianBRP',
     'PairSlopeTrend',
     'PercentAmplitude',
     'PercentDifferenceFluxPercentile',
     'PeriodLS',
     'Period_fit',
     'Psi_CS',
     'Psi_eta',
     'Q31',
     'Rcs',
     'Skew',
     'SlottedA_length',
     'SmallKurtosis',
     'Std',
     'StetsonK',
     'StetsonK_AC',
     'StructureFunction_index_21',
     'StructureFunction_index_31',
     'StructureFunction_index_32']
    X_reshaped = X.reshape(tr_m.shape[0], 6 * 63)
    f_tr = pd.DataFrame(X_reshaped)
    cols = []
    for i in range(6):
        cols += [el + '_passband_' + str(i) for el in feature_names]
    f_tr.columns = cols
    return f_tr


def get_feets(tr_m, tr):
    warnings.filterwarnings('ignore')
    mjd_uns, flux_uns, flux_err_uns, nan_masks = get_mjd_flux_flux_err_nan_masks(tr_m, tr)
    X = np.array(Parallel(n_jobs=-1, verbose=10)( [delayed(process_02)(np.array([mjd[nan_mask], flux[nan_mask], flux_err[nan_mask]])) \
                                              for mjd, flux, flux_err, nan_mask in zip(mjd_uns, flux_uns, flux_err_uns, nan_masks)]))
    f_tr = proc_X_02(tr_m, X)
    return f_tr

