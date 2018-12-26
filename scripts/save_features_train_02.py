from utils import * 
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_02 import * 
from utils import * 

def main():
    tr = feather.read_dataframe('../others/tr.feather')
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    tes = feather.read_dataframe('../others/tes.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')

    save_hostgal_photoz_to_distmod_lgb(tr_m, tes_m)
    f_tr = get_specz_dist(tr_m, load_pickle('../models/exp_6_3_1.pkl'))
    save_df_as_npy_without_fn_tree(f_tr, 'train', path = '../features/')

    f_tr = get_specz_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr, 'train')
    save_df_as_npy(f_tr, 'ts', 'specz_dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_specz_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr, 'train')
    save_df_as_npy(f_tr, 'ts', 'specz_dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_diff_specz_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr, 'train')
    save_df_as_npy(f_tr, 'ts', 'diff_specz_dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_diff_specz_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr, 'train')
    save_df_as_npy(f_tr, 'ts', 'diff_specz_dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_days_from_peak_to_n_percent_flux(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'days_from_peak_to_n_percent_flux', 'train', path='../features/')

    f_tr = get_days_from_peak_to_n_percent_flux_min_corrected(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'days_from_peak_to_n_percent_flux_min_corrected', 'train', path='../features/')

    f_tr = get_from_peak_to_percent_flux_n_days(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'from_peak_to_percent_flux_n_days', 'train', path='../features/')

    f_tr = get_from_peak_to_percent_flux_n_days_min_corrected(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'from_peak_to_percent_flux_n_days_min_corrected', 'train', path='../features/')

    length = 256
    f_tr = get_mm_scaled_mjd_flux_normalized_per_object_min_corrected(tr_m, tr, length = length)
    np.save('../curve/mm_scaled_mjd_flux_normalized_per_object_min_corrected_' + str(length) + '_train' + '.npy', f_tr)

    f_tr = get_color_change_passband('train')
    save_df_as_npy(f_tr, 'ts', 'color_change', 'train', path='../features/')

    f_tr = get_diff_color_change_passband('train')
    save_df_as_npy(f_tr, 'ts', 'diff_color_change', 'train', path='../features/')
    
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()