import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_02 import * 
from utils import * 


def main():
    tes = feather.read_dataframe('../others/tes.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')

    f_tes = get_specz_dist(tes_m, load_pickle('../models/exp_6_3_1.pkl'))
    save_df_as_npy_without_fn_tree(f_tes, 'test', path = '../features/')
    gc.collect()

    f_tes = get_specz_dist_squared_shifted_flux_min_corrected_stats_passband(tes_m, tes, 'test')
    save_df_as_npy(f_tes, 'ts', 'specz_dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_specz_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tes_m, tes, 'test')
    save_df_as_npy(f_tes, 'ts', 'specz_dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_specz_dist_squared_shifted_flux_min_corrected_stats_passband(tes_m, tes, 'test')
    save_df_as_npy(f_tes, 'ts', 'diff_specz_dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_specz_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tes_m, tes, 'test')
    save_df_as_npy(f_tes, 'ts', 'diff_specz_dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_days_from_peak_to_n_percent_flux(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'days_from_peak_to_n_percent_flux', 'test', path='../features/')
    gc.collect()

    f_tes = get_days_from_peak_to_n_percent_flux_min_corrected(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'days_from_peak_to_n_percent_flux_min_corrected', 'test', path='../features/')
    gc.collect()

    f_tes = get_from_peak_to_percent_flux_n_days(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'from_peak_to_percent_flux_n_days', 'test', path='../features/')
    gc.collect()

    f_tes = get_from_peak_to_percent_flux_n_days_min_corrected(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'from_peak_to_percent_flux_n_days_min_corrected', 'test', path='../features/')
    gc.collect()

    length = 256
    f_tes = get_mm_scaled_mjd_flux_normalized_per_object_min_corrected(tes_m, tes, length = length)
    np.save('../curve/mm_scaled_mjd_flux_normalized_per_object_min_corrected_' + str(length) + '_test' + '.npy', f_tes)
    gc.collect()

    f_tes = get_color_change_passband('test')
    save_df_as_npy(f_tes, 'ts', 'color_change', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_color_change_passband('test')
    save_df_as_npy(f_tes, 'ts', 'diff_color_change', 'test', path='../features/')
    gc.collect()
    
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()
