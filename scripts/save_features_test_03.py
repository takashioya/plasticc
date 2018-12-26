from utils import * 
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_03 import * 
from utils import * 

def main():
    tr = feather.read_dataframe('../others/tr.feather')
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    tes = feather.read_dataframe('../others/tes.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    
    f_tes = get_ddf_flux_sorted_diff_stats(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'ddf_flux_sorted_diff', 'test', path='../features/')
    gc.collect()

    f_tes = get_ddf_flux_detected_sorted_diff_stats(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'ddf_flux_sorted_diff', 'test', path='../features/')
    gc.collect()

    f_tes = get_ddf_diff_flux_sorted_diff_stats(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'ddf_diff_flux_sorted_diff', 'test', path='../features/')
    gc.collect()

    f_tes = get_ddf_diff_flux_detected_sorted_diff_stats(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'ddf_diff_flux_sorted_diff', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mjd_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mjd_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mjd_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mjd_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mjd_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mjd_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mjd_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mjd_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_err_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_err_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_err_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_err_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_err_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_err_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_err_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_err_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_curve_angle_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'curve_angle_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_curve_angle_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'curve_angle_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_curve_angle_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_curve_angle_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_curve_angle_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_curve_angle_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_abs_curve_angle_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'abs_curve_angle_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_abs_curve_angle_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'abs_curve_angle_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_flux_n_sigma_stats_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'flux_n_sigma_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_flux_n_sigma_stats_detected_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'flux_n_sigma_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_flux_n_sigma_stats_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_flux_n_sigma_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_flux_n_sigma_stats_detected_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_flux_n_sigma_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mm_scaled_flux_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mm_scaled_flux_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mm_scaled_flux_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mm_scaled_flux_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_min_corrected_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_min_corrected_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_min_corrected_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_min_corrected_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_min_corrected_stats_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_min_corrected_stats_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_stats_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mjd_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mjd_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mjd_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mjd_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mjd_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mjd_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mjd_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mjd_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_err_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_err_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_flux_err_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_err_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_err_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_err_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_flux_err_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_err_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_curve_angle_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'curve_angle_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_curve_angle_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'curve_angle_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_curve_angle_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_curve_angle_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_curve_angle_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_curve_angle_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_abs_curve_angle_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'abs_curve_angle_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_abs_curve_angle_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'abs_curve_angle_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_flux_n_sigma_skew_kurt_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'flux_n_sigma_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_flux_n_sigma_skew_kurt_detected_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'flux_n_sigma_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_flux_n_sigma_skew_kurt_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_flux_n_sigma_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_flux_n_sigma_skew_kurt_detected_p_ignored(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_flux_n_sigma_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mm_scaled_flux_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_mm_scaled_flux_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mm_scaled_flux_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_mm_scaled_flux_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_dist_squared_shifted_flux_min_corrected_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    f_tes = get_diff_dist_squared_shifted_flux_min_corrected_skew_kurt_detected_p_ignored(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_skew_kurt_p_ignored', 'test', path='../features/')
    gc.collect()
    
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()