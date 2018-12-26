from utils import * 
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_01 import * 
from utils import * 

def main():
    tr = feather.read_dataframe('../others/tr.feather')
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    tes = feather.read_dataframe('../others/tes.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')

    f_tes = get_metadata(tes_m)
    save_df_as_npy(f_tes, 'meta', 'main', 'test')
    gc.collect()

    f_tes = get_num_points_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'num_points', 'test', path='../features/')
    gc.collect()

    f_tes = get_num_detected(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'num_detected', 'test', path='../features/')
    gc.collect()

    f_tes = get_num_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'num_detected', 'test', path='../features/')
    gc.collect()

    f_tes = get_ratio_detected(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'ratio_detected', 'test', path='../features/')
    gc.collect()

    f_tes = get_ratio_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'ratio_detected', 'test', path='../features/')
    gc.collect()

    f_tes = get_mjd_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mjd_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_mjd_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mjd_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_mjd_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mjd_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_mjd_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mjd_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_flux_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_stats', 'test', path='../features/')


    f_tes = get_flux_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_flux_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_flux_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_flux_err_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_err_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_flux_err_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'flux_err_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_flux_err_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_err_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_flux_err_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_flux_err_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_curve_angle_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_curve_angle_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_curve_angle_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_curve_angle_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_abs_curve_angle_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'abs_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_abs_curve_angle_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'abs_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_flux_n_sigma_stats_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_flux_n_sigma_stats_detected_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_flux_n_sigma_stats_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_flux_n_sigma_stats_detected_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_mm_scaled_flux_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_mm_scaled_flux_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_mm_scaled_flux_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_mm_scaled_flux_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_mm_scaled_flux_n_sigma_stats_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_mm_scaled_flux_n_sigma_stats_detected_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_mm_scaled_flux_n_sigma_stats_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tes = get_diff_mm_scaled_flux_n_sigma_stats_detected_passband(tes_m, tes, n)
        save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_n_sigma_stats', 'test', path='../features/')
    gc.collect()

    for num_split in tqdm([3, 5, 10]):
        f_tes = get_mm_scaled_flux_hist_passband(tes_m, tes, num_split)
        save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_hist', 'test', path='../features/')
    gc.collect()

    for num_split in tqdm([3, 5, 10]):
        f_tes = get_mm_scaled_flux_hist_detected_passband(tes_m, tes, num_split)
        save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_hist', 'test', path='../features/')
    gc.collect()

    f_tes = get_dist_squared_shifted_flux_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_dist_squared_shifted_flux_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_dist_squared_shifted_flux_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_dist_squared_shifted_flux_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_dist_squared_shifted_flux_min_corrected_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_dist_squared_shifted_flux_min_corrected_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_mm_scaled_flux_curve_angle_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_mm_scaled_flux_curve_angle_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_mm_scaled_flux_curve_angle_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_diff_mm_scaled_flux_curve_angle_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'diff_mm_scaled_flux_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_mm_scaled_flux_abs_curve_angle_stats_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_abs_curve_angle_stats', 'test', path='../features/')
    gc.collect()

    f_tes = get_mm_scaled_flux_abs_curve_angle_stats_detected_passband(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'mm_scaled_flux_abs_curve_angle_stats', 'test', path='../features/')
    gc.collect()
    
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()