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

    f_tr = get_metadata(tr_m)
    save_df_as_npy(f_tr, 'meta', 'main', 'train')

    f_tr = get_num_points_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'num_points', 'train', path='../features/')

    f_tr = get_num_detected(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'num_detected', 'train', path='../features/')

    f_tr = get_num_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'num_detected', 'train', path='../features/')

    f_tr = get_ratio_detected(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'ratio_detected', 'train', path='../features/')

    f_tr = get_ratio_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'ratio_detected', 'train', path='../features/')

    f_tr = get_mjd_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mjd_stats', 'train', path='../features/')

    f_tr = get_mjd_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mjd_stats', 'train', path='../features/')

    f_tr = get_diff_mjd_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mjd_stats', 'train', path='../features/')

    f_tr = get_diff_mjd_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mjd_stats', 'train', path='../features/')

    f_tr = get_flux_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_stats', 'train', path='../features/')

    f_tr = get_flux_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_stats', 'train', path='../features/')

    f_tr = get_diff_flux_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_stats', 'train', path='../features/')

    f_tr = get_diff_flux_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_stats', 'train', path='../features/')

    f_tr = get_flux_err_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_err_stats', 'train', path='../features/')

    f_tr = get_flux_err_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_err_stats', 'train', path='../features/')

    f_tr = get_diff_flux_err_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_err_stats', 'train', path='../features/')

    f_tr = get_diff_flux_err_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_err_stats', 'train', path='../features/')

    f_tr = get_curve_angle_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'curve_angle_stats', 'train', path='../features/')

    f_tr = get_curve_angle_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'curve_angle_stats', 'train', path='../features/')

    f_tr = get_diff_curve_angle_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_curve_angle_stats', 'train', path='../features/')

    f_tr = get_diff_curve_angle_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_curve_angle_stats', 'train', path='../features/')

    f_tr = get_abs_curve_angle_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'abs_curve_angle_stats', 'train', path='../features/')

    f_tr = get_abs_curve_angle_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'abs_curve_angle_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_flux_n_sigma_stats_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'flux_n_sigma_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_flux_n_sigma_stats_detected_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'flux_n_sigma_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_diff_flux_n_sigma_stats_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'diff_flux_n_sigma_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_diff_flux_n_sigma_stats_detected_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'diff_flux_n_sigma_stats', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_stats', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_stats', 'train', path='../features/')

    f_tr = get_diff_mm_scaled_flux_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_stats', 'train', path='../features/')

    f_tr = get_diff_mm_scaled_flux_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_mm_scaled_flux_n_sigma_stats_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_n_sigma_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_mm_scaled_flux_n_sigma_stats_detected_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_n_sigma_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_diff_mm_scaled_flux_n_sigma_stats_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_n_sigma_stats', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_diff_mm_scaled_flux_n_sigma_stats_detected_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_n_sigma_stats', 'train', path='../features/')

    for num_split in tqdm([3, 5, 10]):
        f_tr = get_mm_scaled_flux_hist_passband(tr_m, tr, num_split)
        save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_hist', 'train', path='../features/')

    for num_split in tqdm([3, 5, 10]):
        f_tr = get_mm_scaled_flux_hist_detected_passband(tr_m, tr, num_split)
        save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_hist', 'train', path='../features/')

    f_tr = get_dist_squared_shifted_flux_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'dist_squared_shifted_flux_stats', 'train', path='../features/')

    f_tr = get_dist_squared_shifted_flux_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'dist_squared_shifted_flux_stats', 'train', path='../features/')

    f_tr = get_diff_dist_squared_shifted_flux_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_dist_squared_shifted_flux_stats', 'train', path='../features/')

    f_tr = get_diff_dist_squared_shifted_flux_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_dist_squared_shifted_flux_stats', 'train', path='../features/')

    f_tr = get_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_diff_dist_squared_shifted_flux_min_corrected_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_diff_dist_squared_shifted_flux_min_corrected_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_dist_squared_shifted_flux_min_corrected_stats', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_curve_angle_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_curve_angle_stats', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_curve_angle_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_curve_angle_stats', 'train', path='../features/')

    f_tr = get_diff_mm_scaled_flux_curve_angle_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_curve_angle_stats', 'train', path='../features/')

    f_tr = get_diff_mm_scaled_flux_curve_angle_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_curve_angle_stats', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_abs_curve_angle_stats_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_abs_curve_angle_stats', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_abs_curve_angle_stats_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_abs_curve_angle_stats', 'train', path='../features/')
    
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()
    