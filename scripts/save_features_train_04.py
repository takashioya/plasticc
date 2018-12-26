import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_04 import * 
from utils import * 


def main():
    tr = feather.read_dataframe('../others/tr.feather')
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    
    f_tr = get_mjd_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mjd_skew_kurt', 'train', path='../features/')

    f_tr = get_mjd_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mjd_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_mjd_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mjd_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_mjd_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mjd_skew_kurt', 'train', path='../features/')

    f_tr = get_flux_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_skew_kurt', 'train', path='../features/')

    f_tr = get_flux_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_flux_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_flux_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_flux_err_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_err_skew_kurt', 'train', path='../features/')

    f_tr = get_flux_err_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'flux_err_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_flux_err_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_err_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_flux_err_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_flux_err_skew_kurt', 'train', path='../features/')

    f_tr = get_curve_angle_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'curve_angle_skew_kurt', 'train', path='../features/')

    f_tr = get_curve_angle_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'curve_angle_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_curve_angle_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_curve_angle_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_curve_angle_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_curve_angle_skew_kurt', 'train', path='../features/')

    f_tr = get_abs_curve_angle_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'abs_curve_angle_skew_kurt', 'train', path='../features/')

    f_tr = get_abs_curve_angle_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'abs_curve_angle_skew_kurt', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_flux_n_sigma_skew_kurt_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'flux_n_sigma_skew_kurt', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_flux_n_sigma_skew_kurt_detected_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'flux_n_sigma_skew_kurt', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_diff_flux_n_sigma_skew_kurt_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'diff_flux_n_sigma_skew_kurt', 'train', path='../features/')

    for n in tqdm([-3, -2, -1, 1, 2, 3]):
        f_tr = get_diff_flux_n_sigma_skew_kurt_detected_passband(tr_m, tr, n)
        save_df_as_npy(f_tr, 'ts', 'diff_flux_n_sigma_skew_kurt', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_mm_scaled_flux_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'mm_scaled_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_mm_scaled_flux_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_mm_scaled_flux_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_mm_scaled_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_dist_squared_shifted_flux_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'dist_squared_shifted_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_dist_squared_shifted_flux_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'dist_squared_shifted_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_dist_squared_shifted_flux_skew_kurt_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_dist_squared_shifted_flux_skew_kurt', 'train', path='../features/')

    f_tr = get_diff_dist_squared_shifted_flux_skew_kurt_detected_passband(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'diff_dist_squared_shifted_flux_skew_kurt', 'train', path='../features/')
    
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()
