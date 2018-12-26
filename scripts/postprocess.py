import sys
sys.path.append('..')
from scripts.utils import * 
from tqdm import tqdm
import feather
import numpy as np
import pandas as pd

def make_sub_with_mamas_c99_handling(exp_name):
    data_types = ['ex_gal', 'ex_gal_spec', 'gal']
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    le = load_pickle('../others/label_encoder.pkl')
    W = np.load('../others/W.npy')
    ex_gal_index = ((tes_m['hostgal_specz'].isnull()) & (~tes_m['distmod'].isnull())).values
    ex_gal_spec_index = ((~tes_m['hostgal_specz'].isnull()) & (~tes_m['distmod'].isnull())).values
    gal_index = (tes_m['distmod'].isnull()).values
    ex_gal_pred = np.load('../preds/' + data_types[0] + '_pred_' + exp_name + '.npy')
    ex_gal_spec_pred = np.load('../preds/' + data_types[1] + '_pred_' + exp_name + '.npy')
    gal_pred = np.load('../preds/' + data_types[2] + '_pred_' + exp_name + '.npy')

    # prepare subs for pseudo-labelling
    sub = pd.read_csv('../data/sample_submission.csv.zip')
    cols = sub.columns[1:-1]
    sub_86_s = ['pred16r4qq_0.csv', 'pred16r4qq_1.csv', 'pred16r4qq_2.csv', 'pred16r4qq_3.csv']
    best_sub_dp = get_raw_from_yuval_sub(sub_86_s, np.array([0, 1, 2, 3]), cols)
    threshold = 0.91
    obj_class = 90
    pseudo_idx = best_sub_dp['class_' + str(obj_class)] > threshold

    new_pred = np.zeros((tes_m.shape[0], 14))
    new_pred[ex_gal_index] = ex_gal_pred
    new_pred[ex_gal_spec_index] = ex_gal_spec_pred
    new_pred[gal_index] = gal_pred
    new_pred = new_pred * (1 - W[0])

    for i in range(14):
        sub['class_' + str(le.inverse_transform([i])[0])] = new_pred[:, i]
    sub['class_99'] = W[0]

    sub = sub.astype({'object_id': np.int32})
    sub = sub.astype({el: np.float32 for el in sub.columns[1:]})

    sub_dropped = sub.drop(['object_id', 'class_99'], axis=1)
    sub_dropped = sub_dropped * 1/(1 - sub['class_99'].iloc[0])
    sub_dropped.loc[pseudo_idx] = best_sub_dp.loc[pseudo_idx]
    max_lst = sub_dropped.max(axis=1)

    coef_start = ((1 + W[0]) * W[0])/((1- np.array(max_lst)).mean())
    y_all = 1 - np.array(max_lst)

    scores = []
    for coef in tqdm(np.arange(coef_start, 0.60, 0.0001)):
        scores.append((np.mean((coef * y_all)/(1 + coef * y_all)) - W[0]) ** 2)

    coef_opt = np.arange(coef_start, 0.60, 0.0001)[np.argmin(scores)]
    assert(np.sqrt(np.min(scores)) < 1e-4)

    sub_dropped_cp = sub_dropped.copy()
    sub_dropped_cp['class_99'] = (1 - np.array(max_lst)) * coef_opt
    sub_dropped_cp = sub_dropped_cp.divide(sub_dropped_cp.sum(axis=1), axis=0)
    sub_dropped_cp['object_id'] = sub['object_id']
    sub_dropped_cp = sub_dropped_cp[sub.columns]

    sub_dropped_cp.to_csv('../sub/' + exp_name + '_re' + '.csv.gz', index=False, compression='gzip')


def make_mamas_averaged_model_with_nyanp_c99_handling():
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    
    exp_46_2 = get_raw_from_mamas_sub(pd.read_csv('../sub/exp_46_2_re.csv.gz'))
    exp_44_2 = get_raw_from_mamas_sub(pd.read_csv('../sub/exp_44_2_re.csv.gz'))
    exp_40_4 = get_raw_from_mamas_sub(pd.read_csv('../sub/exp_40_4_re.csv.gz'))
    exp_46_1 = get_raw_from_mamas_sub(pd.read_csv('../sub/exp_46_1_re.csv.gz'))
    exp_45_3 = get_raw_from_mamas_sub(pd.read_csv('../sub/exp_45_3_re.csv.gz'))
    exp_46_4 = get_raw_from_mamas_sub(pd.read_csv('../sub/exp_46_4_re.csv.gz'))
    exp_43_3 = get_raw_from_mamas_sub(pd.read_csv('../sub/exp_43_3_re.csv.gz'))
    
    raw_av = (exp_46_2 + exp_44_2 + exp_40_4 + exp_46_1 + exp_45_3 + exp_43_3 + exp_46_4)/7
    
    is_extra = ~tes_m['distmod'].isnull()
    cols = raw_av.columns
    raw_av['class_99'] = 1
    for c in cols:
        raw_av['class_99'] *= (1 - raw_av[c])
    raw_av['class_99'] *= 1.0 - ~is_extra * 0.8 
    raw_av['class_99'] *= 1.0 - is_extra * 0.1
    
    sub = pd.read_csv('../data/sample_submission.csv.zip')
    raw_av = raw_av.divide(raw_av.sum(axis=1), axis=0)
    raw_av['object_id'] = sub['object_id']
    sub = raw_av.copy()
    
    sub.to_csv('../sub/mamas_average_7_model_for_host.csv.gz', index=False, compression='gzip')


def make_team_clean_averaged_model():
    # I decided these weights using oof prediction with hyperopt.
    space = {'w3': 0.5, 'w4': 0.46, 'w12': 0.66, 'w13': 0.35000000000000003, 
             'w8': 0.79, 'w0': 0.7000000000000001, 'w10': 0.34, 'w7': 0.25, 'w6': 0.72,
             'w5': 0.55, 'w2': 0.8, 'w9': 0.59, 'w1': 0.8, 'w11': 0.38}
    coefs_gbdt = np.array([space['w' + str(i)] for i in range(14)])

    w_s = {'w3': 0.3, 'w4': 0.37, 'w12': 0.63, 'w13': 0.41000000000000003, 
           'w8': 0.63, 'w0': 0.62, 'w10': 0.35000000000000003, 'w7': 0.67, 'w6': 0.32,
           'w5': 0.6900000000000001, 'w2': 0.42, 'w9': 0.64, 'w1': 0.3, 'w11': 0.4}
    coefs = [w_s['w' + str(i)] for i in range(14)]

    tes_m = feather.read_dataframe('../others/tes_m.feather')
    sub = pd.read_csv('../data/sample_submission.csv.zip')
    cols = sub.columns[1:-1]

    sub_94_s = ['pred16r20avn_0.csv', 'pred16r20avn_1.csv', 'pred16r20avn_2.csv', 'pred16r20avn_3.csv']
    sub_105_s = ['pred16r4avnapl2_0.csv', 'pred16r4avnapl2_1.csv', 'pred16r4avnapl2_2.csv', 'pred16r4avnapl2_3.csv']
    sub_107_s = ['pred16r19avnapl2_0.csv', 'pred16r19avnapl2_1.csv', 'pred16r19avnapl2_2.csv', 'pred16r19avnapl2_3.csv']
    raw_94 = get_raw_from_yuval_sub(sub_94_s, np.array([0, 1, 2, 3]), cols)
    raw_105 = get_raw_from_yuval_sub(sub_105_s, np.array([0, 1, 2, 3]), cols)
    raw_107 = get_raw_from_yuval_sub(sub_107_s, np.array([0, 1, 2, 3]), cols)
    avg_3_yuval = (raw_94 + raw_105 + raw_107)/3

    avg_4_nyanp = get_raw_from_nyanp_sub(pd.read_csv('../sub/experiment57_59(th985)_61_62.csv'))

    avg_7_mamas = get_raw_from_mamas_sub(pd.read_csv('../sub/mamas_average_7_model_for_host.csv.gz'))

    gbdt_av = pd.DataFrame()
    for i in range(len(cols)):
        c = cols[i]
        gbdt_av[c] = avg_7_mamas[c] * coefs_gbdt[i] + avg_4_nyanp[c] * (1 - coefs_gbdt[i])
    gbdt_av = gbdt_av.divide(gbdt_av.sum(axis=1), axis=0)

    nn_av = avg_3_yuval

    raw_av = pd.DataFrame()
    for i in range(len(cols)):
        c = cols[i]
        raw_av[c] = gbdt_av[c] * coefs[i] + nn_av[c] * (1 - coefs[i])
    raw_av_base = raw_av.divide(raw_av.sum(axis=1), axis=0)

    cols = raw_av_base.columns
    weight_dict = {col:1 for col in cols}
    coef_ex_gal = 0.9
    coef_gal = 0.2

    raw_av = raw_av_base.copy()
    is_extra = ~tes_m['distmod'].isnull()
    raw_av['class_99'] = 1
    for c in cols:
        raw_av['class_99'] *= (1 - raw_av[c]) ** weight_dict[c]

    raw_av['class_99'] *= 1.0 - ~is_extra * (1 - coef_gal)  # coef for galactic
    raw_av['class_99'] *= 1.0 - is_extra * (1 - coef_ex_gal)

    raw_av = raw_av.divide(raw_av.sum(axis = 1), axis = 0)
    raw_av['object_id'] = sub['object_id']

    sub = raw_av.copy()

    sub.to_csv('../sub/clean_average_for_host.csv.gz', index = False, compression = 'gzip')


def make_host_sub_with_special_c99_handling():
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    sub = pd.read_csv('../data/sample_submission.csv.zip')

    raw_av_base = get_raw_from_mamas_sub(pd.read_csv('../sub/clean_average_for_host.csv.gz'))
    cols = raw_av_base.columns

    coef_ex_gal = 1.65
    coef_gal = 0.2

    raw_av = raw_av_base.copy()
    is_extra = ~tes_m['distmod'].isnull()
    raw_av['class_99'] = 1

    for c in cols:
        raw_av['class_99'] *= (1 - raw_av[c]) 

    c42 = raw_av.loc[is_extra, 'class_42']
    c52 = raw_av.loc[is_extra, 'class_52']
    c62 = raw_av.loc[is_extra, 'class_62']
    c95 = raw_av.loc[is_extra, 'class_95']

    raw_av.loc[is_extra, 'class_99'] = (c42 + c52 + c62 + c95) ** 2.5 * (1 - c95 - c62) ** 0.635
    raw_av['class_99'] *= 1.0 - ~is_extra * (1 - coef_gal)
    raw_av['class_99'] *= 1.0 - is_extra * (1 - coef_ex_gal)
    raw_av = raw_av.divide(raw_av.sum(axis=1), axis=0)
    raw_av['object_id'] = sub['object_id']
    sub = raw_av.copy()

    assert(np.all(raw_av.min() >= 0))
    assert(np.all(np.isclose(sub.drop(['object_id'], axis=1).sum(axis=1), 1)))

    sub.to_csv('../sub/host_sub.csv.gz', index=False, compression='gzip')


def main():
    exp_names = ['exp_46_2', 'exp_44_2', 'exp_40_4', 'exp_46_1', 'exp_45_3', 'exp_46_4', 'exp_43_3']
    for exp_name in tqdm(exp_names):
        make_sub_with_mamas_c99_handling(exp_name)
        
    make_mamas_averaged_model_with_nyanp_c99_handling()
    
    make_team_clean_averaged_model()
    
    make_host_sub_with_special_c99_handling()
            
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()
