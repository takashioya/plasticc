from utils import * 
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from utils import * 

def main():
    tr = feather.read_dataframe('../others/tr.feather')
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    tes = feather.read_dataframe('../others/tes.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    
    inv_cols = ['hostgal_photoz_err', 'distmod', 'mwebv']
    nyanp_f_all = feather.read_dataframe('../buckets/features_nyanp_all_v1_train.f').drop(['object_id'], axis = 1)
    nyanp_f_all = nyanp_f_all.rename(columns = lambda el: el.replace('xxx_', ''))
    nyanp_f = feather.read_dataframe('../buckets/nyanp_feat_v1_train.f').drop(['object_id', 'target'], axis = 1)
    needed_columns = np.array(nyanp_f.columns)[~np.in1d(nyanp_f.columns, nyanp_f_all.columns)]
    f_tr = pd.concat([nyanp_f_all, nyanp_f[needed_columns]], axis = 1).drop(inv_cols, axis = 1).astype(np.float32)
    f_tr.columns = [el.replace('/', '_') for el in f_tr.columns]

    inv_cols = ['hostgal_photoz_err', 'distmod', 'mwebv']
    nyanp_f_all = feather.read_dataframe('../buckets/features_nyanp_all_v1_test.f').drop(['object_id'], axis = 1)
    nyanp_f_all = nyanp_f_all.rename(columns = lambda el: el.replace('xxx_', ''))
    nyanp_f = feather.read_dataframe('../buckets/nyanp_feat_v1_test.f').drop(['object_id', 'target'], axis = 1)
    needed_columns = np.array(nyanp_f.columns)[~np.in1d(nyanp_f.columns, nyanp_f_all.columns)]
    f_tes = pd.concat([nyanp_f_all, nyanp_f[needed_columns]], axis = 1).drop(inv_cols, axis = 1).astype(np.float32)
    f_tes.columns = [el.replace('/', '_') for el in f_tes.columns]

    save_df_as_npy(f_tr, 'nyanp', 'nyanp_f', 'train', path='../features/')
    save_df_as_npy(f_tes, 'nyanp', 'nyanp_f', 'test', path='../features/')

    f_tr = pd.merge(tr_m[['object_id']], feather.read_dataframe('../buckets/features_nyanp_all_v2_train.f'), how = 'left', on = 'object_id')\
    .drop(['object_id'], axis = 1)
    f_tes = pd.merge(tes_m[['object_id']], feather.read_dataframe('../buckets/features_nyanp_all_v2_test.f'), how = 'left', on = 'object_id')\
    .drop(['object_id'], axis = 1)

    save_df_as_npy(f_tr, 'nyanp', 'nyanp_f', 'train', path='../features/')
    save_df_as_npy(f_tes, 'nyanp', 'nyanp_f', 'test', path='../features/')

    f_tr = pd.merge(tr_m[['object_id']], feather.read_dataframe('../buckets/features_nyanp_all_v3_train.f'), how = 'left', on = 'object_id')\
    .drop(['object_id'], axis = 1)
    f_tes = pd.merge(tes_m[['object_id']], feather.read_dataframe('../buckets/features_nyanp_all_v3_test.f'), how = 'left', on = 'object_id')\
    .drop(['object_id'], axis = 1)

    save_df_as_npy(f_tr, 'nyanp', 'nyanp_f', 'train', path='../features/')
    save_df_as_npy(f_tes, 'nyanp', 'nyanp_f', 'test', path='../features/')

    f_tr = feather.read_dataframe('../buckets/features_nyanp_all_v4_train.f').drop(['object_id'], axis = 1)
    f_tes = feather.read_dataframe('../buckets/features_nyanp_all_v4_test.f').drop(['object_id'], axis = 1)

    save_df_as_npy(f_tr, 'nyanp', 'nyanp_f', 'train', path='../features/')
    save_df_as_npy(f_tes, 'nyanp', 'nyanp_f', 'test', path='../features/')
    
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()