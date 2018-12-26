import pandas as pd
import numpy as np
from utils import * 


def main():
    initialize_feature_name_tree('../others/fn_tree.pkl')
    
    tr = pd.read_csv('../data/training_set.csv.zip')
    tes = pd.read_csv('../data/test_set.csv.zip')
    tr_m = pd.read_csv('../data/training_set_metadata.csv')
    tes_m = pd.read_csv('../data/test_set_metadata.csv.zip')

    ts_convert_dict = {'detected': np.uint8, 'passband': np.uint8, 'object_id': np.int32, 'mjd': np.float64,
                       'flux': np.float32, 'flux_err': np.float32}
    tes = tes.astype(ts_convert_dict)
    tes.to_feather('../others/tes.feather')

    tr = tr.astype(ts_convert_dict)
    tr.to_feather('../others/tr.feather')

    meta_convert_dict  = {'object_id': np.int32, 'ra': np.float32, 'decl': np.float32, 'gal_l': np.float32,
                          'gal_b': np.float32, 'ddf': np.uint8, 'hostgal_specz': np.float32,
                          'hostgal_photoz': np.float32, 'hostgal_photoz_err': np.float32, 'distmod': np.float32,
                          'mwebv': np.float32}
    tr_m = tr_m.astype(meta_convert_dict)
    tr_m = tr_m.astype({'target': np.int32})
    tes_m = tes_m.astype(meta_convert_dict)
    tr_m.to_feather('../others/tr_m.feather')
    tes_m.to_feather('../others/tes_m.feather')
           
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()