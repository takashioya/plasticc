import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_05 import * 
from utils import * 


def main():
    tes = feather.read_dataframe('../others/tes.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    
    f_tes = get_feets(tes_m, tes)
    save_df_as_npy(f_tes, 'ts', 'feets', 'test', path='../features/')
    
    f_tes = get_gp_fitted(tes_m, tes) # will take one month
    save_df_as_npy(f_tes, 'ts', 'gp_fitted', 'test', path='../features/')
    
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()
