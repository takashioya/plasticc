import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_05 import * 
from utils import * 

def main():
    tr = feather.read_dataframe('../others/tr.feather')
    tr_m = feather.read_dataframe('../others/tr_m.feather')
    tes = feather.read_dataframe('../others/tes.feather')
    tes_m = feather.read_dataframe('../others/tes_m.feather')
    
    f_tr = get_feets(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'feets', 'train', path='../features/')
    
    f_tr = get_gp_fitted(tr_m, tr)
    save_df_as_npy(f_tr, 'ts', 'gp_fitted', 'train', path='../features/')
    
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()