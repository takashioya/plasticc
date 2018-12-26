import gc
from tqdm import tqdm
import feather
import sys
sys.path.append('..')
from fe_extract.make_features_06 import * 
from utils import * 
 
def main():
    get_and_save_spectrum_features('test')
    get_and_save_spectrum_features_nyanp('test')
   
    print('===== Process sucessfuly finished =====')

if __name__ == '__main__':
    main()
    