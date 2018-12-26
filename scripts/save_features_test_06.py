import sys
sys.path.append('..')
from fe_extract.make_features_06 import *


def main():
    get_and_save_spectrum_features('test')
    get_and_save_spectrum_features_nyanp('test')
   
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()
