from utils import *
import numpy as np
import feather
from sklearn.preprocessing import LabelEncoder

def main():
    tr_m = feather.read_dataframe('../others/tr_m.feather')

    np.save('../others/train_target.npy', tr_m['target'].values)
    y_orig = np.load('../others/train_target.npy')
    le = LabelEncoder()
    le.fit(y_orig)
    save_pickle(le, '../others/label_encoder.pkl')

    np.save('../others/distmod_mask.npy', (~tr_m['distmod'].isnull()).values)
           
    print('===== Process sucessfuly finished =====')


if __name__ == '__main__':
    main()
