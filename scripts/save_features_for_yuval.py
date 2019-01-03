from utils import *
import numpy as np


def main():
    fn_s = np.load('../fi/mamas_feature_names_v1.npy')
    X = load_arr(fn_s, 'train')
    X_test = load_arr(fn_s, 'test')
    np.save('../features/mamas_feat_v1_train.npy', X)
    np.save('../features/mamas_feat_v1_test.npy', X_test)

    fn_s = np.load('../fi/mamas_feature_names_v2.npy')
    X = load_arr(fn_s, 'train')
    X_test = load_arr(fn_s, 'test')
    np.save('../features/mamas_feat_v2_train.npy', X)
    np.save('../features/mamas_feat_v2_test.npy', X_test)

    fn_s = np.load('../fi/mamas_feature_names_v3.npy')
    X = load_arr(fn_s, 'train')
    X_test = load_arr(fn_s, 'test')
    np.save('../features/mamas_feat_v3_train.npy', X)
    np.save('../features/mamas_feat_v3_test.npy', X_test)

if __name__ == '__main__':
    main()


