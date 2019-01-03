# PLAsTiCC Astronomical Classification 3rd-place solution

# Overview of solution
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75131

# environment
I used pyenv virtualenv to set up the environment.
I think `catboost==0.10.4.1` is important, but the versions of the other libraries won't affect the score so much.
```
$ pyenv install 3.5.1
$ pyenv virtualenv 3.5.1 plasticc
$ pyenv activate plasticc
$ pip install --upgrade pip
$ pip install cython==0.27.3
$ pip install numpy==1.13.0
$ pip install PyYAML==3.12
$ pip install -r requirements.txt
```
I used n1-standard-64 in Google Cloud Engine, which has 240GB RAM and 64 CPUs. <br>
OS/Platform : Ubuntu 16.04 <br>


# datasets & result files
I will upload prepare.zip for the host, which contains these directories. <br>
- `buckets`: <br>
It contains nyanp's train & test features.

- `data`: <br>
It contains kaggle datasets.
you can also download them via
```
kaggle competitions download -c PLAsTiCC-2018
```

- `features`: <br>
It contains all of my train & test features.

- `fi`: <br>
It contains feature names and the number of rounds used for training.
    - `exp_*.npy`<br>
    numpy array that contains feature names.
    - `exp_*rounds.pkl`<br>
    pickle object that contains the number of rounds.
    - `whole_fn_s.npy`<br>
    numpy array that contains all feature names.
    - `mamas_feature_names_v1.npy`<br>
    the names of features that yuval used.


- `models`: <br>
It contains trained models.
    - `exp*.cbm`<br>
    trained catboost model.

- `others`: <br>
It contains class weights.
    - `W.npy`<br>
    numpy array that contains class weights.

- `sub`: <br>
It contains submission files
    - `experiment57_59(th985)_61_62.csv`<br>
    nyanp's averaged submission file.
    - `pred*.csv`<br>
    yuval's submission file.

# scripts
- `utils.py` : <br>
It contains utility functions.
- `preprocess_*.py` : <br>
I did easy preprocessing here, like converting .csv files into .feather files.
- `save_features_train_*.py` : <br>
I saved test features here.
- `save_features_test_*.py` : <br>
I saved train features here.
- `save_features_nyanp.py` : <br>
I saved nyanp's train & test features here.
- `train.py` : <br>
I trained models here.
- `predict.py` : <br>
I made predictions here.
- `postprocess.py` : <br>
I did postprocessing like ensembling and class99 handling here.

# usage
- `full version` : <br>
It will take a few months to run with a single machine (64 core, 240GB RAM). <br>
I never recommend you to run it.
```
cd mamas/
unzip prepare.zip
cp -r prepare/* .
rm features/*
rm models/*
cd ../scripts
python preprocess_01.py
python preprocess_02.py
python save_features_train_01.py
python save_features_train_02.py
python save_features_train_03.py
python save_features_train_04.py
python save_features_train_05.py
python save_features_train_06.py
python save_features_test_01.py
python save_features_test_02.py
python save_features_test_03.py
python save_features_test_04.py
python save_features_test_05.py
python save_features_test_06.py
python save_features_nyanp.py
python save_features_for_yuval.py
python train.py
python predict.py
python postprocess.py
```
Then, mamas/sub/host_sub.csv.gz will be generated. <br>

- `short version` : <br>
It's a short version, which will take about 4 hours.
I use extracted features and trained model here.
```
cd mamas/
unzip prepare.zip
cp -r prepare/* .
cd scripts
python preprocess_01.py
python preprocess_02.py
python predict.py
python postprocess.py
```
Then, mamas/sub/host_sub.csv.gz will be generated. <br>
It should score 0.680 on public LB, 0.700 on private LB.

# directory
- `preds/` : <br>
prediction files.
- `curve/` : <br>
linear interpolated curve files with yuval's method.
- `fe_extract/` : <br>
feature extraction library.
- `notebook/`: <br>
It contains .ipynb files.
- `scripts/`:  <br>
It contains scripts.