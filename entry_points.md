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
