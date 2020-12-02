import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from module.ModelFactory import ModelFactory


###########################
# init
#
save_dir = 'result/single_result_box3'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir + '/img', exist_ok=True)
feature_box_ = pd.DataFrame({}, columns=['feature_name'])


###########################
# read datasets
#
# 一番左にindexがある場合
df = pd.read_csv(r'result\single_result_box3\des_box2.csv', index_col=0)
print(df.head())



fname_list = []

for fname in df.columns:
    if 'mean' in fname:
        fname_list.append(fname)
        feature_box_.loc[fname, 'feature_sum'] = np.sum(df[fname].values)
        feature_box_.loc[fname, 'feature_maen'] = np.mean(df[fname].values)

print(fname_list)

print(feature_box_.head())
feature_box_.to_csv(save_dir + '/feature_box_.csv')