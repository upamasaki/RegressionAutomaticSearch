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
# model import
#
model_factory = ModelFactory()
models, names = model_factory.model_import()
print("検証パターン : {}".format(len(names)))

###########################
# init
#
save_dir = 'result/single_result_box3'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir + '/img', exist_ok=True)
result_box_ = pd.DataFrame(names, columns=['model_name'])
des_box = pd.DataFrame({})
#

###########################
# read datasets
#
# 一番左にindexがある場合
df = pd.read_csv('./datasets/boston_datasets_norm.csv', index_col=0)

# df_norm = ((df - df.min()) / (df.max() - df.min()))*2 - 1
# df_norm.to_csv('./datasets/boston_datasets_norm.csv')
# raise

#
# 一番左にindexがない場合
# df = pd.read_csv('./datasets/boston_datasets.csv')
#
# datasetsの確認
print(df.head())
#
target_col_name = 'MONEY'
# 説明変数に "MONEY 以外すべて" を利用
boston_X = df.drop(target_col_name, axis=1)
X = boston_X.values
#
# 目的変数に "MONEY" を利用
Y = df[target_col_name].values

kf = KFold(n_splits=5)
# kf = StratifiedKFold(n_splits=5)

###########################
# estimate section
#
i = 0
for imodel in tqdm(range(len(names))):

    model = models[imodel]
    model_name = names[imodel]
    # result_box  = pd.DataFrame(names, columns=['model_name'])
    result_box  = pd.DataFrame({})

    for idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        # print(idx, train_idx, test_idx)
        result_box.loc[i, 'idx'] = idx
        result_box.loc[i, 'model_name'] = model_name

        train_df = df[df.index.isin(train_idx)]
        test_df = df[df.index.isin(test_idx)]

        train_X = train_df.drop(target_col_name, axis=1).values
        train_Y = train_df[target_col_name].values
        test_X  = test_df.drop(target_col_name, axis=1).values
        test_Y  = test_df[target_col_name].values

        # 予測モデルを作成
        model.fit(train_X, train_Y)

        train_X_pred = model.predict(train_X)
        train_MSE = np.mean((train_X_pred - train_Y)**2)
        train_RMSE = np.sqrt(train_MSE)
        train_R2  = r2_score(train_X_pred, train_Y) 
        result_box.loc[i, 'train_MSE'] = train_MSE
        result_box.loc[i, 'train_RMSE'] = train_RMSE
        result_box.loc[i, 'train_R2'] = train_R2

        test_X_pred = model.predict(test_X)
        test_MSE = np.mean((test_X_pred - test_Y)**2)
        test_RMSE = np.sqrt(test_MSE)
        test_R2  = r2_score(test_X_pred, test_Y) 
        result_box.loc[i, 'test_MSE'] = test_MSE
        result_box.loc[i, 'test_RMSE'] = test_RMSE
        result_box.loc[i, 'test_R2'] = test_R2

        feature_importances_ = [0 for _ in range(13)]
        if hasattr(model, 'feature_importances_'):
            # print("model : {}".format(model))
            feature_importances_ = model.feature_importances_
            # print("feature_importances_ : {}".format(len(feature_importances_)))  
            # print(feature_importances_)

        for col, value in zip(df.columns, feature_importances_):
            result_box.loc[i, col] = value

        # print(result_box.head())
        # raise

        

        ###########################
        # figure section
        #
        fig = plt.figure()
        #
        plt.scatter(train_X_pred, train_Y,  color='black')
        plt.plot([np.min([train_Y]), np.max([train_Y])], [np.min([train_Y]), np.max([train_Y])], color='blue', linewidth=3)
        #
        plt.title(model_name + '\nMSE:{:6.3f}, RMSE:{:6.3f}, R2:{:6.3f}'.format(train_MSE, train_RMSE, train_R2))
        plt.xlabel('X_pred', fontsize=14)
        plt.ylabel('target', fontsize=14)
        # plt.show()
        save_path = save_dir + '/img/' + model_name + '_idx{:04d}_loss_train.png'.format(idx)
        # print(save_path)
        fig.savefig(save_path)
        plt.close(fig)
        #
        #-----------------------------------------
        #
        fig = plt.figure()
        #
        plt.scatter(test_X_pred, test_Y,  color='black')
        plt.plot([np.min([test_Y]), np.max([test_Y])], [np.min([test_Y]), np.max([test_Y])], color='blue', linewidth=3)
        #
        plt.title(model_name + '\nMSE:{:6.3f}, RMSE:{:6.3f}, R2:{:6.3f}'.format(test_MSE, test_RMSE, test_R2))
        plt.xlabel('X_pred', fontsize=14)
        plt.ylabel('target', fontsize=14)
        # plt.show()
        save_path = save_dir + '/img/' + model_name + '_idx{:04d}_loss_test.png'.format(idx)
        # print(save_path)
        fig.savefig(save_path)
        plt.close(fig)


        i += 1

    # print(result_box)
    result_box_des = result_box.describe()
    # print(result_box_des)
    des_box.loc[imodel, 'model_name'] = model_name
    # print(des_box)
    # print()
    for icol in result_box_des.columns[1:]:
        for iind in result_box_des.index:
            des_box.loc[imodel, icol + '_' + iind] = result_box_des[icol][iind]
    
    result_box.to_csv(save_dir + '/result_single_{}.csv'.format(model_name))

    if(len(result_box_) > 0):
        result_box_ = pd.concat([result_box_, result_box], axis=0)
    else:
        result_box_ = result_box.copy()
    


result_box_.to_csv(save_dir + '/result_box_.csv')
des_box.to_csv(save_dir + '/des_box2.csv')