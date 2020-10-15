import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score

###########################
# model import
#
from module.ModelFactory import ModelFactory
model_factory = ModelFactory()
models, names = model_factory.model_import()
print("検証パターン : {}".format(len(names)))

###########################
# init
#
save_dir = 'result'
result_box = pd.DataFrame(names, columns=['model_name'])
#

###########################
# read datasets
#
# 一番左にindexがある場合
df = pd.read_csv('./datasets/boston_datasets.csv', index_col=0)
#
# 一番左にindexがない場合
# df = pd.read_csv('./datasets/boston_datasets.csv')
#
# datasetsの確認
print(df.head())
#
# 説明変数に "MONEY 以外すべて" を利用
boston_X = df.drop("MONEY", axis=1)
X = boston_X.values
#
# 目的変数に "MONEY" を利用
Y = df['MONEY'].values


###########################
# estimate section
#
for i in tqdm(range(len(names))):

    model = models[i]
    model_name = names[i]

    # 予測モデルを作成
    model.fit(X, Y)

    X_pred = model.predict(X)
    MSE = np.mean((X_pred - Y)**2)
    R2  = r2_score(X_pred, Y) 
    result_box.loc[i, 'MSE'] = MSE
    result_box.loc[i, 'R2'] = R2
    result_box.to_csv(save_dir + '/result.csv')

    ###########################
    # figure section
    #
    fig = plt.figure()
    #
    plt.scatter(X_pred, Y,  color='black')
    plt.plot([np.min([Y]), np.max([Y])], [np.min([Y]), np.max([Y])], color='blue', linewidth=3)
    #
    plt.title(model_name + ' MSE:{:6.3f}, R2:{:6.3f}'.format(MSE, R2))
    plt.xlabel('X_pred', fontsize=14)
    plt.ylabel('target', fontsize=14)
    # plt.show()
    save_path = save_dir + '/img/' + model_name + '_loss.png'
    # print(save_path)
    fig.savefig(save_path)
    plt.close(fig)