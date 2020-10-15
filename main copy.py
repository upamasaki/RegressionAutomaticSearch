import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###########################
# model import
#
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor


models = [linear_model.LinearRegression(), 
          DecisionTreeRegressor(max_depth=2)]
names = ["LinearRegression", 
         "DecisionTreeRegressor"]


###########################
# init
#
save_dir = 'result'

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



for i, (model, model_name) in enumerate(zip(models, names)):

    # 予測モデルを作成
    model.fit(X, Y)

    # X_pred = np.dot(X, model.coef_)
    X_pred = model.predict(X)
    
    for i in range(20):
        sum = 0
        for x, c in zip(X[i, :], model.coef_):
            # print("x:{}, c:{}, x*c:{}".format(x, c, x*c))
            sum += x*c
        print("X_pred:{}, target:{}".format(sum + model.intercept_, Y[i]))


    print("X : {}".format(X.shape))
    print("model.coef_ : {}".format(model.coef_.shape))
    print("X : {}".format(X[0, :]))
    print("model.coef_ : {}".format(model.coef_))
    print("X_pred")
    print(X_pred + model.intercept_)
    print("sum")
    print(np.sum((X_pred + model.intercept_) - Y))
    print(X_pred.shape)
    # 偏回帰係数
    print(pd.DataFrame({"Name":boston_X.columns,
                        "Coefficients":model.coef_}).sort_values(by='Coefficients') )
    
    # 切片 (誤差)
    print(model.intercept_)

    ###########################
    # figure section
    #
    fig = plt.figure()
    #
    plt.scatter(X_pred + model.intercept_, Y,  color='black')
    plt.plot([np.min([Y]), np.max([Y])], [np.min([Y]), np.max([Y])], color='blue', linewidth=3)
    #
    plt.title(model_name)
    plt.xlabel('X_pred', fontsize=14)
    plt.ylabel('target', fontsize=14)
    # plt.show()
    save_path = save_dir + '/' + model_name + '_loss.png'
    print(save_path)
    fig.savefig(save_path)