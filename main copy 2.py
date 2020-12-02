import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from module.ModelFactory import ModelFactory



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
        train_R2  = r2_score(train_X_pred, train_Y) 
        result_box.loc[i, 'train_MSE'] = train_MSE
        result_box.loc[i, 'train_R2'] = train_R2

        test_X_pred = model.predict(test_X)
        test_MSE = np.mean((test_X_pred - test_Y)**2)
        test_R2  = r2_score(test_X_pred, test_Y) 
        result_box.loc[i, 'test_MSE'] = test_MSE
        result_box.loc[i, 'test_R2'] = test_R2

        

        ###########################
        # figure section
        #
        # fig = plt.figure()
        # #
        # plt.scatter(X_pred, Y,  color='black')
        # plt.plot([np.min([Y]), np.max([Y])], [np.min([Y]), np.max([Y])], color='blue', linewidth=3)
        # #
        # plt.title(model_name + ' MSE:{:6.3f}, R2:{:6.3f}'.format(MSE, R2))
        # plt.xlabel('X_pred', fontsize=14)
        # plt.ylabel('target', fontsize=14)
        # # plt.show()
        # save_path = save_dir + '/img/' + model_name + '_loss.png'
        # # print(save_path)
        # fig.savefig(save_path)
        # plt.close(fig)
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
    
    result_box.to_csv(save_dir + '/single_result_box/result_single_{}.csv'.format(model_name))

    if(len(result_box_) > 0):
        result_box_ = pd.concat([result_box_, result_box], axis=0)
    else:
        result_box_ = result_box.copy()
    

result_box_.to_csv(save_dir + '/result_box_.csv')
des_box.to_csv(save_dir + '/des_box.csv')