import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import os
import glob
import pprint

from module.ModelFactory import ModelFactory


class RegressionAutomaticSearch:
    def __init__(self):
        ###########################
        # model import
        #
        model_factory = ModelFactory()
        self.models, self.names = model_factory.model_import()
        print(">>>> 検証パターン : {}".format(len(self.names)))

        ###########################
        # init
        #
        self.save_dir = 'result'
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_dir_single = 'result/single'
        os.makedirs(self.save_dir_single, exist_ok=True)
        self.result_box_ = pd.DataFrame(self.names, columns=['model_name'])
        self.des_box = pd.DataFrame({})

        ###########################
        # read datasets
        #
        # 一番左にindexがある場合
        self.df = pd.read_csv('./datasets/boston_datasets.csv', index_col=0)
        #
        # 一番左にindexがない場合
        # df = pd.read_csv('./datasets/boston_datasets.csv')
        #
        # datasetsの確認
        print(self.df.head())
        #
        self.target_col_name = 'MONEY'
        self.kf = KFold(n_splits=5)

    
    def kesson_table(self, df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns



    def preprocess_titanic(self):
        train= pd.read_csv("./datasets/train.csv")
        test= pd.read_csv("./datasets/test.csv")
        print(train.head())

        print(self.kesson_table(train))
        print(self.kesson_table(test))

        train["Age"] = train["Age"].fillna(train["Age"].mean())
        train["Embarked"] = train["Embarked"].fillna("S")
        train["Sex"][train["Sex"] == "male"] = 0
        train["Sex"][train["Sex"] == "female"] = 1
        train["Embarked"][train["Embarked"] == "S" ] = 0
        train["Embarked"][train["Embarked"] == "C" ] = 1
        train["Embarked"][train["Embarked"] == "Q"] = 2
        train.head(10)

        test["Age"] = test["Age"].fillna(test["Age"].mean())
        test["Sex"][test["Sex"] == "male"] = 0
        test["Sex"][test["Sex"] == "female"] = 1
        test["Embarked"][test["Embarked"] == "S"] = 0
        test["Embarked"][test["Embarked"] == "C"] = 1
        test["Embarked"][test["Embarked"] == "Q"] = 2
        test.Fare[152] = test.Fare.median()
        test.head(10)

        print(self.kesson_table(train))
        print(self.kesson_table(test))


    def estimate(self): 
        ###########################
        # estimate section
        #
        i = 0
        for imodel in tqdm(range(len(self.names))):

            model = self.models[imodel]
            model_name = self.names[imodel]
            # result_box  = pd.DataFrame(names, columns=['model_name'])
            result_box  = pd.DataFrame({})

            for idx, (train_idx, test_idx) in enumerate(self.kf.split(self.df)):
                # print(idx, train_idx, test_idx)
                result_box.loc[i, 'idx'] = idx
                result_box.loc[i, 'model_name'] = model_name

                train_df = self.df[self.df.index.isin(train_idx)]
                test_df = self.df[self.df.index.isin(test_idx)]

                train_X = train_df.drop(self.target_col_name, axis=1).values
                train_Y = train_df[self.target_col_name].values
                test_X  = test_df.drop(self.target_col_name, axis=1).values
                test_Y  = test_df[self.target_col_name].values

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
            result_box_des.to_csv(self.save_dir_single + '/{}.csv'.format(model_name))

            # print(result_box_des)
            self.des_box.loc[imodel, 'model_name'] = model_name
            # print(des_box)
            # print()
            for icol in result_box_des.columns[1:]:
                for iind in result_box_des.index:
                    self.des_box.loc[imodel, icol + '_' + iind] = result_box_des[icol][iind]
            
            # result_box.to_csv(self.save_dir_single + '/{}.csv'.format(model_name))

            if(len(self.result_box_) > 0):
                self.result_box_ = pd.concat([self.result_box_, result_box], axis=0)
            else:
                self.result_box_ = result_box.copy()

    def result(self): 
        des_csv_list = glob.glob(self.save_dir_single + '/*.csv')
        # pprint.pprint(des_csv_list)

        for ipath, csv_path in enumerate(tqdm(des_csv_list)):
            csv_path = csv_path.replace('\\', '/')
            result_box_des = pd.read_csv(csv_path, index_col=0)

            self.des_box.loc[ipath, 'model_name'] = csv_path.split('/')[-1].split('.')[0]
            # print(des_box)
            # print()
            for icol in result_box_des.columns[1:]:
                for iind in result_box_des.index:
                    # print(icol, iind)
                    self.des_box.loc[ipath, icol + '_' + iind] = result_box_des[icol][iind]
            
        self.des_box.to_csv(self.save_dir + '/des_box2.csv')

if __name__ == "__main__":
    RAS = RegressionAutomaticSearch()
    RAS.preprocess_titanic()
    # RAS.estimate()
    # RAS.result()