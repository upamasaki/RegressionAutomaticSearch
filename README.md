# RegressionAutomaticSearch

任意の機械学習でパラメータを変更したモデルで回帰分析を行うプログラムを作成しました．
これで最適なモデルとパラメータを見つけましょう．
今回はボストン市の住宅価格の予測を行ってみます．

## 仮想環境の作成

venvを作成
```
C:\RegressionAutomaticSearch>py -m venv venv
```
venvを適用
```
C:\RegressionAutomaticSearch>.\venv\Scripts\activate.bat
(venv) C:\RegressionAutomaticSearch>
```

## パッケージのインストール
パッケージをアップデート
```
(venv)C:\RegressionAutomaticSearch>python -m pip install --upgrade pip
```


必要なパッケージを一括インストール
```
(venv) C:\RegressionAutomaticSearch>pip install -r requirements.txt
```

## データセットの差し替え

回帰を行いたいデータセットのパスに変更します．

```python
###########################
# read datasets
#
# 一番左にindexがある場合
df = pd.read_csv('./datasets/boston_datasets.csv', index_col=0)
```

中身はこんな感じです．説明変数と目的変数が同じファイルを読み込みます．
```
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MONEY
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98   24.0
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14   21.6
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03   34.7
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94   33.4
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33   36.2
```

## 目的変数の変更

説明変数をデータフレームからdropします．
```python
# 説明変数に "MONEY 以外すべて" を利用
boston_X = df.drop("MONEY", axis=1)
X = boston_X.values
```

## パラメータの調整
モデルに渡すリストでパラメータを調整します．

```python
    def model_import(self):
        models_names = [ self.LinearRegression(), 
                         self.DecisionTreeRegressor(list(range(2, 30, 2))),
                         self.RandomForestRegressor(list(range(2, 30, 2)), list(range(20, 200, 20))),
                         self.AdaBoostRegressor(list(range(20, 200, 20)))]
        models = []
        names  = []
        for model_, name_ in models_names:
            if isinstance(model_, list):
                models.extend(model_)
                names.extend(name_)
            else:
                models.append(model_)
                names.append(name_)
        return models, names 
```

## 回帰の実行

main.pyを実行します．

```
(venv) C:\RegressionAutomaticSearch>python main.py  
```

## 回帰結果

resultフォルダに結果の画像と誤差，決定係数のcsvが出力されます．

![代替テキスト](https://github.com/upamasaki/RegressionAutomaticSearch/blob/main/manual/fig_demo.PNG)
![代替テキスト](https://github.com/upamasaki/RegressionAutomaticSearch/blob/main/manual/result_MSE_R2.PNG)