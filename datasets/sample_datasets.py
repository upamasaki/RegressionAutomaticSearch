from sklearn.datasets import load_boston
boston = load_boston()

import pandas as pd

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
print(df_x.head())


df_y = pd.DataFrame(boston.target, columns=['MONEY'])
print(df_y.head())

df = pd.concat([df_x, df_y], axis=1)

print(df.head())
df.to_csv("boston_datasets.csv")