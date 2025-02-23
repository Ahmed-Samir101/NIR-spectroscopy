"""
    NOTE:
        The provided datasets on Github are already processed.
        ONLY apply this code on the original datasets.
"""
import pandas as pd

df1 = pd.read_csv('datasets/train.csv')
df2 = pd.read_csv('datasets/test.csv')

print(df1.head())

df1.drop(columns=['序号'], inplace=True)
df2.drop(columns=['序号'], inplace=True)

df1.to_csv('datasets/train.csv', index=False)
df2.to_csv('datasets/test.csv', index=False)
print("Done")
