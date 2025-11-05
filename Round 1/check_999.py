import pandas as pd
import numpy as np
df = pd.read_csv('data/main.csv')
print(f'Dataset shape: {df.shape}\n')
numeric_cols = df.select_dtypes(include=[np.number]).columns
rows_with_999 = df[(df[numeric_cols] == 999).any(axis=1)]
print(f'Rows containing 999: {len(rows_with_999)} ({len(rows_with_999) / len(df) * 100:.2f}%)')
print(f'\nLabel distribution in rows with 999:')
print(rows_with_999['Label'].value_counts())
print(f'\nColumns with 999 values:')
counts_999 = (df[numeric_cols] == 999).sum()
print(counts_999[counts_999 > 0].sort_values(ascending=False))
print(f'\nSample row with 999 values:')
sample = rows_with_999.iloc[0]
cols_with_999 = [col for col in numeric_cols if sample[col] == 999]
print(f'Columns with 999: {cols_with_999}')
print(f'\nFirst row with 999:')
print(rows_with_999.iloc[0][['Label', 'sha256'] + cols_with_999[:5]].to_string())
