import pandas as pd

class Cleaner():
  def clean_nan(df: pd.DataFrame, in_cols = []):
    # drop NaN in rows
    df.dropna(axis=0, thresh=int())
    
 
import pandas as pd
from normalize import Normalize

df = pd.read_csv('res/dataset.csv')

df['DATE'] = df['DATE'].apply(lambda x: Normalize.normalize_date(x, format='%d/%m/%Y'))

columns_to_normalize = ['PRICE', 'OPEN', 'HIGH', 'LOW', 'VOL', 'CHANGE']
for column in columns_to_normalize:
    df[column] = df[column].apply(lambda x: Normalize.normalize_number(str(x)))
print(f"Num rows with NaN: {df.isna().any(axis=1).sum()}")
print(f"Rows with NaN: {df[df.isna().any(axis=1)]}")

import torch
torch.concat()

