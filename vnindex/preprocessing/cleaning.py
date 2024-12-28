import pandas as pd

class Cleaner():
  def clean_nan(df: pd.DataFrame, in_cols = []):
    # drop NaN in rows
    df.dropna(axis=0, thresh=int())
    

    