import pandas as pd
from vnindex.preprocessing.normalize import Normalize
from sklearn.preprocessing import StandardScaler

def read_data(file_path: str):
  """
      Use for read data from file
      input: file name
      output: dataframe
  """
  df = pd.read_csv(file_path)
  # Lazy data handling, need to be removed in real competition
  df = df.dropna()
  # Reverse data from bottom to top (from oldest to latest for more accurate prediction)
  df = df.iloc[::-1].reset_index(drop=True)
  return df

def execute(data):
  """
      Use for execute preprocessing
  """
  new_data = {'DATE': [], 'PRICE': [], 'OPEN': [], 'HIGH': [], 'LOW': [], 'VOL': [], 'CHANGE': []}
  for col in data:
    for row in data[col]:
      if row != '':
        row = str(row)
        if col == 'DATE':
          new_data[col].append(Normalize.normalize_date(row))
        else:
          new_data[col].append(Normalize.normalize_number(row))

  data = pd.DataFrame(new_data)

  return data