import pandas as pd
from vnindex.preprocessing.normalize import Normalize
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    """
        This class for preprocessing, read data and normalize data from file_path dataset
    """
    def __init__(self, file_path: str):
        self.data = self.read_data(file_path)
        self.execute()
        self.scaler = StandardScaler(self.data)
        self.data = self.scaler.fit(self.data)
        self.data = pd.DataFrame(self.data)

    def read_data(self, file_path: str):
        """
            Use for read data from file
            input: file name
            output: dataframe
        """
        df = pd.read_csv(file_path)
        df = df.dropna()
        return df

    def execute(self):
        """
            Use for execute preprocessing
        """
        new_data = {'DATE': [], 'PRICE': [], 'OPEN': [], 'HIGH': [], 'LOW': [], 'VOL': [], 'CHANGE': []}
        for col in self.data:
            for row in self.data[col]:
                if row != '':
                    row = str(row)

                    if col == 'DATE':
                        new_data[col].append(Normalize.normalize_date(row))
                    else:
                        new_data[col].append(Normalize.normalize_number(row))

        self.data = pd.DataFrame(new_data)

        return self.data
