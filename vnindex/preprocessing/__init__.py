import pandas as pd
from vnindex.preprocessing.normalize import Normalize
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    """
        This class for preprocessing, read data and normalize data from file_path dataset
    """
    def __init__(self, file_path: str, train_file=True, scaler=None):
        self.data = self.read_data(file_path)
        if train_file:
            self.execute()
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
            self.data = pd.DataFrame(self.data,  columns=["DATE", "PRICE", "OPEN", 'HIGH', 'LOW', 'VOL', 'CHANGE'])
        else:
            self.execute2()
            self.scaler = scaler
            self.data = self.scaler.fit(self.data)
            self.data = pd.DataFrame(self.data, columns=['DATE'])

    def read_data(self, file_path: str):
        """
            Use for read data from file
            input: file name
            output: dataframe
        """
        df = pd.read_csv(file_path)
        # Lazy data handling, need to be removed in real competition
        df = df.dropna()
        # Reverse data from bottom to top (from oldest to latest for more accurate prediction)
        # df = df.iloc[::-1].reset_index(drop=True)
        return df

    def execute(self):
        """
            Use for execute preprocessing
        """
        new_data = {'Date': [], 'Price': [], 'Open': [], 'High': [], 'Low': [], 'Vol.': [], 'Change %': []}
        for col in self.data:
            for row in self.data[col]:
                if row != '':
                    row = str(row)

                    if col == 'Date':
                        new_data[col].append(Normalize.normalize_date(row))
                    else:
                        new_data[col].append(Normalize.normalize_number(row))

        self.data = pd.DataFrame(new_data)

    def execute2(self):
        """
            Use for execute preprocessing
        """
        new_data = {'Date': []}
        for col in self.data:
            for row in self.data[col]:
                if row != '':
                    row = str(row)
                    new_data[col].append(Normalize.normalize_date(row))

        self.data = pd.DataFrame(new_data)
