import pandas as pd
# from .vnindex.preprocessing.normalize import Normalize

class Preprocessing:
    """
        This class for preprocessing, read data and normalize data from file_path dataset
    """
    def __init__(self, file_path: str):
        self.data = self.read_data(file_path)
        # self.normalize = Normalize(self.data)

        self.execute()
        return self.data

    def read_data(self, file_path: str):
        """
            Use for read data from file
            input: file name
            output: dataframe
        """
        df = pd.read_csv(file_path)
        return df

    def execute(self):
        """
            Use for execute preprocessing
        """
        new_data = {'DATE': [], 'PRICE': [], 'OPEN': [], 'HIGH': [], 'LOW': [], 'VOL': [], 'CHANGE': []}
        for col in self.data:
            for row in self.data[col]:
                if col == 'DATE':
                    new_data[col].append(self.normalize.normalize_date(row))
                else:
                    new_data[col].append(self.normalize.normalize_number(row))

        self.data = pd.DataFrame(new_data)

        return self.data
