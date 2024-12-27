from vnindex.models.xgboosst import XGBoost
import numpy as np
from vnindex.models.lstm import LSTM

class Modelling:
    def __init__(self, input=None, num_predict_date=1, num_date=6):
        input = self.data_split(input, num_date)
        print(f'Input shape: {input.shape}')

        threshold = int(0.8 * len(input))
        data = {
            'train': {
                'x': np.delete(input[:threshold], 2, axis=-1),
                'y': input[:threshold, :, 2]
            },
            'test': {
                'x': np.delete(input[threshold:], 2, axis=-1),
                'y': input[threshold:, :, 2]
            }
        }
        # chia tap test ra thanh output n ngay tiep theo, chua xong
        print(f'X Training shape: {data["train"]["x"].shape}, X Testing shape: {data["test"]["x"].shape}')
        print(f'Y Training shape: {data["train"]["y"].shape}, Y Testing shape: {data["test"]["y"].shape}')
        self.input = data

        self.xgboost = XGBoost(self.input)
        self.lstm = LSTM(self.input, num_predict_date)

    def data_split(self, data, num_date):
        new_df = []

        for i in range(0, len(data) - num_date + 1):
            temp = []
            for j in range(i, i + num_date):
                temp.append(data[j])
            new_df.append(temp)

        new_df = np.array(new_df)

        return new_df

