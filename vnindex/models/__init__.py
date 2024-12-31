import numpy as np
from vnindex.models.xgboosst import XGBoost
from vnindex.models.lstm import LSTM
from vnindex.models.gru import GRU

class Modelling:
    '''
        This class is for modelling ML and DL models
    '''
    def __init__(self, input=None, num_predict_date=1, num_date=6, target_col=2):
        inp, out = self.data_split(input, num_date, num_predict_date, target_col)
        print(f'Input shape: {inp.shape}, Output shape: {out.shape}')

        # Split into training and testing data
        data = {
            'train': {
                'x': inp[:, :, :],
                'y': out[:, :, :]
            }
        }

        print(f'X Training shape: {data["train"]["x"].shape}')
        print(f'Y Training shape: {data["train"]["y"].shape}')
        self.input = data

        # Call models
        # self.xgboost = XGBoost(self.input)
        # self.lstm = LSTM(data, num_predict_date)
        # self.xgboost = XGBoost(self.input)
        self.gru = GRU(self.input, num_predict_date)

    def data_split(self, data, num_date, num_predict_date, target_col):
        '''
            Handling data from 2D to 3D (batch, date, col), using algorithm to split input and output for modelling step
        '''
        input = []
        output = []

        for i in range(0, len(data) - num_date - num_predict_date + 1):
            temp_inp = []
            temp_out = []
            for j in range(i, i + num_date):
                temp_inp.append(data[j])

            temp_out.append(data[j + 1])

            input.append(temp_inp)
            output.append(temp_out)

        input = np.array(input)
        output = np.array(output)

        return input, output

