from vnindex.models.xgboosst import XGBoost
# from vnindex.models.lstm import LSTM

class Modeling:
    def __init__(self, input):
        self.input = input
        self.xgboost = XGBoost(self.input)
        # self.lstm = LSTM(self.input)
