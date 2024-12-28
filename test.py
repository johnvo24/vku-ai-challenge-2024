from vnindex import Preprocessing, Modelling, Chart
import numpy as np
import pandas as pd

# Draw charts
data = pd.read_csv('res/dataset.csv')
data = data.dropna()
chart = Chart(data)
# chart.ScatterPlot('VOL', 'CHANGE')
# chart.ScatterPlot('PRICE', 'OPEN')
chart.PairPlot()

# preprocessing = Preprocessing('res/dataset.csv')
# data = preprocessing.data
# data = data.values
#
# model = Modelling(input=data, num_predict_date=1, num_date=14)

# XGBOOST MODELLING
# model.xgboost.train()
# model.xgboost.eval()

# LSTM MODELLING
# model.lstm.train()
# model.lstm.eval()
