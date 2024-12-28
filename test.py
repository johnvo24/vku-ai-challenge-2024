from vnindex import Preprocessing, Modelling, Chart
import numpy as np
import pandas as pd

preprocessing = Preprocessing('res/dataset.csv')
data = preprocessing.data
print(data)

# Draw charts
chart = Chart(data)
# chart.ScatterPlot('VOL', 'CHANGE')
# chart.ScatterPlot('PRICE', 'OPEN')
chart.PairPlot()

# data = data.values
#
# model = Modelling(input=data, num_predict_date=1, num_date=14)

# XGBOOST MODELLING
# model.xgboost.train()
# model.xgboost.eval()

# LSTM MODELLING
# model.lstm.train()
# model.lstm.eval()
