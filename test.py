from vnindex import Preprocessing, Modelling, Chart
import numpy as np
import pandas as pd

train_preprocessing = Preprocessing('res/dataset.csv')
train_data = train_preprocessing.data
print(train_data)

# Draw charts
# chart = Chart(data)
# chart.ScatterPlot('VOL', 'CHANGE')
# chart.ScatterPlot('PRICE', 'OPEN')
# chart.PairPlot()

train_data = train_data.values
model = Modelling(input=train_data, num_predict_date=7, num_date=6, target_col=1)

# XGBOOST MODELLING
# model.xgboost.train()
# model.xgboost.eval()

# LSTM MODELLING
# model.lstm.train()
# model.lstm.eval()

# GRU MODELLING
model.gru.train()
# model.gru.eval()
# day = pd.read_csv('res/test.csv')['Date']
# model.gru.use(day)
