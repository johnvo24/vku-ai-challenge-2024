from vnindex import Preprocessing, Modeling
import numpy as np
import pandas as pd

preprocessing = Preprocessing('res/dataset.csv')
data = preprocessing.data
data = data.values

model = Modeling(data)
model.xgboost.train()
model.xgboost.eval()
