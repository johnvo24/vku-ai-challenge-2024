import xgboost as xgb
from sklearn.model_selection import train_test_split
from vnindex import Preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np

class XGBoost:
    def __init__(self, data):
        preprocessing = Preprocessing('res/dataset.csv')
        data = preprocessing.data
        data = data.values
        self.X_data = data[:-1,:]
        self.y_data = data[1:, 1:5]
        print((self.X_data, self.y_data))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=42)

        self.model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.01,
                         max_depth=5, alpha=10, n_estimators=1000)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def eval(self):
        preds = self.model.predict(self.X_test)
        print(preds.shape)
        mse = mean_squared_error(self.y_test, preds)
        print("MSE: %f" % (mse))
