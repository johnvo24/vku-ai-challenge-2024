import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class XGBoost:
    def __init__(self, data):
        self.data = data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data[:, :-1], self.data[:, -1], test_size=0.2, random_state=42
        )
        self.model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.01,
                         max_depth=5, alpha=10, n_estimators=1000)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def eval(self):
        preds = self.model.predict(self.x_test)
        print(preds.shape)
        mse = mean_squared_error(self.y_test, preds)
        print("MSE: %f" % (mse))
