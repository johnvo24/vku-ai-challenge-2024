import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from vnindex import Preprocessing
import matplotlib.pyplot as plt
import numpy as np
import joblib
from helper import *

class LinearRegressionModel():
  def __init__(self):
    self.preprocessing = Preprocessing('res/AIChallenge_Training.csv')
    data = self.preprocessing.data
    data = data.values
    self.X_data = data[:-1,:]
    self.y_data = data[1:, 1]
    self.model = LinearRegression()
    self.folder = "results/linear_regression/"
    print(pd.concat([pd.DataFrame(self.X_data), pd.DataFrame(self.y_data)], axis=1))

  def train(self): 
    # X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=42)
    threshold = int(len(self.X_data) * 0.8)
    X_train = self.X_data[:threshold]
    X_test = self.X_data[threshold:]
    y_train = self.y_data[:threshold]
    y_test = self.y_data[threshold:]
    self.model.fit(X_train, y_train)
    # Save model
    joblib.dump(self.model, self.folder+'linear_regression_model.pkl')
    print("Saved the model!")
    self.evaluate(X_test, y_test)

  def evaluate(self, X_test, y_test):
    # Đánh giá mô hình
    self.model = joblib.load(self.folder+'linear_regression_model.pkl')
    y_pred = self.model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # In các hệ số của mô hình
    # print("Hệ số của mô hình (coefficients):", self.model.coef_)
    # print("Giao điểm (intercept):", self.model.intercept_)
    print("Root Mean Squared Error (RMSE):  ", rmse)
    print("Mean Squared Error (MSE):  ", mse)
    print("R-squared (R2):            ", r2)

    # Bước 5: Biểu đồ so sánh giá thực tế và dự đoán
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Giá thực tế", marker='o')
    plt.plot(y_pred, label="Giá dự đoán", marker='x')
    plt.legend()
    plt.title("So sánh giá thực tế và giá dự đoán")
    plt.xlabel("Mẫu")
    plt.ylabel("Giá cổ phiếu")
    plt.savefig(self.folder+'/plot.png')

  def predict(self, x_data):
    self.model = joblib.load(self.folder+'linear_regression_model.pkl')
    y_pred = self.model.predict([x_data])
    predicted_scaled_expanded = np.repeat([y_pred], 7, axis=1)
    print(predicted_scaled_expanded)
    predict = self.preprocessing.scaler.inverse_transform(predicted_scaled_expanded)
    # print(self.preprocessing.scaler.inverse_transform([x_data]))
    return predict

lr = LinearRegressionModel()
# lr.train()
# lr.evaluate(lr.X_data, lr.y_data)

# print(lr.X_data)
# print(f"Prediction Input: {lr.X_data[3]}")
# print(f"Prediction Input: {lr.X_data[3]}")
print(lr.predict(lr.X_data[990]))