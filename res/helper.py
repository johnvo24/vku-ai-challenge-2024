from pandas import DataFrame
from vnindex import Preprocessing

def inverse_transform(y_pred: DataFrame):
  preprocessing = Preprocessing('res/AIChallenge_Training.csv')
  predict = preprocessing.scaler.inverse_transform(y_pred)
  # preprocessing.scaler.fit
  return predict

print(inverse_transform(DataFrame([[-2.048243, -0.877249 ,-0.816304 ,-0.845086 ,-0.839612 ,-1.637142  ,0.249599]])))