from pandas import DataFrame
from vnindex import Preprocessing

def inverse_transform(y_pred: DataFrame):
  preprocessing = Preprocessing('res/dataset.csv')
  predict = preprocessing.scaler.inverse_transform(y_pred)
  # preprocessing.scaler.fit
  return predict

def normalize_date(date_str, format='%m/%d/%Y'):
    date_str = date_str.split('/')
    date_str[2] = '20' + date_str[2][0] + date_str[2][1]
    date_str = date_str[0] + '/' + date_str[1] + '/' + date_str[2]

    date_obj = datetime.strptime(date_str, format)
    return int(date_obj.strftime("%Y%m%d"))%1000000

# print(inverse_transform(DataFrame([[-2.048243, -0.877249 ,-0.816304 ,-0.845086 ,-0.839612 ,-1.637142  ,0.249599]])))
