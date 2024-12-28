from sklearn.model_selection import train_test_split

class LinearRegression():
  def __init__(self, data):
    self.data = data
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.data[:, :-1], self.data[:, -1], test_size=0.2, random_state=42
    )