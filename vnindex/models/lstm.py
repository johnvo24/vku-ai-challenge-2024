import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

class LSTM_Module(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Module, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTM:
    def __init__(self, data, output_size):
        self.data = data
        self.x_train = torch.tensor(self.data['train']['x'], dtype=torch.float32)
        self.y_train = torch.tensor(self.data['train']['y'], dtype=torch.float32)
        self.x_test = torch.tensor(self.data['test']['x'], dtype=torch.float32)
        self.y_test = torch.tensor(self.data['test']['y'], dtype=torch.float32)
        print(self.x_train)

        self.epoch = 1000
        self.batch_size = 8
        self.hidden_size = 32
        self.output_size = output_size

        self.model = LSTM_Module(self.x_train.size(2), self.hidden_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def train(self):
        self.model.train()
        for epoch in range(self.epoch):
            self.optimizer.zero_grad()
            output = self.model(self.x_train)
            loss = self.loss(output, self.y_train)
            loss.backward()
            self.optimizer.step()

            print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x_test)
            loss = self.loss(output, self.y_test)
            print(f'Loss: {loss.item():.4f}')
