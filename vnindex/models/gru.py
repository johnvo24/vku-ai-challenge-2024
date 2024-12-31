import torch
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import numpy as np
from helper import inverse_transform, normalize_date
from sklearn.model_selection import train_test_split

class GRU_Module(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU_Module, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        gru_out, hn = self.gru(x)
        # out = self.drop(gru_out)
        out = self.fc(gru_out[:, -1, :])
        return out

class GRU:
    def __init__(self, data, output_size):
        self.data = data
        self.x_train = torch.tensor(self.data['train']['x'], dtype=torch.float32)
        self.y_train = torch.tensor(self.data['train']['y'], dtype=torch.float32)
        print(self.y_train.size())
        # self.x_test = torch.tensor(self.data['test']['x'], dtype=torch.float32)
        # self.y_test = torch.tensor(self.data['test']['y'], dtype=torch.float32)

        self.epoch = 300
        self.batch_size = 32
        self.hidden_size = 132
        self.output_size = output_size

        self.model = GRU_Module(self.x_train.size(2), self.hidden_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()

    def train(self):
        # best_loss = 100000000000
        for epoch in range(self.epoch):
            # Train
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.x_train)
            loss = self.loss(output, self.y_train)
            loss.backward()
            self.optimizer.step()

            # Eval
            # self.model.eval()
            # with torch.no_grad():
            #     output = self.model(self.x_test)
            #     loss = self.loss(output, self.y_test)
            #     current_loss = loss.item()
            #
            # if best_loss > current_loss:
            #     best_loss = current_loss
            #     print(f'Model Updated, Val Loss: {best_loss:.4f}')

            torch.save(self.model.state_dict(), f'results/gru/gru.pth')
            print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

    def eval(self):
        self.model.load_state_dict(torch.load(f'results/gru/gru.pth'))
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x_test)
            loss = self.loss(output, self.y_test)
            mse = loss.item()
            print(f'RMSE: {math.sqrt(loss.item()):.4f}')
            print(f'MSE: {loss.item():.4f}')

            return mse

    def use(self, day):
        input = torch.tensor(np.array([self.x_train[-6]]))

        self.model.load_state_dict(torch.load(f'results/gru/gru.pth'))
        self.model.eval()


        array = {
            'Date': day,
            'Price': [],
            'Open': [],
            'High': [],
            'Low': [],
            'Vol.': [],
            'Change %': []
        }

        for d in day:
            next_day = self.model(input)
            t_next_day = next_day
            temp = input[: ,1:, :]
            next_day = next_day.tolist()
            next_day = inverse_transform(next_day)
            array['Price'].append(str(next_day[0][1]))
            array['Open'].append(str(next_day[0][2]))
            array['High'].append(str(next_day[0][3]))
            array['Low'].append(str(next_day[0][4]))
            array['Vol.'].append(str(next_day[0][5]))
            array['Change %'].append(str(next_day[0][6]))

            temp = torch.concat((temp[0], t_next_day), dim=0)
            #input = temp
            input = torch.tensor(np.array([temp.tolist()]), dtype=torch.float32)

        df = pd.DataFrame(array)
        df.to_csv('results/result.csv', index=False)



