import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from SIEMENS import OPTIMALSTART

class SOFT():
    def __init__(self, path, time, features, scaler):
        """
        :param path: 데이터 파일 경로
        :param time: 시계열 인덱스 컬럼명
        :param features: 특징들
        스케일러 작업 필요
        """
        self.path = path
        self.data_org = pd.read_csv(self.path, index_col=time)
        self.data = self.data_org[features]

        if scaler == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler == 'RobustScaler':
            scaler = RobustScaler()
        elif scaler == 'MaxAbsScaler':
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()

        self.data[list(self.data.columns)] = scaler.fit_transform(self.data[list(self.data.columns)])
        # print(self.data[features])
        torch.manual_seed(1)

    def TEMP(self, x1, x2, y):
        var1 = torch.Tensor(self.data[x1].tolist()).unsqueeze(1)
        var2 = torch.Tensor(self.data[x2].tolist()).unsqueeze(1)
        tar = torch.Tensor(self.data[y].tolist()).unsqueeze(1)

        W1 = torch.zeros(1, requires_grad=True)
        W2 = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        optimizer = optim.SGD([W1, b], lr=0.01)
        nb_epochs = 100
        for epoch in range(nb_epochs + 1):

            # Hypo 계산
            hypothesis = W1 * var1 * var2 + W2 * var2 + b

            # Cost 계산
            cost = torch.mean((hypothesis - tar) ** 2)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if epoch % 10 == 0:
                #변수에 따라서 웨이트 값 출력 조정 필요
                # print('Epoch {:4d}/{} W1: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W1.item(), b.item(), cost.item()))
                print('Epoch {:4d}/{} W1: {:.3f}, W2: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W1.item(), W2.item(), b.item(), cost.item()))
        return 0


path = 'D:/OPTIMAL/Data/2022-03-07/outdoor_909.csv'
time = 'updated_time'
features = ['total_indoor_capa', 'comp1','comp2', 'suction_temp1',
            'discharge_temp1', 'discharge_temp2', 'discharge_temp3',
            'outdoor_temperature', 'high_pressure', 'low_pressure', 'eev1',
            'value','double_tube_temp', 'evi_eev','fan_step']

x1 = 'discharge_temp1'
x2 = 'high_pressure'
y = 'value'

# Scaler option : StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler
scaler = 'MinMaxScaler'

"""Temperature virtual sensor"""
SOFT(path=path, time=time, features=features, scaler=scaler).TEMP(x1=x1, x2=x2, y=y)



