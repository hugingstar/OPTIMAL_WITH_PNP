import numpy as np
import datetime
import torch
import os
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class REGRESSION_SENSORS():
    def __init__(self, path, time, features, scaler):
        """
        :param path: 데이터 파일 경로
        :param time: 시계열 인덱스 컬럼명
        :param features: 원본 데이터의
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
            pass #스케일러 옵션을 설정 안한 경우에는 Pass

        try:
            self.data[list(self.data.columns)] = scaler.fit_transform(self.data[list(self.data.columns)])
        except:
            pass #스케일러 옵션을 설정 안한 경우에는 Pass

        # print(self.data[features])
        torch.manual_seed(1)

    def REG_MODEL(self, x1, x2, y):
        # 입력값을 설정하여 텐서로 만들어 준다.
        var1 = torch.Tensor(self.data[x1].tolist()).unsqueeze(1)
        var2 = torch.Tensor(self.data[x2].tolist()).unsqueeze(1)
        tar = torch.Tensor(self.data[y].tolist()).unsqueeze(1)

        # 학습시킬 가중치 설정
        W1 = torch.zeros(1, requires_grad=True)
        W2 = torch.zeros(1, requires_grad=True)
        W3 = torch.zeros(1, requires_grad=True)
        W4 = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        # Optimizer
        optimizer = optim.SGD([W1, W2, W3, W4, b], lr=0.01)
        # Epochs
        nb_epochs = 100
        for epoch in range(nb_epochs + 1):

            #만들고 싶은 회귀식
            # hypothesis = b + W1 * var1 + W2 * var2
            hypothesis = b + W1 * var1 + W2 * var2 + W3 * var1**2 + W4 * var2**2

            # Cost
            cost = torch.mean((hypothesis - tar) ** 2)


            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if epoch % 10 == 0:
                #변수에 따라서 웨이트 값 출력 조정 필요
                print('Epoch {:4d}/{} - b : {:.4f} - W1 : {:.4f} - W2 : {:.4f} - W3 : {:.4f} - W4 : {:.4f}  Cost: {:.4f}'
                      .format(epoch, nb_epochs, b.item(), W1.item(), W2.item(), W3.item(), W4.item(), cost.item()))

# 경로
path = 'D:/OPTIMAL/Data/Optimal/2022-03-30/outdoor_909.csv'

#시계열 컬럼 이름
time = 'updated_time'
# 원본 데이터에서 필요한 데이터만 불러오기
features = ['Bldg_Jinli/Outd_909/suction_temp1', 'Bldg_Jinli/Outd_909/discharge_temp1', 'Bldg_Jinli/Outd_909/discharge_temp2',
            'Bldg_Jinli/Outd_909/outdoor_temperature', 'Bldg_Jinli/Outd_909/high_pressure', 'Bldg_Jinli/Outd_909/low_pressure',
            'Bldg_Jinli/Outd_909/value', 'Bldg_Jinli/Outd_909/double_tube_temp', 'Bldg_Jinli/Outd_909/total_indoor_capa']
# 변수
x1 = 'Bldg_Jinli/Outd_909/suction_temp1'
x2 = 'Bldg_Jinli/Outd_909/discharge_temp1'
y = 'Bldg_Jinli/Outd_909/total_indoor_capa'

# Scaler option : StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler
scaler = 'StandardScaler' #StandardScaler

# 스케일러 옵션은 끄면 Pass된다.
RVS = REGRESSION_SENSORS(path=path, time=time, features=features, scaler=scaler)

RVS.REG_MODEL(x1=x1, x2=x2, y=y)



