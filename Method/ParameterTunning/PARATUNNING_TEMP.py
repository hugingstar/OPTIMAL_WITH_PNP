import numpy as np
import datetime
import torch
from torch import nn
import os
import torch.optim as optim
import pandas as pd
import math
import CoolProp as CP
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class REGRESSION_SENSORS():
    def __init__(self,COMP_MODEL_NAME, TIME, start, end):
        """
        :param path: 데이터 파일 경로
        :param time: 시계열 인덱스 컬럼명
        :param features: 원본 데이터의
        스케일러 작업 필요
        """
        "파일을 호출할 경로"
        self.DATA_PATH = "/Data/ParameterTunning"
        self.SAVE_PATH = "/Results"
        self.TIME = TIME
        self.COMP_MODEL_NAME = COMP_MODEL_NAME

        self.jinli = {
            909 : [961, 999, 985, 1019, 1021, 1009, 939],
            910 : [940, 954, 958, 938, 944],
            921 : [922, 991, 977, 959, 980, 964, 1000, 1007],
            920 : [1022, 1011, 998, 981, 1005, 924, 1017],
            919 : [984, 988, 993, 950, 976, 956],
            917 : [971, 955, 1002, 1023, 1016, 922, 934],
            918 : [963, 986, 996, 1012, 1024, 1015, 943, 966],
            911 : [970, 974, 931, 948, 1014, 930, 968],
        }

        """디지털 도서관 정보"""
        self.dido = {
            3065 : [3109, 3100, 3095, 3112, 3133, 3074, 3092, 3105, 3091, 3124,
                   3071, 3072, 3123, 3125, 3106, 3099, 3081, 3131, 3094, 3084],
            3069 : [3077, 3082, 3083, 3089, 3096, 3104, 3110, 3117, 3134, 3102,
                   3116, 3129, 3090],
            3066 : [3085, 3086, 3107, 3128, 3108, 3121],
            3067 : [3075, 3079, 3080, 3088, 3094, 3101, 3111, 3114, 3115, 3119,
                   3120, 3122, 3130]
        }

        self.coefdict = {} #결과 저장

        #데이터 시작/끝 시간
        self.start = start
        self.end = end

        self.start_year = start[:4]
        self.start_month = start[5:7]
        self.start_date = start[8:10]
        self.end_year = end[:4]
        self.end_month = end[5:7]
        self.end_date = end[8:10]

        # 시작전에 폴더를 생성
        self.folder_name = "{}-{}-{}".format(self.start_year, self.start_month, self.start_date)
        self.create_folder('{}/ParameterTunning'.format(self.SAVE_PATH))  # ParameterTunning 폴더를 생성

    def PROCESSING(self, out_unit, X1, X2, target, TdisValue, TsucValue, freqValue, Method):
        # 예측 대상
        self.target = target
        self.freq = freqValue
        self.Method = Method

        # 건물 인식(딕셔너리에서 포함된 건물로 인식한다.)
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        # 실외기 데이터
        self._MapDatapath = "{}/GB066.csv".format(self.DATA_PATH)
        self._Mapdata = pd.read_csv(self._MapDatapath)
        print(self._Mapdata)

        # # 미터 값
        self.target = list(pd.Series(list(self._Mapdata.columns))[
                                    pd.Series(list(self._Mapdata.columns)).str.contains(pat=target, case=False)])[0]
        self.SuctionTemp = list(pd.Series(list(self._Mapdata.columns))[
                                    pd.Series(list(self._Mapdata.columns)).str.contains(pat=TsucValue, case=False)])[0]
        self.DischargeTemp = list(pd.Series(list(self._Mapdata.columns))[
                                      pd.Series(list(self._Mapdata.columns)).str.contains(pat=TdisValue, case=False)])[0]

        print("Suction temp : {} - Discharge temp : {}".format(self.SuctionTemp, self.DischargeTemp))
        #Density
        self.SuctionDensity()

        save = "{}/ParameterTunning/{}/{}".format(self.SAVE_PATH, self.folder_name, out_unit)
        self.create_folder(save)
        self._Mapdata.to_csv("{}/Before_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 전

        """필요한 경우 조건을 적용하는 장소이다."""
        # self._outdata = self._outdata[self._outdata[self.TotalIndoorCapacity] != 0] #작동중인 데이터만 사용
        # self._Mapdata = self._Mapdata.dropna(axis=0) # 결측값을 그냥 날리는 경우

        self._Mapdata.to_csv("{}/After_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 후

        self.X1 = list(pd.Series(list(self._Mapdata.columns))[
                                    pd.Series(list(self._Mapdata.columns)).str.contains(pat=X1, case=False)])[0]
        self.X2 = list(pd.Series(list(self._Mapdata.columns))[
                                    pd.Series(list(self._Mapdata.columns)).str.contains(pat=X2, case=False)])[0]
        print("X1 : {} - X2 : {}".format(self.X1, self.X2))
        self.ParametersTunningRated(save=save, out_unit=out_unit)

    def ParametersTunningRated(self, save, out_unit):
        var1 = torch.Tensor(self._Mapdata[self.X1].tolist()).unsqueeze(1)
        var2 = torch.Tensor(self._Mapdata[self.X2].tolist()).unsqueeze(1)
        dens = torch.Tensor(self._Mapdata[self.MapDensity].tolist()).unsqueeze(1)
        tar = torch.Tensor(self._Mapdata[self.target].tolist()).unsqueeze(1)
        # tar_avg = sum(self._Mapdata[self.target].tolist())/len(self._Mapdata[self.target].tolist())
        tar_avg = torch.Tensor([sum(self._Mapdata[self.target].tolist())/len(self._Mapdata[self.target].tolist())]).unsqueeze(1)

        # print(tar_avg)
        # print("{} - {}".format(self.X1 , list(var1)))
        # print("{} - {}".format(self.X2, list(var2)))
        # print("{} - {}".format(self.MapDensity, list(dens)))
        # print("{} - {}".format(self.target, list(tar)))

        b0 = torch.zeros(1, requires_grad=True)
        W1 = torch.zeros(1, requires_grad=True)
        W2 = torch.zeros(1, requires_grad=True)
        W3 = torch.zeros(1, requires_grad=True)
        W4 = torch.zeros(1, requires_grad=True)
        W5 = torch.zeros(1, requires_grad=True)
        W6 = torch.zeros(1, requires_grad=True)
        W7 = torch.zeros(1, requires_grad=True)
        W8 = torch.zeros(1, requires_grad=True)
        W9 = torch.zeros(1, requires_grad=True)
        W10 = torch.zeros(1, requires_grad=True)

        print("Optimizer : {}".format(self.Method))
        if self.Method == "Adam" :
            optimizer = optim.Adam([b0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.01)
        elif self.Method == "SGD":
            optimizer = optim.SGD([b0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.01)
        else:
            optimizer = optim.Adam([b0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.01)

        num = 0
        while True:
            #만들고 싶은 회귀식
            hypothesis = (b0
                                 + W1 * var1
                                 + W2 * var2
                                 + W3 * var1**2
                                 + W4 * var2**2
                                 + W5 * var1**3
                                 + W6 * var2**3
                                 + W7 * var1 * var2
                                 + W8 * var1 * var2**2
                                 + W9 * var1**2 * var2
                                 + W10 * var1**2 * var2**2)

            # Cost : RMSE
            cost = torch.mean(abs(tar - hypothesis)**2)
            cost = torch.sqrt(cost)/ tar_avg

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if num % 10000 == 0:
                # 변수에 따라서 웨이트 값 출력 조정 필요
                print('[Iteration {}] W0: {:.4f}, W1: {:.4f}, W2: {:.4f}, W3: {:.4f}, W4: {:.4f}, W5: {:.4f}, W6: {:.4f}, W7: {:.4f}, W8: {:.4f}, W9: {:.4f}, W10: {:.4f}, cost: {:.4f}'
                    .format(num, b0.item(),  W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item(), W7.item(), W8.item(), W9.item(), W10.item(), cost.item()))
            if cost.item() < 25:
                print("Done!")
                break
            num += 1

        self.coefdict = {
            self.target : [b0.item(),  W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item(), W7.item(), W8.item(), W9.item(), W10.item()]
        }
        df_coef = pd.DataFrame(self.coefdict)
        df_coef.to_csv("{}/ParameterRated_{}_{}.csv".format(save, self.target, out_unit))

        self.predperform = {
            "CvRMSE": [cost.item()]
        }
        df_per = pd.DataFrame(self.predperform)
        df_per.to_csv("{}/CvRMSERated_{}_{}.csv".format(save, self.target, out_unit))


    def SuctionDensity(self):
        """
        냉매 Density를 Coolprop에서 추정하는 함수
        밀도는 전원이 켜지나 안 켜지나 항상 물성치로 측정되도록 프로그래밍 되었다.
        SuctionPressure : 저압측 [Pa]로 환산하여 입력 [bar] --> [Pa]
        SuctionTemp : 흡입 온도 [K] = [C] + 373.15
        Desity :  R410A의 밀도 [kg/m^3]
        :return: 가상센서 값이 데이터에 합쳐짐
        """
        self.MapDensity = 'density'
        self._Mapdata[self.MapDensity] = None

        num = 0
        while num < self._Mapdata.shape[0]:
            self._Mapdata.at[self._Mapdata.index[num], "{}".format(self.MapDensity)] \
                = CP.CoolProp.PropsSI('D', 'T', self._Mapdata[self.SuctionTemp][num] + 273.15, 'Q', 1, 'R410A')
            num += 1

        print(self._Mapdata)

    def create_folder(self, directory):
        """
        폴더를 생성하는 함수이다.
        :param directory: 디렉토리 입력
        :return: 폴더가 생성되어 있을 것이다.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: creating directory. ' + directory)

#시계열 컬럼 이름
TIME = 'updated_time'
start ='2021-12-29' #데이터 시작시간
end = '2021-12-29' #데이터 끝시간

COMP_MODEL_NAME = 'GB066' # GB052, GB066, GB070, GB080
freqValue = 'frequency'
TdisValue = 'discharge_temp'
TsucValue = 'suction_temp'

# 변수
X1 = 'discharge_temp'
X2 = 'suction_temp'
TARGET = 'mass_flow_rate'

#Optimizer
METHOD = 'Adam' #SGD

# 스케일러 옵션은 끄면 Pass된다.
RVS = REGRESSION_SENSORS(COMP_MODEL_NAME=COMP_MODEL_NAME, TIME=TIME, start=start, end=end)

for outdv in [3066]:
    RVS.PROCESSING(out_unit=outdv, X1=X1, X2=X2, target=TARGET,
                   TsucValue=TsucValue, TdisValue=TdisValue, freqValue=freqValue, Method=METHOD)
