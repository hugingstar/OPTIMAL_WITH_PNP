import numpy as np
import datetime
import torch
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
        self.DATA_PATH = "D:/OPTIMAL/Data/ParameterTunning"
        self.SAVE_PATH = "D:/OPTIMAL/Results"
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

    def PROCESSING(self, out_unit, X1, X2, target, PdisValue, PsucValue, TdisValue, TsucValue, TotIndCapa, Method):
        # 예측 대상
        self.target = target

        # 건물 인식(딕셔너리에서 포함된 건물로 인식한다.)
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        # 실외기 데이터
        self._outdpath = "{}/{}/Outdoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit)
        self._outdata = pd.read_csv(self._outdpath, index_col=self.TIME)

        # 미터 값
        self.target = list(pd.Series(list(self._outdata.columns))[
                                    pd.Series(list(self._outdata.columns)).str.contains(pat=target, case=False)])[0]
        self.SuctionPressure = list(pd.Series(list(self._outdata.columns))[
                                        pd.Series(list(self._outdata.columns)).str.contains(pat=PsucValue, case=False)])[0]
        self.SuctionTemp = list(pd.Series(list(self._outdata.columns))[
                                    pd.Series(list(self._outdata.columns)).str.contains(pat=TsucValue, case=False)])[0]
        self.DischargePressure = list(pd.Series(list(self._outdata.columns))[
                                          pd.Series(list(self._outdata.columns)).str.contains(pat=PdisValue, case=False)])[0]
        self.DischargeTemp = list(pd.Series(list(self._outdata.columns))[
                                      pd.Series(list(self._outdata.columns)).str.contains(pat=TdisValue, case=False)])[0]
        self.TotalIndoorCapacity = list(pd.Series(list(self._outdata.columns))[
                                      pd.Series(list(self._outdata.columns)).str.contains(pat=TotIndCapa, case=False)])[0]

        # 가상센서 결과를 저장하기 위한 컬럼명 생성
        self.CondMapTemp = self.target.replace(target, 'cond_map_temp')  # 맵에서 찾은 데이터 컬럼이름 조정
        self.EvaMapTemp = self.target.replace(target, 'evap_map_temp')
        self.MapDensity = self.target.replace(target, 'density')

        # Map data
        Mdensity_prev = 0
        evapMtemp_prev = 0
        condMtemp_prev = 0
        for o in range(self._outdata.shape[0]):
            # MapDensity()
            try:
                Mdensity = CP.CoolProp.PropsSI('D', 'P', self._outdata[self.SuctionPressure][o] * 98.0665 * 1000, 'T', self._outdata[self.SuctionTemp][o] + 273.15, 'R410A')
                evapMtemp = CP.CoolProp.PropsSI('T', 'P', self._outdata[self.SuctionPressure][o] * 98.0665 * 1000, 'Q', 0.5, 'R410A') - 273.15
                condMtemp = CP.CoolProp.PropsSI('T', 'P', self._outdata[self.DischargePressure][o] * 98.0665 * 1000, 'Q', 0.5, 'R410A') - 273.15
                if Mdensity_prev != 0:
                    if abs(Mdensity/Mdensity_prev) > 20:  # 값이 튀는 Outlier 현상을 막음
                        Mdensity = min(Mdensity, Mdensity_prev)
                Mdensity_prev = Mdensity
                evapMtemp_prev = evapMtemp
                condMtemp_prev = condMtemp
            except:
                Mdensity = Mdensity_prev
                evapMtemp = evapMtemp_prev
                condMtemp = condMtemp_prev

            self._outdata.at[self._outdata.index[o], "{}".format(self.MapDensity)] = Mdensity
            # EvaMapTemp(Celcius) Quality : 1
            self._outdata.at[self._outdata.index[o], "{}".format(self.EvaMapTemp)] = evapMtemp
            # CondMapTemp(Celcius) Quality : 1
            self._outdata.at[self._outdata.index[o], "{}".format(self.CondMapTemp)] = condMtemp

        # # 미터기값의 차이 컬럼 추가
        # for o in range(self._outdata.shape[0]-1):
        #     c_o = round(self._outdata[self.meter_value][o + 1] - self._outdata[self.meter_value][o], 3)
        #     self._outdata.at[self._outdata.index[o], "{}_difference".format(self.meter_value)] = c_o
        # self._outdata.at[self._outdata.index[-1], "{}_difference".format(self.meter_value)] = c_o

        # 데이터 생성된것을 시계열을 기준으로 정렬 하고 결측값 처리
        self._outdata = self._outdata.sort_values(self.TIME)
        self._outdata = self._outdata.fillna(method='ffill')  # 결측값 처리

        save = "{}/ParameterTunning/{}/{}".format(self.SAVE_PATH, self.folder_name, out_unit)
        self.create_folder(save)
        self._outdata.to_csv("{}/Before_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 전

        """필요한 경우 조건을 적용하는 장소이다."""
        self._outdata = self._outdata[self._outdata[self.TotalIndoorCapacity] != 0] #작동중인 데이터만 사용
        # self._outdata = self.data.dropna(axis=0) # 결측값을 그냥 날리는 경우

        self._outdata.to_csv("{}/After_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 후

        self.X1 = list(pd.Series(list(self._outdata.columns))[
                                    pd.Series(list(self._outdata.columns)).str.contains(pat=X1, case=False)])[0]
        self.X2 = list(pd.Series(list(self._outdata.columns))[
                                    pd.Series(list(self._outdata.columns)).str.contains(pat=X2, case=False)])[0]
        self.Method = Method # Optimizer method
        Wdot_rated_list = self.WdotRatedParaTunning() # Wdot Rated Parameters Tunning

        self.coefdict = {
            self.COMP_MODEL_NAME :
                {"Wdot_rated" : Wdot_rated_list}
        }


        # for indv in list(self.bldginfo[out_unit]):
        #     #실내기 데이터
        #     self._indpath = "{}/{}/{}/Outdoor_{}_Indoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit, out_unit, indv)
        #     self._indata = pd.read_csv(self._indpath, index_col=self.TIME)
        #
        #     #실내기 및 실외기의 데이터 통합
        #     self.data = pd.concat([self._outdata, self._indata], axis=1)
        #     self.data.index.names = [self.TIME] # 인덱스 컬럼명이 없는 경우를 대비하여 보완
        #     #문자열로 된 원본 데이터의 '모드'를 숫자로 변환
        #     self.data = self.data.replace({"High": 3, "Mid" : 2, "Low" : 1, "Auto" : 3.5})

    def WdotRatedParaTunning(self):
        var1 = torch.Tensor(self._outdata[self.X1].tolist()).unsqueeze(1) #unsqeeze : 1인 차원을 생성하는 함수이다.
        var2 = torch.Tensor(self._outdata[self.X2].tolist()).unsqueeze(1)
        dens = torch.Tensor(self._outdata[self.MapDensity].tolist()).unsqueeze(1)
        tar = torch.Tensor(self._outdata[self.target].tolist()).unsqueeze(1)
        print("{} - {}".format(self.X1 , list(var1)))
        print("{} - {}".format(self.X2, list(var2)))
        print("{} - {}".format(self.MapDensity, list(dens)))
        print("{} - {}".format(self.target, list(tar)))

        W0 = torch.zeros(1, requires_grad=True)
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

        if self.Method == "Adam" :
            optimizer = optim.Adam([W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.01)
        elif self.Method == "SGD":
            optimizer = optim.SGD([W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.01)
        else:
            optimizer = optim.Adam([W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.01)

        num = 0
        while True:
            #만들고 싶은 회귀식
            hypothesis = dens * (W0 + W1 * var1 + W2 * var2 + W3 * var1**2 + W4 * var2**2
                         + W5 * var1**3 + W6 * var2**3 + W7 * var1 * var2 + W8 * var1 * var2**2
                         + W9 * var1**2 * var2 + W10 * var1**2 * var2**2)
            # Cost : RMSE
            cost = torch.mean((hypothesis - tar) ** 2)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if num % 10000 == 0:
                # 변수에 따라서 웨이트 값 출력 조정 필요
                print('Iteration Number: {} - W0: {:.4f} - W1: {:.4f} - W2: {:.4f} - W3: {:.4f} - W4: {:.4f} - W5: {:.4f} - W6: {:.4f} - W7: {:.4f} - W8: {:.4f} - W9: {:.4f} - W10: {:.4f} - cost: {:.4f}'
                    .format(num, W0.item(),  W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item(), W7.item(), W8.item(), W9.item(), W10.item(), math.sqrt(cost.item())))
            num += 1
            if math.sqrt(cost.item()) < 20:
                break

        weights = [W0.item(),  W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item(), W7.item(), W8.item(), W9.item(), W10.item()]
        print(weights)
        return weights

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
start ='2022-03-30' #데이터 시작시간
end = '2022-03-31' #데이터 끝시간

COMP_MODEL_NAME = 'GB066' # GB052, GB066, GB070, GB080

PdisValue = 'high_pressure'
PsucValue = 'low_pressure'
TdisValue = 'discharge_temp1'
TsucValue = 'suction_temp1'
TotIndCapa = 'total_indoor_capa'

# 변수
# X1 = 'suction_temp1'
# X2 = 'discharge_temp1'
X1 = 'cond_map_temp'
X2 = 'evap_map_temp'
TARGET = 'value'

#Optimizer
METHOD = 'Adam' #SGD

# 스케일러 옵션은 끄면 Pass된다.
RVS = REGRESSION_SENSORS(COMP_MODEL_NAME=COMP_MODEL_NAME, TIME=TIME, start=start, end=end)

for outdv in [909]:
    RVS.PROCESSING(out_unit=outdv, X1=X1, X2=X2, target=TARGET,
                   PdisValue=PdisValue, PsucValue=PsucValue, TsucValue=TsucValue,
                   TdisValue=TdisValue, TotIndCapa=TotIndCapa,
                   Method=METHOD)
