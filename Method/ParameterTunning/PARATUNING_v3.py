import numpy as np
import datetime
import torch
from torch import nn
import os
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import math
import CoolProp as CP
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# print(torch.cuda.is_available())
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
pd.set_option('mode.chained_assignment',  None)

class REGRESSION_SENSORS():
    def __init__(self,COMP_MODEL_NAME, TIME, start, end):
        print("Start : {}".format(datetime.datetime.now()))
        """
        :param path: 데이터 파일 경로
        :param time: 시계열 인덱스 컬럼명
        :param features: 원본 데이터의
        스케일러 작업 필요
        """
        "파일을 호출할 경로"
        self.DATA_PATH = "D:/OPTIMAL/Data/ParameterTuning"
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
            911 : [970, 974, 931, 948, 1014, 930, 968]
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


        self.exp_3065 = {"Normal": ['2022-01-10', '2022-01-20'],
                           "KT2": ['2022-01-20'],
                           "KT3": ['2022-01-21'],
                           "KT4": []}

        self.exp_3066 = {"Normal": ['2021-12-28', '2021-12-29', '2022-01-03'],
                           "KT2": ['2022-01-04', '2022-01-05'],
                           "KT3": ['2022-01-06'],
                           "KT4": ['2022-01-07']}

        self.exp_3067 = {"Normal": ['2022-01-24'],
                           "KT2": ['2022-01-24'],
                           "KT3": ['2022-01-26'],
                           "KT4": ['2022-01-26']}

        self.exp_3069 = {"Normal": ['2022-01-17'],
                           "KT2": ['2022-01-18'],
                           "KT3": ['2022-01-19'],
                           "KT4": ['2022-01-19']}

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
        self.create_folder('{}/ParameterTuning'.format(self.SAVE_PATH))  # ParameterTuning 폴더를 생성

    def PROCESSING(self, out_unit, target, TdisValue, TsucValue, compValue, freqValue,
                   PsucValue, PdisValue, comp_num, Method):

        # 예측 대상
        self.target = target
        self.Method = Method
        self.TrainIter = 15000

        # 건물 인식(딕셔너리에서 포함된 건물로 인식한다.)
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        # 실외기 데이터
        self._OutUnitDataPath = "{}/{}/Outdoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit)
        self._OutUnitData = pd.read_csv(self._OutUnitDataPath)
        # print(self._OutUnitData)

        # 미터 값
        self.target = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=target, case=False)])
        self.SuctionTemp = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=TsucValue, case=False)])
        del self.SuctionTemp[-1]
        self.DischargeTemp = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=TdisValue, case=False)])
        del self.DischargeTemp[-1]
        self.CompressorSignal = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=compValue, case=False)])
        del self.CompressorSignal[2:]
        self.Compfreq = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=freqValue, case=False)])
        self.SuctionPressure = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=PsucValue, case=False)])
        self.DischargePressure = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=PdisValue, case=False)])

        self._OutUnitData['comp_frequency_sum'] = self._OutUnitData[self.Compfreq[0]] + self._OutUnitData[self.Compfreq[1]]
        self._OutUnitData['discharge_temp_avg'] = (self._OutUnitData[self.DischargeTemp[0]] + self._OutUnitData[self.DischargeTemp[1]]) / 2

        self.selected_feature = (['updated_time']
                                 + self.SuctionTemp
                                 + self.DischargeTemp
                                 + ['discharge_temp_avg']
                                 + self.CompressorSignal
                                 + self.Compfreq
                                 + ['comp_frequency_sum']
                                 + self.SuctionPressure
                                 + self.DischargePressure
                                 + self.target)

        print(self.selected_feature)
        save = "{}/ParameterTuning/{}/{}".format(self.SAVE_PATH, self.folder_name, out_unit)
        self.create_folder(save)
        self._OutUnitData = self._OutUnitData[self.selected_feature]
        self._OutUnitData.to_csv("{}/Before_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 전

        """필요한 경우 조건을 적용하는 장소이다."""
        self.comp_num = comp_num

        #Frequency Filtering
        if self.comp_num == 1:
            # Mode1 : comp1 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData[self.Compfreq[0]] > 15) # 최소 추파수 값
                                                  & (self._OutUnitData[self.CompressorSignal[0]] != 0)
                                                  & (self._OutUnitData[self.CompressorSignal[1]] == 0)
                                                  & (self._OutUnitData[self.target[0]] > 1000)]  # 작동중인 데이터만 사용
            self.freq_rated = float(self.OutIntegData[self.Compfreq[0]].mode()[0])
            print("Comp {} -  Rated frequency : {}".format(self.comp_num, self.freq_rated))
            self.OutIntegData = self.OutIntegData[(self.OutIntegData[self.Compfreq[0]] == self.freq_rated)]

        elif self.comp_num == 2:
            # Mode2 : comp2 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData[self.Compfreq[1]] > 15) # 최소 추파수 값
                                                  & (self._OutUnitData[self.CompressorSignal[0]] == 0)
                                                  & (self._OutUnitData[self.CompressorSignal[1]] != 0)
                                                  & (self._OutUnitData[self.target[0]] > 1000)]  # 작동중인 데이터만 사용
            self.freq_rated = float(self.OutIntegData[self.Compfreq[1]].mode()[0])
            print("Comp {} -  Rated frequency : {}".format(self.comp_num, self.freq_rated))
            self.OutIntegData = self.OutIntegData[(self.OutIntegData[self.Compfreq[1]] == self.freq_rated)]

        elif self.comp_num == 3:
            # Mode3 : comp1 + comp2 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData['comp_frequency_sum'] > 15)
                                                  & (self._OutUnitData[self.CompressorSignal[0]] != 0)
                                                  & (self._OutUnitData[self.CompressorSignal[1]] != 0)
                                                  & (self._OutUnitData[self.target[0]] > 1000)]  # 작동중인 데이터만 사용
            self.freq_rated = float(self.OutIntegData['comp_frequency_sum'].mode()[0])
            print("Comp {} -  Rated frequency : {}".format(self.comp_num, self.freq_rated))
            self.OutIntegData = self.OutIntegData[(self.OutIntegData['comp_frequency_sum'] == self.freq_rated)]

        #Saturated 값 계산
        self.OutIntegData = self.CalculatingSaturatedValue(data=self.OutIntegData)
        # [Step 1] Rating : 작동 데이터로 파라미터 튜닝 1
        self.OutIntegData = self.ParaTuneRating(data=self.OutIntegData, save=save, out_unit=out_unit)
        # kp 값 계산
        self.OutIntegData = self.CalculatingKpvalue(data=self.OutIntegData)

        self.OutIntegData = self.ParaTuneFreq(data=self.OutIntegData, save=save, out_unit=out_unit)

        self.OutIntegData.to_csv("{}/After_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

    def CalculatingSaturatedValue(self, data):
        self.SatCondTemp = 'condenser_temp'
        self.SatEvapTemp = 'evaporator_temp'

        # ParaTuneRating test
        data.reset_index(inplace=True, drop=True)
        num = 0
        while num < data.shape[0]:
            SatCndTemp = CP.CoolProp.PropsSI('T', 'P', data[self.DischargePressure[0]][num] * 100 * 1000, 'Q', 0.5, 'R410A') - 273.15
            SatEvaTemp = CP.CoolProp.PropsSI('T', 'P', data[self.SuctionPressure[0]][num] * 100 * 1000, 'Q', 0.5, 'R410A') - 273.15
            data.at[data.index[num], "{}".format(self.SatCondTemp)] = SatCndTemp
            data.at[data.index[num], "{}".format(self.SatEvapTemp)] = SatEvaTemp
            num += 1

        num = 0
        while num < self._OutUnitData.shape[0]:
            try:
                SatCndTemp = CP.CoolProp.PropsSI('T', 'P', self._OutUnitData[self.DischargePressure[0]][num] * 100 * 1000, 'Q', 0.5,'R410A') - 273.15
            except ValueError:
                SatCndTemp = 0

            try:
                SatEvaTemp = CP.CoolProp.PropsSI('T', 'P', self._OutUnitData[self.SuctionPressure[0]][num] * 100 * 1000, 'Q', 0.5,'R410A') - 273.15
            except ValueError:
                SatCndTemp = 0
            self._OutUnitData.at[self._OutUnitData.index[num], "{}".format(self.SatCondTemp)] = SatCndTemp
            self._OutUnitData.at[self._OutUnitData.index[num], "{}".format(self.SatEvapTemp)] = SatEvaTemp
            num += 1
        return data

    def ParaTuneRating(self, data, save, out_unit):
        self.RatingPower = 'power_rating'
        """독립변수"""
        X1_ = self.SatEvapTemp
        var1 = torch.Tensor(data[X1_].tolist()).unsqueeze(1)
        X2_ = self.SatCondTemp
        var2 = torch.Tensor(data[X2_].tolist()).unsqueeze(1)

        """종속변수, 타겟 값"""
        TARGET_ = self.target[0]
        tar = torch.Tensor(data[TARGET_].tolist()).unsqueeze(1)

        W1 = torch.zeros(1, requires_grad=True)
        W2 = torch.zeros(1, requires_grad=True)
        W3 = torch.zeros(1, requires_grad=True)
        W4 = torch.zeros(1, requires_grad=True)
        W5 = torch.zeros(1, requires_grad=True)
        W6 = torch.zeros(1, requires_grad=True)

        if self.Method == "Adam" :
            optimizer = optim.Adam([W1, W2, W3, W4, W5, W6], lr=0.001)
        elif self.Method == "RMSprop":
            optimizer = optim.RMSprop([W1, W2, W3, W4, W5, W6], lr=0.001)
        elif self.Method == "SGD":
            optimizer = optim.SGD([W1, W2, W3, W4, W5, W6], lr=0.001)
        else:
            optimizer = optim.Adam([W1, W2, W3, W4, W5, W6], lr=0.001)

        num = 0
        while True:
            #만들고 싶은 회귀식
            hypothesis = (W1
                          + W2 * var1
                          + W3 * var2
                          + W4 * pow(var1, 2)
                          + W5 * pow(var2, 2)
                          + W6 * var1 * var2)

            # Cost : Cv(RMSE)
            cost = torch.mean(pow(hypothesis - tar, 2)) # MSE
            cost = 100 * torch.sqrt(cost) / torch.mean(tar)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if num % 5000 == 0:
                # 변수에 따라서 웨이트 값 출력 조정 필요
                print(
                    '[Iteration : {} / Cost : {:.4f}] W1: {:.4f}, W2: {:.4f}, W3: {:.4f}, W4: {:.4f}, W5: {:.4f}, W6: {:.4f}'
                    .format(num, cost.item(), W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item()))
            if num > self.TrainIter:
                print("Done!")
                acc = round(cost.item(), 2)
                df_performing = pd.DataFrame()
                df_performing.loc[0, "Cv(RMSE)"] = acc
                df_performing.to_csv("{}/CvRMSE(Rating)_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
                break
            num += 1
        coef_list = [W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item()]
        df_coefing = pd.DataFrame()
        df_coefing["Weight"] = coef_list
        df_coefing.to_csv("{}/Coefficient(Rating)_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

        # ParaTuneRating test
        data.reset_index(inplace=True, drop=True)
        num = 0
        while num < data.shape[0]:
            if self.comp_num == 1:
                hypothesis = (coef_list[0]
                              + coef_list[1] * data[self.SatEvapTemp][num]
                              + coef_list[2] * data[self.SatCondTemp][num]
                              + coef_list[3] * pow(data[self.SatEvapTemp][num], 2)
                              + coef_list[4] * pow(data[self.SatCondTemp][num], 2)
                              + coef_list[5] * data[self.SatEvapTemp][num] * data[self.SatCondTemp][num])
            elif self.comp_num == 2:
                hypothesis = (coef_list[0]
                              + coef_list[1] * data[self.SatEvapTemp][num]
                              + coef_list[2] * data[self.SatCondTemp][num]
                              + coef_list[3] * pow(data[self.SatEvapTemp][num], 2)
                              + coef_list[4] * pow(data[self.SatCondTemp][num], 2)
                              + coef_list[5] * data[self.SatEvapTemp][num] * data[self.SatCondTemp][num])
            elif self.comp_num == 3:
                hypothesis = (coef_list[0]
                              + coef_list[1] * data[self.SatEvapTemp][num]
                              + coef_list[2] * data[self.SatCondTemp][num]
                              + coef_list[3] * pow(data[self.SatEvapTemp][num], 2)
                              + coef_list[4] * pow(data[self.SatCondTemp][num], 2)
                              + coef_list[5] * data[self.SatEvapTemp][num] * data[self.SatCondTemp][num])
            data.at[data.index[num], self.RatingPower] = hypothesis
            num += 1

        # 테스트
        self._OutUnitData.reset_index(inplace=True, drop=True)
        num = 0
        while num < self._OutUnitData.shape[0]:
            if self.comp_num == 1:
                hypothesis = (coef_list[0]
                              + coef_list[1] * self._OutUnitData[self.SatEvapTemp][num]
                              + coef_list[2] * self._OutUnitData[self.SatCondTemp][num]
                              + coef_list[3] * pow(self._OutUnitData[self.SatEvapTemp][num], 2)
                              + coef_list[4] * pow(self._OutUnitData[self.SatCondTemp][num], 2)
                              + coef_list[5] * self._OutUnitData[self.SatEvapTemp][num] * self._OutUnitData[self.SatCondTemp][num])
            elif self.comp_num == 2:
                hypothesis = (coef_list[0]
                              + coef_list[1] * self._OutUnitData[self.SatEvapTemp][num]
                              + coef_list[2] * self._OutUnitData[self.SatCondTemp][num]
                              + coef_list[3] * pow(self._OutUnitData[self.SatEvapTemp][num], 2)
                              + coef_list[4] * pow(self._OutUnitData[self.SatCondTemp][num], 2)
                              + coef_list[5] * self._OutUnitData[self.SatEvapTemp][num] * self._OutUnitData[self.SatCondTemp][num])
            elif self.comp_num == 3:
                hypothesis = (coef_list[0]
                              + coef_list[1] * self._OutUnitData[self.SatEvapTemp][num]
                              + coef_list[2] * self._OutUnitData[self.SatCondTemp][num]
                              + coef_list[3] * pow(self._OutUnitData[self.SatEvapTemp][num], 2)
                              + coef_list[4] * pow(self._OutUnitData[self.SatCondTemp][num], 2)
                              + coef_list[5] * self._OutUnitData[self.SatEvapTemp][num] * self._OutUnitData[self.SatCondTemp][num])
            self._OutUnitData.at[self._OutUnitData.index[num], self.RatingPower] = hypothesis
            num += 1
        self._OutUnitData.to_csv("{}/RatingTest_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        return data

    def CalculatingKpvalue(self, data):
        self.kpValue_ref = 'kpValue_ref'

        # ParaTuneRating test
        data.reset_index(inplace=True, drop=True)
        num = 0
        while num < data.shape[0]:
            # Rating == Rated로 놓고 계산
            k_p = data[self.target[0]][num] / data[self.RatingPower][num]
            data.at[data.index[num], "{}".format(self.kpValue_ref)] = k_p
            num += 1

        self._OutUnitData.reset_index(inplace=True, drop=True)
        num = 0
        while num < self._OutUnitData.shape[0]:
            # Rating == Rated로 놓고 계산
            k_p = self._OutUnitData[self.target[0]][num] / self._OutUnitData[self.RatingPower][num]
            self._OutUnitData.at[self._OutUnitData.index[num], "{}".format(self.kpValue_ref)] = k_p
            num += 1
        return data

    def ParaTuneFreq(self, data, save, out_unit):
        self.VirtualPower = 'virtual_power'
        self.powerRated = 'power_rating'
        self.kpValue = 'kpValue'

        if self.comp_num == 1:
            X1_ = self.Compfreq[0]  # comp1
            var1 = torch.Tensor(data[X1_].tolist()).unsqueeze(1)
        elif self.comp_num == 2:
            X1_ = self.Compfreq[1]  # comp2
            var1 = torch.Tensor(data[X1_].tolist()).unsqueeze(1)
        elif self.comp_num == 3:
            X1_ = 'comp_frequency_sum'  # comp1 + comp2
            var1 = torch.Tensor(data[X1_].tolist()).unsqueeze(1)

        """종속변수, 타겟 값"""
        TARGET_ = self.kpValue_ref #!!! 확인하기
        tar = torch.Tensor(data[TARGET_].tolist()).unsqueeze(1)

        W1 = torch.zeros(1, requires_grad=True)
        W2 = torch.zeros(1, requires_grad=True)
        W3 = torch.zeros(1, requires_grad=True)

        if self.Method == "Adam":
            optimizer = optim.Adam([W1, W2, W3], lr=0.001)
        elif self.Method == "RMSprop":
            optimizer = optim.RMSprop([W1, W2, W3], lr=0.001)
        elif self.Method == "SGD":
            optimizer = optim.SGD([W1, W2, W3], lr=0.001)
        else:
            optimizer = optim.Adam([W1, W2, W3], lr=0.001)

        num = 0
        while True:
            # 만들고 싶은 회귀식
            hypothesis = (W1
                          + W2 * (var1 - self.freq_rated)
                          + W3 * pow((var1 - self.freq_rated), 2))

            # Cost : Cv(RMSE)
            cost = torch.mean(pow(hypothesis - tar, 2))  # MSE
            cost = 100 * torch.sqrt(cost) / torch.mean(tar)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if num % 5000 == 0:
                # 변수에 따라서 웨이트 값 출력 조정 필요
                print(
                    '[Iteration : {} / Cost : {:.4f}] W1: {:.4f}, W2: {:.4f}, W3: {:.4f}'
                    .format(num, cost.item(), W1.item(), W2.item(), W3.item()))
            if num > self.TrainIter:
                print("Done!")
                acc = round(cost.item(), 2)
                df_performfreq = pd.DataFrame()
                df_performfreq.loc[0, "Cv(RMSE)"] = acc
                df_performfreq.to_csv("{}/CvRMSE(Freq)_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
                break
            num += 1
        coef_list = [W1.item(), W2.item(), W3.item()]
        df_coeffreq = pd.DataFrame()
        df_coeffreq["Weight"] = coef_list
        df_coeffreq.to_csv("{}/Coefficient(Freq)_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

        num = 0
        while num < data.shape[0]:
            K_P = (coef_list[0]
                  + coef_list[1] * (data[X1_][num] - self.freq_rated)
                  + coef_list[2] * pow((data[X1_][num] - self.freq_rated), 2))

            hypothesis = K_P * data[self.powerRated][num]
            data.at[data.index[num], "{}".format(self.kpValue)] = K_P
            data.at[data.index[num], "{}".format(self.VirtualPower)] = hypothesis
            num += 1

        num = 0
        while num < self._OutUnitData.shape[0]:
            K_P = (coef_list[0]
                  + coef_list[1] * (self._OutUnitData[X1_][num] - self.freq_rated)
                  + coef_list[2] * pow((self._OutUnitData[X1_][num] - self.freq_rated), 2))

            hypothesis = K_P * self._OutUnitData[self.powerRated][num]
            self._OutUnitData.at[self._OutUnitData.index[num], "{}".format(self.kpValue)] = K_P
            self._OutUnitData.at[self._OutUnitData.index[num], "{}".format(self.VirtualPower)] = hypothesis
            num += 1

        self._OutUnitData.to_csv("{}/VirtualPowerSensorTest_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        return data

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
start ='2021-11-01' #데이터 시작시간
end = '2022-02-28' #데이터 끝시간

COMP_MODEL_NAME = 'GB066' # GB052, GB066, GB070, GB080
compValue = 'comp'
freqValue = 'comp_current_frequency'
TdisValue = 'discharge_temp'
TsucValue = 'suction_temp'
PsucValue = 'low_pressure'
PdisValue = 'high_pressure'


# 변수
# X1 = 'frequency'
TARGET = 'value'

#Optimizer
METHOD = 'Adam' #SGD, Adam, RMSprop

#Compressor number
comp_num = 1

# 스케일러 옵션은 끄면 Pass된다.
RVS = REGRESSION_SENSORS(COMP_MODEL_NAME=COMP_MODEL_NAME,
                         TIME=TIME,
                         start=start,
                         end=end)

for outdv in [3066]: # 3067, 3069,3065
    for compN in [1, 2, 3]:
        if outdv != 3065:
            print("Outdoor unit : {} Compressor : {}".format(outdv, compN))
            RVS.PROCESSING(out_unit=outdv, target=TARGET,
                           TsucValue=TsucValue,
                           TdisValue=TdisValue,
                           compValue=compValue,
                           freqValue=freqValue,
                           PsucValue=PsucValue,
                           PdisValue=PdisValue,
                           comp_num =compN,
                           Method=METHOD)
        elif outdv == 3065:
            if compN >= 2:
                break
            else:
                print("Outdoor unit : {} Compressor : {}".format(outdv, compN))
                RVS.PROCESSING(out_unit=outdv, target=TARGET,
                               TsucValue=TsucValue,
                               TdisValue=TdisValue,
                               compValue=compValue,
                               freqValue=freqValue,
                               PsucValue=PsucValue,
                               PdisValue=PdisValue,
                               comp_num=compN,
                               Method=METHOD)

    print("Next : {}".format(datetime.datetime.now()))
print("End : {}".format(datetime.datetime.now()))