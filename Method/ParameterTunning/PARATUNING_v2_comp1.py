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
        self.DischargeTemp = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=TdisValue, case=False)])
        self.CompressorSignal = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=compValue, case=False)])
        self.Compfreq = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=freqValue, case=False)])
        self.SuctionPressure = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=PsucValue, case=False)])
        self.DischargePressure = list(pd.Series(list(self._OutUnitData.columns))[pd.Series(list(self._OutUnitData.columns)).str.contains(pat=PdisValue, case=False)])

        self._OutUnitData['comp_frequency_sum'] = self._OutUnitData[self.Compfreq[0]] + self._OutUnitData[self.Compfreq[1]]
        self._OutUnitData['discharge_temp_avg'] = (self._OutUnitData[self.DischargeTemp[0]] + self._OutUnitData[self.DischargeTemp[1]]) / 2

        save = "{}/ParameterTuning/{}/{}".format(self.SAVE_PATH, self.folder_name, out_unit)
        self.create_folder(save)

        self._OutUnitData.to_csv("{}/Before_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 전

        """필요한 경우 조건을 적용하는 장소이다."""
        self.comp_num = comp_num

        #Frequency Filtering
        if self.comp_num == 1:
            # Mode1 : comp1 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData[self.Compfreq[0]] > 10) # 최소 추파수 값
                                                  & (self._OutUnitData[self.CompressorSignal[0]] != 0)
                                                  & (self._OutUnitData[self.target[0]] > 1000)]  # 작동중인 데이터만 사용
            self.OutIntegData.to_csv("{}/After_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        elif self.comp_num == 2:
            # Mode2 : comp2 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData[self.Compfreq[1]] > 10) # 최소 추파수 값
                                                  & (self._OutUnitData[self.CompressorSignal[1]] != 0)
                                                  & (self._OutUnitData[self.target[0]] > 1000)]  # 작동중인 데이터만 사용
            self.OutIntegData.to_csv("{}/After_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        elif self.comp_num == 3:
            # Mode3 : comp1 + comp2 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData['comp_frequency_sum'] > 10)
                                                  & (self._OutUnitData[self.CompressorSignal[0]] != 0)
                                                  & (self._OutUnitData[self.CompressorSignal[1]] != 0)
                                                  & (self._OutUnitData[self.target[0]] > 1000)]  # 작동중인 데이터만 사용
            self.OutIntegData.to_csv("{}/After_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

        #실외기 데이터 확인용 Plotting
        self.PlottingSystem(save=save, out_unit=out_unit)

        # [Step 1] Rating : 작동 데이터로 파라미터 튜닝 1
        self.Rating_coef = self.ParaTuneRating(data=self.OutIntegData,
                                               save=save, out_unit=out_unit)

        # [Step 2] Rated : ParaTune Rated 안에는 RapaTuneRating 에서 학습한 파라미터를 사용해서 테스트 진행하고 파라미터 튜닝 2
        self.Rated_coef = self.ParaTuneRated(data=self.OutIntegData,
                                             coef=self.Rating_coef,
                                             save=save, out_unit=out_unit)

        # [Step 3] kp > freq
        self.kpTuned_coef = self.ParaTuneFreq(test_data=self.OutIntegData,
                                              coef=self.Rated_coef,
                                              save=save, out_unit=out_unit)

        self.VirtualPowerSensor(data=self.OutIntegData,
                                Ratingcoef=self.Rating_coef,
                                Ratedcoef=self.Rated_coef,
                                Freqcoef=self.kpTuned_coef,
                                save=save,
                                out_unit=out_unit)

    def ParaTuneRating(self, data, save, out_unit):
        """독립변수"""
        X1_ = self.SuctionTemp[0]
        var1 = torch.Tensor(data[X1_].tolist()).unsqueeze(1)
        X2_ = 'discharge_temp_avg'
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
        return coef_list

    def ParaTuneRated(self, data, coef, save, out_unit):
        pd.set_option('mode.chained_assignment', None)
        self.SuctionDensity = 'suction_density'  # 컬럼이름을 바꾼것
        self.ratTemp = 'rated_temp'
        self.ratPress = 'rated_pressure'
        self.ratDensity = 'rated_density'
        self.DensRatio = 'densityRatio'

        # ParaTuneRating test
        data.reset_index(inplace=True, drop=True)
        num = 0
        dens_Suction_prev = 0
        while num < data.shape[0]:
            hypothesis = (coef[0]
                          + coef[1] * data[self.SuctionTemp[0]][num]
                          + coef[2] * data['discharge_temp_avg'][num]
                          + coef[3] * pow(data[self.SuctionTemp[0]][num], 2)
                          + coef[4] * pow(data['discharge_temp_avg'][num], 2)
                          + coef[5] * data[self.SuctionTemp[0]][num] * data['discharge_temp_avg'][num])
            try:
                dens_Suction = CP.CoolProp.PropsSI('D', 'P', data[self.SuctionPressure[0]][num] * 100 * 1000, 'T', data[self.SuctionTemp[0]][num] + 273.15, 'R410A')
                if dens_Suction > 1000:
                    dens_Suction = dens_Suction_prev
                dens_Suction_prev = dens_Suction
            except ValueError:
                dens_Suction = 0

            data.at[data.index[num], "power_rating"] = hypothesis
            data.at[data.index[num], self.SuctionDensity] = dens_Suction
            num += 1
        data.to_csv("{}/RatingTest_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

        if self.comp_num == 1:
            self.freq_rated = float(data[self.Compfreq[0]].mode()[0])
            self.OutRatedData = data[(data[self.Compfreq[0]] == self.freq_rated)]
            print(self.OutRatedData)
            print("Rated frequency", self.freq_rated)
        elif self.comp_num == 2:
            self.freq_rated = float(data[self.Compfreq[1]].mode()[0])
            self.OutRatedData = data[(data[self.Compfreq[1]] == self.freq_rated)]
            print(self.OutRatedData)
            print("Rated frequency", self.freq_rated)
        elif self.comp_num == 3:
            self.freq_rated = float(data['comp_frequency_sum'].mode()[0])
            self.OutRatedData = data[(data['comp_frequency_sum'] == self.freq_rated)]
            print("Rated frequency", self.freq_rated)

        #Rated density
        rated_press = self.OutRatedData[self.SuctionPressure[0]].mean()
        rated_temp = self.OutRatedData[self.SuctionTemp[0]].mean()
        print("[Rated condition] Pressure : {} bar - Temp : {} C".format(round(rated_press, 3), round(rated_temp, 3)))
        dens_rat = CP.CoolProp.PropsSI('D', 'P', rated_press * 100 * 1000, 'T', rated_temp + 273.15, 'R410A')
        print("[Rated condition] Density : {}".format(round(dens_rat, 3)))

        # 학습용 데이터
        self.OutRatedData[self.ratTemp] = rated_temp
        self.OutRatedData[self.ratPress] = rated_press
        self.OutRatedData[self.ratDensity] = dens_rat
        self.OutRatedData[self.DensRatio] = self.OutRatedData[self.SuctionDensity] / self.OutRatedData[self.ratDensity]
        self.OutRatedData.to_csv("{}/RatedFreq_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

        # 테스트용 데이터에도 정보 추가
        self.OutIntegData[self.ratTemp] = dens_rat
        self.OutIntegData[self.ratPress] = dens_rat
        self.OutIntegData[self.ratDensity] = dens_rat
        self.OutIntegData[self.DensRatio] = self.OutIntegData[self.SuctionDensity] / self.OutIntegData[self.ratDensity]

        # X1_ : Rating power
        X1_ = "power_rating" # Rating power
        var1 = torch.Tensor(self.OutRatedData[X1_].tolist()).unsqueeze(1)

        # X2_ : Density Ratio
        X2_ = self.DensRatio #Density Ratio
        var2 = torch.Tensor(self.OutRatedData[X2_].tolist()).unsqueeze(1)

        # Target : Measured power
        TARGET_ = self.target[0] #!!! 확인하기
        tar = torch.Tensor(self.OutRatedData[TARGET_].tolist()).unsqueeze(1)

        W1 = torch.zeros(1, requires_grad=True)

        #Optimizer
        if self.Method == "Adam" :
            optimizer = optim.Adam([W1], lr=0.001)
        elif self.Method == "RMSprop":
            optimizer = optim.RMSprop([W1], lr=0.001)
        elif self.Method == "SGD":
            optimizer = optim.SGD([W1], lr=0.001)
        else:
            optimizer = optim.Adam([W1], lr=0.001)

        num = 0
        while True:
            #만들고 싶은 회귀식
            hypothesis = var1 * (1 + W1 * (var2 - 1))

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
                    '[Iteration : {} / Cost : {:.4f}] W1: {:.4f}'.format(num, cost.item(), W1.item()))
            if num > self.TrainIter:
                print("Done!")
                acc = round(cost.item(), 2)
                df_performed = pd.DataFrame()
                # df_performed["Cv(RMSE)"] = acc
                df_performed.loc[0, "Cv(RMSE)"] = acc
                df_performed.to_csv("{}/CvRMSE(Rated)_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
                break
            num += 1
        coef_list = [W1.item()]
        df_coefed = pd.DataFrame()
        df_coefed["Weight"] = coef_list
        df_coefed.to_csv("{}/Coefficient(Rated)_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        return coef_list

    def ParaTuneFreq(self, test_data, coef, save, out_unit):
        self.kpValue = 'kpValue'
        self.powerRated = 'power_rated'

        #ParaTuneRated test

        test_data.reset_index(inplace=True, drop=True)
        num = 0
        while num < test_data.shape[0]:
            hypothesis = test_data['power_rating'][num] * (1 + coef[0] * (test_data[self.DensRatio][num] - 1))
            k_p = test_data[self.target[0]][num] / hypothesis

            test_data.at[test_data.index[num], "{}".format(self.powerRated)] = hypothesis
            test_data.at[test_data.index[num], "{}".format(self.kpValue)] = k_p
            num += 1
        test_data.to_csv("{}/RatedTest_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        self.OutIntegData = test_data

        if self.comp_num == 1:
            X1_ = self.Compfreq[0]  # comp1
            var1 = torch.Tensor(test_data[X1_].tolist()).unsqueeze(1)
        elif self.comp_num == 2:
            X1_ = self.Compfreq[1]  # comp2
            var1 = torch.Tensor(test_data[X1_].tolist()).unsqueeze(1)
        elif self.comp_num == 3:
            X1_ = 'comp_frequency_sum'  # comp1 + comp2
            var1 = torch.Tensor(test_data   [X1_].tolist()).unsqueeze(1)

        """종속변수, 타겟 값"""
        TARGET_ = self.kpValue #!!! 확인하기
        tar = torch.Tensor(test_data[TARGET_].tolist()).unsqueeze(1)

        W1 = torch.zeros(1, requires_grad=True)
        W2 = torch.zeros(1, requires_grad=True)
        W3 = torch.zeros(1, requires_grad=True)

        if self.Method == "Adam" :
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
        return coef_list

    def VirtualPowerSensor(self, data, Ratingcoef, Ratedcoef, Freqcoef, save, out_unit):
        print("Rating coefficients : {}".format(Ratingcoef))
        print("Rated coefficients : {}".format(Ratedcoef))
        print("Frequency coefficients : {}".format(Freqcoef))
        print("Rated Frequency : {}".format(self.freq_rated))
        self.VirtualPower = 'virtual_power'
        # print(data.columns)

        data.reset_index(inplace=True, drop=True)

        if self.comp_num == 1:
            xValue = self.Compfreq[0]  # comp1
        elif self.comp_num == 2:
            xValue = self.Compfreq[1]  # comp2
        elif self.comp_num == 3:
            xValue = 'comp_frequency_sum'  # comp1 + comp2
        num = 0
        while num < data.shape[0]:
            hypothesis = (Freqcoef[0]
                          + Freqcoef[1] * (data[xValue][num] - self.freq_rated)
                          + Freqcoef[2] * pow((data[xValue][num] - self.freq_rated), 2)) * data[self.powerRated][num]

            data.at[data.index[num], "{}".format(self.VirtualPower)] = hypothesis
            num += 1
        data.to_csv("{}/VirtualPowerSensorTest_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

        self.PlottingRealAndVirtual(data=data, save=save, out_unit=out_unit)

    def PlottingRealAndVirtual(self, data, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = data.dropna()
        print(solve)

        Fault1DateList = ['2022-']
        Fault2DateList = []
        Fault3DateList = []


        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 1, 1)

        if self.comp_num == 1:
            ax1.scatter(solve[self.target[0]], solve[self.VirtualPower], s=80, alpha=0.5, color='steelblue')



            ax1.legend(['Compressor 1\n(Rated frequency : {} ($Hz$))'.format(self.freq_rated)], fontsize=47, ncol=1, loc='upper right')
            ax1.set_xticks([0, 5000, 10000, 15000, 20000])
            ax1.set_yticks([0, 5000, 10000, 15000, 20000])
            ax1.set_xlim([0, 15000])
            ax1.set_ylim([0, 15000])
            ax1.tick_params(axis="x", labelsize=60)
            ax1.tick_params(axis="y", labelsize=60)
            ax1.set_xlabel('Virtual Power($W$)', fontsize=65)
            ax1.set_ylabel('Real Power($W$)', fontsize=65)
            # ax1.set_title("Rated Frequency : {} Hz".format(self.freq_rated), fontsize=65)
            ax1.grid()
            plt.tight_layout()
            plt.savefig("{}/freq1_VirtualPower_Outdoor_{}.png".format(save, out_unit))
            # plt.show()
            plt.clf()
        elif self.comp_num == 2:
            ax1.scatter(solve[self.target[0]], solve[self.VirtualPower], s=80, alpha=0.5, color='cornflowerblue')
            ax1.legend(['Compressor 2\n(Rated frequency : {} ($Hz$))'.format(self.freq_rated)], fontsize=47, ncol=1, loc='upper right')
            ax1.set_xticks([0, 5000, 10000, 15000, 20000])
            ax1.set_yticks([0, 5000, 10000, 15000, 20000])
            ax1.set_xlim([0, 15000])
            ax1.set_ylim([0, 15000])
            ax1.tick_params(axis="x", labelsize=60)
            ax1.tick_params(axis="y", labelsize=60)
            ax1.set_xlabel('Virtual Power($W$)', fontsize=65)
            ax1.set_ylabel('Real Power($W$)', fontsize=65)
            # ax1.set_title("Rated Frequency : {} Hz".format(self.freq_rated), fontsize=65)
            ax1.grid()
            plt.tight_layout()
            plt.savefig("{}/freq2_VirtualPower_Outdoor_{}.png".format(save, out_unit))
            # plt.show()
            plt.clf()
        elif self.comp_num == 3:
            ax1.scatter(solve[self.target[0]], solve[self.VirtualPower], s=80, alpha=0.5, color='royalblue')
            ax1.legend(['Compressor 1 + 2\n(Rated frequency : {} ($Hz$))'.format(self.freq_rated)], fontsize=47, ncol=1, loc='upper right')
            ax1.set_xticks([0, 5000, 10000, 15000, 20000])
            ax1.set_yticks([0, 5000, 10000, 15000, 20000])
            ax1.set_xlim([0, 15000])
            ax1.set_ylim([0, 15000])
            ax1.tick_params(axis="x", labelsize=60)
            ax1.tick_params(axis="y", labelsize=60)
            ax1.set_xlabel('Virtual Power($W$)', fontsize=65)
            ax1.set_ylabel('Real Power($W$)', fontsize=65)
            # ax1.set_title("Rated Frequency : {} Hz".format(self.freq_rated), fontsize=65)
            ax1.grid()
            plt.tight_layout()
            plt.savefig("{}/freq3_VirtualPower_Outdoor_{}.png".format(save, out_unit))
            # plt.show()
            plt.clf()

    def PlottingSystem(self, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self.OutIntegData.fillna(0)

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 1, 1)

        if self.comp_num == 1:
            ax1.scatter(solve[self.Compfreq[0]], solve[self.target], s=80, alpha=0.5, color='steelblue')
            ax1.legend(['Compressor 1'], fontsize=50, ncol=1, loc='upper right')
            ax1.set_xticks([0, 25, 50, 75, 100])
            ax1.set_yticks([0, 1500, 3000, 4500, 6000])
            ax1.set_xlim([0, 100])
            ax1.set_ylim([0, 6000])
            ax1.tick_params(axis="x", labelsize=60)
            ax1.tick_params(axis="y", labelsize=60)
            ax1.set_xlabel('Compressor Frequency($Hz$)', fontsize=65)
            ax1.set_ylabel('Power($W$)', fontsize=65)
            ax1.grid()
            plt.tight_layout()
            plt.savefig("{}/freq1_target_Outdoor_{}.png".format(save, out_unit))
            # plt.show()
            plt.clf()
        elif self.comp_num == 2:
            ax1.scatter(solve[self.Compfreq[1]], solve[self.target], s=80, alpha=0.5, color='cornflowerblue')
            ax1.legend(['Compressor 2'], fontsize=50, ncol=1, loc='upper right')
            ax1.set_xticks([0, 25, 50, 75, 100])
            ax1.set_yticks([0, 1500, 3000, 4500, 6000])
            ax1.set_xlim([0, 100])
            ax1.set_ylim([0, 6000])
            ax1.tick_params(axis="x", labelsize=60)
            ax1.tick_params(axis="y", labelsize=60)
            ax1.set_xlabel('Compressor Frequency($Hz$)', fontsize=65)
            ax1.set_ylabel('Power($W$)', fontsize=65)
            ax1.grid()
            plt.tight_layout()
            plt.savefig("{}/freq2_target_Outdoor_{}.png".format(save, out_unit))
            # plt.show()
            plt.clf()
        elif self.comp_num == 3:
            ax1.scatter(solve['comp_frequency_sum'], solve[self.target], s=80, alpha=0.5, color='royalblue')
            ax1.legend(['Compressor 1 & 2'], fontsize=50, ncol=1, loc='upper right')
            ax1.set_xticks([0, 50, 100, 150, 200])
            ax1.set_yticks([0, 1500, 3000, 4500, 6000])
            ax1.set_xlim([0, 200])
            ax1.set_ylim([0, 6000])
            ax1.tick_params(axis="x", labelsize=60)
            ax1.tick_params(axis="y", labelsize=60)
            ax1.set_xlabel('Compressor Frequency($Hz$)', fontsize=65)
            ax1.set_ylabel('Power($W$)', fontsize=65)
            ax1.grid()
            plt.tight_layout()
            plt.savefig("{}/freq3_target_Outdoor_{}.png".format(save, out_unit))
            # plt.show()
            plt.clf()

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

for outdv in [3066]: #, 3065, 3067, 3069]:
    for compN in [1, 2, 3]:
        RVS.PROCESSING(out_unit=outdv, target=TARGET,
                       TsucValue=TsucValue,
                       TdisValue=TdisValue,
                       compValue=compValue,
                       freqValue=freqValue,
                       PsucValue=PsucValue,
                       PdisValue=PdisValue,
                       comp_num =compN,
                       Method=METHOD)

    print("Next : {}".format(datetime.datetime.now()))
print("End : {}".format(datetime.datetime.now()))