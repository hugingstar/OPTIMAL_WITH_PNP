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
torch.cuda.is_available()
class REGRESSION_SENSORS():
    def __init__(self,COMP_MODEL_NAME, freq_rated, TIME, start, end):
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
        self.freq_rated = freq_rated

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

    def PROCESSING(self, out_unit, target, TdisValue, TsucValue, compValue, freqValue, Method):
        # 예측 대상
        self.target = target
        self.Method = Method

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
        self.target = list(pd.Series(list(self._OutUnitData.columns))[
                                    pd.Series(list(self._OutUnitData.columns)).str.contains(pat=target, case=False)])
        self.SuctionTemp = list(pd.Series(list(self._OutUnitData.columns))[
                                    pd.Series(list(self._OutUnitData.columns)).str.contains(pat=TsucValue, case=False)])
        self.DischargeTemp = list(pd.Series(list(self._OutUnitData.columns))[
                                      pd.Series(list(self._OutUnitData.columns)).str.contains(pat=TdisValue, case=False)])
        self.CompressorSignal = list(pd.Series(list(self._OutUnitData.columns))[
                                      pd.Series(list(self._OutUnitData.columns)).str.contains(pat=compValue, case=False)])
        self.Compfreq = list(pd.Series(list(self._OutUnitData.columns))[
                                      pd.Series(list(self._OutUnitData.columns)).str.contains(pat=freqValue, case=False)])

        self._OutUnitData['comp_frequency_sum'] = self._OutUnitData[self.Compfreq[0]] + self._OutUnitData[self.Compfreq[1]]
        self._OutUnitData['discharge_temp_avg'] = (self._OutUnitData[self.DischargeTemp[0]] + self._OutUnitData[self.DischargeTemp[1]])/2


        save = "{}/ParameterTuning/{}/{}".format(self.SAVE_PATH, self.folder_name, out_unit)
        self.create_folder(save)

        self._OutUnitData.to_csv("{}/Before_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 전

        """필요한 경우 조건을 적용하는 장소이다."""
        self.comp_num = 3

        if self.comp_num == 1:
            # Mode1 : comp1 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData[self.Compfreq[0]] > 15)
                                                  & (self._OutUnitData[self.Compfreq[0]] < 90)
                                                  & (self._OutUnitData[self.CompressorSignal[0]] != 0)] #작동중인 데이터만 사용
            self.range = {
                "Band1": [15, 30],
                "Band2": [30, 45],
                "Band3": [45, 60],
                "Band4": [60, 75],
                "Band5": [75, 90]
            }

            df_coef = pd.DataFrame(columns=self.range.keys())
            df_perform = pd.DataFrame(columns=self.range.keys())
            df_banded = pd.DataFrame()
            for i in range(len(self.range)):
                low_boundary = self.range[list(self.range.keys())[i]][0]
                high_boundary = self.range[list(self.range.keys())[i]][1]
                self.OutIntegData_Banded = self.OutIntegData[(self.OutIntegData[self.Compfreq[0]] > low_boundary)
                                                             & (self.OutIntegData[self.Compfreq[0]] < high_boundary)]

                self.OutIntegData_Banded = self.OutIntegData_Banded[[self.Compfreq[0], self.Compfreq[1],
                                                                     'comp_frequency_sum',
                                                                     self.CompressorSignal[0], self.CompressorSignal[1],
                                                                     self.SuctionTemp[0],
                                                                     self.DischargeTemp[0], self.DischargeTemp[1],
                                                                     'discharge_temp_avg',
                                                                     self.target[0]]]
                avg = (low_boundary + high_boundary) / 2
                self.OutIntegData_Banded['Band_Frequency'] = avg
                self.OutIntegData_Banded.to_csv("{}/Band{}_Outdoor_{}_{}.csv".format(save, int(i) + 1, out_unit, self.comp_num))

                df_banded = pd.concat([df_banded, self.OutIntegData_Banded], axis=0)
                self.tune_info = self.ParaTune(data=self.OutIntegData_Banded)
                df_coef["Band{}".format(int(i) + 1)] = self.tune_info[0]
                df_perform["Band{}".format(int(i) + 1)] = self.tune_info[1]
            df_coef.to_csv("{}/Coefficient_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            df_perform.to_csv("{}/CvRMSE_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            df_banded.to_csv("{}/After_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            self.df_banded = df_banded

        elif self.comp_num == 2:
            # Mode2 : comp2 사용
            self.OutIntegData = self._OutUnitData[(self._OutUnitData[self.Compfreq[1]] > 15)
                                                  & (self._OutUnitData[self.Compfreq[1]] < 90)
                                                  & (self._OutUnitData[self.CompressorSignal[1]] != 0)] #작동중인 데이터만 사용
            self.range = {
                "Band1": [15, 30],
                "Band2": [30, 45],
                "Band3": [45, 60],
                "Band4": [60, 75],
                "Band5": [75, 90]
            }

            df_coef = pd.DataFrame(columns=self.range.keys())
            df_perform = pd.DataFrame(columns=self.range.keys())
            df_banded = pd.DataFrame()
            for i in range(len(self.range)):
                low_boundary = self.range[list(self.range.keys())[i]][0]
                high_boundary = self.range[list(self.range.keys())[i]][1]
                self.OutIntegData_Banded = self.OutIntegData[(self.OutIntegData[self.Compfreq[1]] > low_boundary)
                                                             & (self.OutIntegData[self.Compfreq[1]] < high_boundary)]

                self.OutIntegData_Banded = self.OutIntegData_Banded[[self.Compfreq[0], self.Compfreq[1],
                                                                     'comp_frequency_sum',
                                                                     self.CompressorSignal[0], self.CompressorSignal[1],
                                                                     self.SuctionTemp[0],
                                                                     self.DischargeTemp[0], self.DischargeTemp[1],
                                                                     'discharge_temp_avg',
                                                                     self.target[0]]]
                avg = (low_boundary + high_boundary) / 2
                self.OutIntegData_Banded['Band_Frequency'] = avg
                self.OutIntegData_Banded.to_csv("{}/Band{}_Outdoor_{}_{}.csv".format(save, int(i) + 1, out_unit, self.comp_num))

                df_banded = pd.concat([df_banded, self.OutIntegData_Banded], axis=0)
                self.tune_info = self.ParaTune(data=self.OutIntegData_Banded)
                df_coef["Band{}".format(int(i) + 1)] = self.tune_info[0]
                df_perform["Band{}".format(int(i) + 1)] = self.tune_info[1]
            df_coef.to_csv("{}/Coefficient_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            df_perform.to_csv("{}/CvRMSE_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            df_banded.to_csv("{}/After_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            self.df_banded = df_banded

        elif self.comp_num == 3:
            # Mode3 : comp1 + comp2
            self.OutIntegData = self._OutUnitData[(self._OutUnitData['comp_frequency_sum'] > 15)
                                                  & (self._OutUnitData['comp_frequency_sum'] < 165)
                                                  & (self._OutUnitData[self.CompressorSignal[0]] != 0)
                                                  & (self._OutUnitData[self.CompressorSignal[1]] != 0)] #작동중인 데이터만 사용
            self.range = {
                "Band1": [15, 45],
                "Band2": [45, 75],
                "Band3": [75, 105],
                "Band4": [105, 135],
                "Band5": [135, 165]
            }

            df_coef = pd.DataFrame(columns=self.range.keys())
            df_perform = pd.DataFrame(columns=self.range.keys())
            df_banded = pd.DataFrame()
            for i in range(len(self.range)):
                low_boundary = self.range[list(self.range.keys())[i]][0]
                high_boundary = self.range[list(self.range.keys())[i]][1]
                self.OutIntegData_Banded = self.OutIntegData[(self.OutIntegData['comp_frequency_sum'] > low_boundary)
                                                             & (self.OutIntegData['comp_frequency_sum'] < high_boundary)]

                self.OutIntegData_Banded = self.OutIntegData_Banded[[self.Compfreq[0], self.Compfreq[1],
                                                                     'comp_frequency_sum',
                                                                     self.CompressorSignal[0], self.CompressorSignal[1],
                                                                     self.SuctionTemp[0],
                                                                     self.DischargeTemp[0], self.DischargeTemp[1],
                                                                     'discharge_temp_avg',
                                                                     self.target[0]]]
                avg = (low_boundary + high_boundary) / 2
                self.OutIntegData_Banded['Band_Frequency'] = avg
                self.OutIntegData_Banded.to_csv("{}/Band{}_Outdoor_{}_{}.csv".format(save, int(i) + 1, out_unit, self.comp_num))

                df_banded = pd.concat([df_banded, self.OutIntegData_Banded], axis=0)
                self.tune_info = self.ParaTune(data=self.OutIntegData_Banded)
                df_coef["Band{}".format(int(i) + 1)] = self.tune_info[0]
                df_perform["Band{}".format(int(i) + 1)] = [self.tune_info[1]]
            df_coef.to_csv("{}/Coefficient_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            df_perform.to_csv("{}/CvRMSE_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            df_banded.to_csv("{}/After_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
            self.df_banded = df_banded

        #실외기 데이터 확인용 Plotting
        self.PlottingSystem(save=save, out_unit=out_unit)
        self.PlottingBanded(save=save, out_unit=out_unit)

        self.ComputeKvalue(data=self.df_banded, save=save, out_unit=out_unit)

        df_coef_freq = pd.DataFrame()
        df_perform_freq = pd.DataFrame()
        self.tune_info_freq = self.ParaTuneFreq(data=self.IntegFreq)
        df_coef_freq["Parameters"] = self.tune_info_freq[0]
        df_perform_freq["Cv(RMSE)"] = self.tune_info_freq[1]
        df_coef_freq.to_csv("{}/Coefficient_freq_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        df_perform_freq.to_csv("{}/CvRMSE_freq_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))

    def ComputeKvalue(self, data, save, out_unit):
        """
        K를 구하기 위해서는 온도 Range가 모두 똑같아야 한다.
        1.Min Max 값으로 전체 Range를 지정
        2. 튜닝된 파라미터 값을 사용하여 예측한 값으로 비율을 계산
        Example)
            [Band1] Tc
            Te    A   ==> k = A/C

            [Band2] Tc
            Te    B   ==> k = B/C

            [Band3] Tc
            Te    C   ==> k = C/C = 1

            [Band4] Tc
            Te    D   ==> k = D/C

            [Band5] Tc
            Te    E   ==> k = E/C
        """
        #데이터 정보를 확인
        # Suction_MIN =int(min(data[self.SuctionTemp[0]]))
        # Suction_MAX = int(max(data[self.SuctionTemp[0]]))
        # Discharge_MIN = int(min(data['discharge_temp_avg']))
        # Discharge_MAX = int(max(data['discharge_temp_avg']))
        # print("Suction Line [Low Temp] {} [High Temp] {} // Discharge Line [Low Temp] {} [High Temp] {}".format(Suction_MIN, Suction_MAX, Discharge_MIN, Discharge_MAX))

        #Kp값을 계산하기 위한 입력 데이터
        sucls = data[self.SuctionTemp[0]].astype(int).unique().tolist() # 중복값 제거
        disls = data['discharge_temp_avg'].astype(int).unique().tolist()
        Suction_Range = sorted(sucls, reverse=False)
        Discharge_Range = sorted(disls, reverse=False)
        # print("Suction Range : {}".format(Suction_Range))
        # print("Discharge Range : {}".format(Discharge_Range))

        coef_df = pd.read_csv("{}/Coefficient_Comp{}_Outdoor_{}.csv".format(save, self.comp_num, out_unit))
        for k in range(len(self.range)):
            matrix_ = np.zeros([int(len(Suction_Range)), int(len(Discharge_Range))])
            print("Band{} Matric Shape : {}".format(k + 1, matrix_.shape))
            coef_list = coef_df['Band{}'.format(k + 1)].tolist()
            for i in range(len(Suction_Range)):
                for j in range(len(Discharge_Range)):
                    hypothesis = (coef_list[0] * Suction_Range[i]
                                  + coef_list[1] * pow(Suction_Range[i], 2)
                                  + coef_list[2] * pow(Suction_Range[i], 3)
                                  + coef_list[3] * Discharge_Range[j]
                                  + coef_list[4] * pow(Discharge_Range[j], 2)
                                  + coef_list[5] * pow(Discharge_Range[j], 3)
                                  + coef_list[6] * Suction_Range[i] * Discharge_Range[j]
                                  + coef_list[7] * Suction_Range[i] * pow(Discharge_Range[j], 2)
                                  + coef_list[8] * pow(Suction_Range[i], 2) * Discharge_Range[j]
                                  + coef_list[9] * pow(Suction_Range[i], 2) * pow(Discharge_Range[j], 2))
                    matrix_[i][j] = hypothesis
            df_band_matrix = pd.DataFrame(matrix_, index=Suction_Range, columns=Discharge_Range)
            df_band_matrix.to_csv("{}/Band{}_Matrix_Outdoor_{}_{}.csv".format(save, int(k+1), out_unit, self.comp_num))

        #Compute Kvalue
        df_Kvalue = pd.DataFrame()
        Ref_Matrix = pd.read_csv("{}/Band{}_Matrix_Outdoor_{}_{}.csv".format(save, 3, out_unit, self.comp_num)).to_numpy()
        for k in range(len(self.range)):
            Tar_Matrix = pd.read_csv("{}/Band{}_Matrix_Outdoor_{}_{}.csv".format(save, int(k + 1), out_unit, self.comp_num)).to_numpy()
            matrix_ = np.zeros([int(len(Suction_Range)), int(len(Discharge_Range))])
            for i in range(len(Suction_Range)):
                for j in range(len(Discharge_Range)):
                    # print(Tar_Matrix[i][j], Ref_Matrix[i][j])
                    matrix_[i][j] = Tar_Matrix[i][j] / Ref_Matrix[i][j]
            df_Kval_matrix = pd.DataFrame(matrix_, index=Suction_Range, columns=Discharge_Range)
            df_Kval_matrix.to_csv("{}/Band{}_Kvalue_Outdoor_{}_{}.csv".format(save, int(k+1), out_unit, self.comp_num))
            df_Kvalue = pd.concat([df_Kvalue, df_Kval_matrix], axis=0)
        df_Kvalue.to_csv("{}/After_KValue_Outdoor_{}_{}.csv".format(save, out_unit, self.comp_num))

        #Concat Kvalue
        df_mat = pd.DataFrame()
        for k in range(len(self.range)):
            _Matrix = pd.read_csv("{}/Band{}_Matrix_Outdoor_{}_{}.csv".format(save, int(k + 1), out_unit, self.comp_num))
            val = sum(self.range['Band{}'.format(str(k+1))])/len(self.range['Band{}'.format(str(k+1))])
            _Matrix['Band_Frequency'] = val
            df_mat = pd.concat([df_mat, _Matrix], axis=0)
        df_mat.set_index('Unnamed: 0', inplace=True) #Suction Temp
        df_mat.to_csv("{}/After_Matrix_Outdoor_{}_{}.csv".format(save, out_unit, self.comp_num))

        Value_array = df_mat.drop(columns='Band_Frequency').to_numpy()
        Kvalue_array =df_Kvalue.to_numpy()
        self.IntegFreq = pd.DataFrame(columns=['suction_temp', 'discharge_temp', 'frequency', 'value', 'Kvalue'])
        col_list = list(df_mat.columns) # Discharge Temp
        col_list.remove('Band_Frequency')
        index_list = list(df_mat.index) # Suction Temp
        # freq_list = df_mat['Band_Frequency']
        # df_mat.drop('Band_Frequency')
        num = 0
        for i in range(len(col_list)):
            for j in range(len(index_list)):
                self.IntegFreq.at[num, "discharge_temp"] = col_list[i]
                self.IntegFreq.at[num, "suction_temp"] = index_list[j]
                self.IntegFreq.at[num, "frequency"] = df_mat['Band_Frequency'].tolist()[j]
                self.IntegFreq.at[num, "value"] = Value_array[j][i]
                self.IntegFreq.at[num, "Kvalue"] = Kvalue_array[j][i]
                num += 1
        self.IntegFreq.to_csv("{}/IntegFreq_Outdoor_{}_{}.csv".format(save, out_unit, self.comp_num))

    def ParaTuneFreq(self, data):
        """독립변수"""
        F = 'frequency'
        var1 = torch.Tensor(data[F].tolist()).unsqueeze(1)

        low_boundary = self.range[list(self.range.keys())[2]][0]
        high_boundary = self.range[list(self.range.keys())[2]][1]
        avg = (low_boundary + high_boundary) / 2
        print(avg)
        rated_f = torch.Tensor([avg]).unsqueeze(1)

        """종속변수, 타겟 값"""
        TARGET_ = 'Kvalue'
        tar = torch.Tensor(data[TARGET_].tolist()).unsqueeze(1)

        W1 = torch.zeros(1, requires_grad=True)
        W2 = torch.zeros(1, requires_grad=True)
        W3 = torch.zeros(1, requires_grad=True)

        if self.Method == "Adam" :
            optimizer = optim.Adam([W1, W2, W3], lr=0.0001)
        elif self.Method == "RMSprop":
            optimizer = optim.RMSprop([W1, W2, W3], lr=0.0001)
        elif self.Method == "SGD":
            optimizer = optim.SGD([W1, W2, W3], lr=0.0001)
        else:
            optimizer = optim.Adam([W1, W2, W3], lr=0.0001)

        num = 0
        while True:
            # 만들고 싶은 회귀식
            hypothesis = (W1 + W2 * (var1 - rated_f) + W3 * pow(var1 - rated_f, 2))

            # Cost : Cv(RMSE)
            cost = torch.mean(pow(hypothesis - tar, 2))  # MSE
            cost = 100 * torch.sqrt(cost) / torch.mean(tar)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if num % 20000 == 0:
                # 변수에 따라서 웨이트 값 출력 조정 필요
                print(
                    '[Iteration : {} / Cost : {:.4f}] W1: {:.4f}, W2: {:.4f}, W3: {:.4f}'.format(num, cost.item(), W1.item(), W2.item(), W3.item()))
            if (cost.item() < 30) | num > 1000000:
                print("Done!")
                acc = float(cost.item())
                break
            num += 1
        coef_list = [W1.item(), W2.item(), W3.item()]
        return coef_list, acc


    def PlottingBanded(self, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self.df_banded.fillna(0)

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.scatter(solve['Band_Frequency'], solve[self.target], s=80, alpha=0.5, color='steelblue')
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
        plt.savefig("{}/freqBanded_target_Outdoor_{}_{}.png".format(save, out_unit, self.comp_num))
        # plt.show()
        plt.clf()

    def ParaTune(self, data):
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
        W7 = torch.zeros(1, requires_grad=True)
        W8 = torch.zeros(1, requires_grad=True)
        W9 = torch.zeros(1, requires_grad=True)
        W10 = torch.zeros(1, requires_grad=True)

        # print("Optimizer : {}".format(self.Method))

        if self.Method == "Adam" :
            optimizer = optim.Adam([W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.001)
        elif self.Method == "RMSprop":
            optimizer = optim.RMSprop([W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.001)
        elif self.Method == "SGD":
            optimizer = optim.SGD([W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.001)
        else:
            optimizer = optim.Adam([W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], lr=0.001)

        num = 0
        while True:
            #만들고 싶은 회귀식
            hypothesis = (W1 * var1
                          + W2 * var1 ** 2
                          + W3 * var1 ** 3
                          + W4 * var2
                          + W5 * var2**2
                          + W6 * var2**3
                          + W7 * var1 * var2
                          + W8 * var1 * var2**2
                          + W9 * var1**2 * var2
                          + W10 * var1**2 * var2**2)

            # Cost : Cv(RMSE)
            cost = torch.mean(pow(hypothesis - tar, 2)) # MSE
            cost = 100 * torch.sqrt(cost) / torch.mean(tar)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if num % 20000 == 0:
                # 변수에 따라서 웨이트 값 출력 조정 필요
                print('[Iteration : {} / Cost : {:.4f}] W1: {:.4f}, W2: {:.4f}, W3: {:.4f}, W4: {:.4f}, W5: {:.4f}, W6: {:.4f}, W7: {:.4f}, W8: {:.4f}, W9: {:.4f}, W10: {:.4f}'
                    .format(num, cost.item(), W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item(), W7.item(), W8.item(), W9.item(), W10.item()))
            if (cost.item() < 30) | num > 100000:
                print("Done!")
                acc = float(cost.item())
                break
            num += 1
        coef_list = [W1.item(), W2.item(), W3.item(), W4.item(), W5.item(), W6.item(), W7.item(), W8.item(), W9.item(), W10.item()]
        return coef_list, acc

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


# 변수
# X1 = 'frequency'
TARGET = 'value'
freq_rated = 90

#Optimizer
METHOD = 'Adam' #SGD, Adam, RMSprop

# 스케일러 옵션은 끄면 Pass된다.
RVS = REGRESSION_SENSORS(COMP_MODEL_NAME=COMP_MODEL_NAME,
                         freq_rated=freq_rated,
                         TIME=TIME,
                         start=start,
                         end=end)

for outdv in [3066, 3065, 3067, 3069]:
    RVS.PROCESSING(out_unit=outdv, target=TARGET,
                   TsucValue=TsucValue,
                   TdisValue=TdisValue,
                   compValue=compValue,
                   freqValue=freqValue,
                   Method=METHOD)
    print("Next : {}".format(datetime.datetime.now()))

print("End : {}".format(datetime.datetime.now()))