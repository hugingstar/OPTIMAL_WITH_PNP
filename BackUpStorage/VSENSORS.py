import os
import pandas as pd
import math
import CoolProp as CP
import numpy as np

class COMPRESSORMAPMODEL():
    def __init__(self, COMP_MODEL_NAME, TIME, start, end):
        "파일을 호출할 경로"
        self.DATA_PATH = "/Data/VirtualSensor"
        self.SAVE_PATH = "/Results"
        self.TIME = TIME

        # 진리관
        # 실외기-실내기
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
            3065: [3109, 3100, 3095, 3112, 3133, 3074, 3092, 3105, 3091, 3124,
                   3071, 3072, 3123, 3125, 3106, 3099, 3081, 3131, 3094, 3084],
            3069: [3077, 3082, 3083, 3089, 3096, 3104, 3110, 3117, 3134, 3102,
                   3116, 3129, 3090],
            3066: [3085, 3086, 3107, 3128, 3108, 3121],
            3067: [3075, 3079, 3080, 3088, 3094, 3101, 3111, 3114, 3115, 3119,
                   3120, 3122, 3130]
        }

        #Coefficient Library
        self.coefdict = {
            "GB052":
                {"Mdot_rated" :[1.34583367e+0, 1.30612964e+0, 8.16473285e-2, 6.64829686e-5, 1.71071419e+1,
                                -3.2517798e-1, 2.0076971000e-3, 3.35711090e-1, -3.11725301e-3, 6.35877854e-4, 2.06143118e-13],
                 "Mdot_pred":[9.98623382E-1 * 0.30, 1.70178410E-2 * 0.30, -1.85413544E-5 * 0.30],
                 "Wdot_rated": [3.51846146e+0, 3.57686903e+0, 6.43560572e-1, 1.50045118e-5, 3.07724735e+1,
                                1.96728924e+0, -1.55914878e-2, -4.89411467e-2, -7.434599943e-4, -9.79711966e-3, 2.82415926e-12],
                 "Wdot_pred":[9.99818595E-01 * 0.30, 2.11652803E-02 * 0.30, 1.17252618E-04 * 0.30]},
            "GB066":
                {"Mdot_rated": [1.91778583E+0, 3.28857174E+0, 1.84065620E-1, 7.14011551E-5, 2.10278731E+1,
                                -3.92042237E-1, 2.38168548E-3, 3.65647991E-1, -3.43302726E-3, -5.06182999E-4, -1.49453769E-13],
                 "Mdot_pred": [9.90958566E-1, 1.66658435E-2, -1.91782998E-5],
                 "Wdot_rated": [4.68221681E+0, 2.89315135E+1, 5.08822631E-1, -2.52904377E-6, 3.72538174E+1,
                                2.52480352E+0, -1.98829304E-2, -6.79818927E-1, 1.96893378E-3, -3.26935360E-3, -2.85508042E-12],
                 "Wdot_pred": [6.5629020e-02, 1.3365647e-03, 3.4921488e-06]},
            "GB070":
                {"Mdot_rated": [3.45223093E+0, 9.58731730E+0, 2.65999052E-1, 4.99983074E-5, 2.61357700E+1,
                                -4.95946275E-1, 3.07594686E-3, 2.45668661E-1, -2.42475689E-3, -1.05368068E-3, -3.92888272E-13],
                 "Mdot_pred": [8.73877476E-1 * 0.24, 1.42782414E-2 * 0.24, -1.66033155E-5 * 0.24],
                 "Wdot_rated": [4.68221681E+0, 2.89315135E+1, 5.08822631E-1, -2.52904377E-6, 3.72538174E+1,
                                2.52480352E+0, -1.98829304E-2, -6.79818927E-1, 1.96893378E-3, -3.26935360E-3, -2.85508042E-12],
                 "Wdot_pred": [1.03655761E+0 * 0.24, 2.18790914E-02 * 0.24, 1.20530958E-04 * 0.24]},
            "GB080":
                {"Mdot_rated": [2.94692558E+0, 8.09167658E+0, 2.62344701E-1, 5.32184444E-5, 2.59520736E+1,
                                -4.81374673E-1, 2.88635012E-3, 2.94984013E-1, -2.80522711E-3, -1.41173400E-3, 1.58730325E-13],
                 "Mdot_pred": [1.00362719E-0, 1.75723734E-2, -6.37848299E-5],
                 "Wdot_rated": [2.65608412E+0, 2.81012263E+1, -2.82198326E-0, -2.80384826E-4, 2.81501305E+1,
                                3.50317115E+0, -2.70107151E-2, -6.07810636E-1, 2.41499759E-3, 6.42414150E-3, 3.60750696E-12],
                 "Wdot_pred": [9.97164782E-1 * 0.24, 2.10053578E-2 * 0.24, 6.99834241E-5 * 0.24]},

        }
        self.coef = self.coefdict[COMP_MODEL_NAME] # Coefficient Model 결정
        self.RatedFrequency = 58

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
        self.create_folder('{}/VirtualSensor'.format(self.SAVE_PATH))  # Deepmodel 폴더를 생성

    def VSENSOR_PROCESSING(self, out_unit, freqValue, PdisValue, PsucValue, TsucValue,
                           TdisValue, TcondOutValue, TliqValue, TinAirValue, ToutAirValue, target):
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

        for indv in list(self.bldginfo[out_unit]):
            #실내기 데이터
            self._indpath = "{}/{}/{}/Outdoor_{}_Indoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit, out_unit, indv)
            self._indata = pd.read_csv(self._indpath, index_col=self.TIME)

            #실내기 및 실외기의 데이터 통합
            self.data = pd.concat([self._outdata, self._indata], axis=1)
            self.data.index.names = [self.TIME] # 인덱스 컬럼명이 없는 경우를 대비하여 보완
            #문자열로 된 원본 데이터의 '모드'를 숫자로 변환
            self.data = self.data.replace({"High": 3, "Mid" : 2, "Low" : 1, "Auto" : 3.5})

            # 컬럼이름 탐색
            self.freq = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=freqValue, case=False)])[0]
            self.DischargePressure = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=PdisValue, case=False)])[0]
            self.DischargeTemp = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=TdisValue, case=False)])[0]
            self.SuctionPressure = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=PsucValue, case=False)])[0]
            self.SuctionTemp = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=TsucValue, case=False)])[0]
            self.CondOutTemp = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=TcondOutValue, case=False)])[0]
            self.LiqTemp = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=TliqValue, case=False)])[0]
            self.AirSideInletTemp = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=TinAirValue, case=False)])[0]
            self.AirSideOutletTemp = list(pd.Series(list(self.data.columns))[pd.Series(list(self.data.columns)).str.contains(pat=ToutAirValue, case=False)])[0]

            # 가상센서 결과를 저장하기 위한 컬럼명 생성
            self.CondMapTemp = self.DischargePressure.replace(PdisValue, 'cond_map_temp') # 맵에서 찾은 데이터 컬럼이름 조정
            self.EvaMapTemp = self.SuctionPressure.replace(PsucValue, 'evap_map_temp')

            self.MdotRated = self.DischargePressure.replace(PdisValue, 'm_dot_rated')
            self.WdotRated = self.DischargePressure.replace(PdisValue, 'w_dot_rated')
            self.MdotPred = self.DischargePressure.replace(PdisValue, 'm_dot_pred')
            self.WdotPred = self.DischargePressure.replace(PdisValue, 'w_dot_pred')

            self.Qevap = self.DischargePressure.replace(PdisValue, 'capa_eva')
            self.Qcond = self.DischargePressure.replace(PdisValue, 'capa_cond')

            self.hCondOut = self.DischargePressure.replace(PdisValue, 'enthalpy_cond_out')
            self.hsuc = self.DischargePressure.replace(PdisValue, 'enthalpy_suc')
            self.hdis = self.DischargePressure.replace(PdisValue, 'enthalpy_dis')

            self.del_h_evap = self.DischargePressure.replace(PdisValue, 'enthalpy_difference_evap')
            self.del_h_cond = self.DischargePressure.replace(PdisValue, 'enthalpy_difference_cond')

            self.ua = self.DischargePressure.replace(PdisValue, 'ua')
            self.cp = self.DischargePressure.replace(PdisValue, 'cp')

            # 컬럼이 두개인것을 모두 활용해야 하므로 아래 과정 추가
            #Frequency
            if self.freq[-1] == '1': #frequency1
                self.freq_sub = self.freq.replace('1', '2')
            elif self.freq[-1] == '2': #frequency2
                self.freq_sub = self.freq.replace('2', '1')

            #DischargeTemperature
            if self.DischargeTemp[-1] == '1':
                self.DischargeTemp_sub = self.DischargeTemp.replace('1', '2')
            elif self.DischargeTemp[-1] == '2':
                self.DischargeTemp_sub = self.DischargeTemp.replace('2', '1')

            #Coefficients
            self.coef_mdotrated = list(self.coef['Mdot_rated'])
            self.coef_wdotrated = list(self.coef['Wdot_rated'])
            self.coef_mdotpred = list(self.coef['Mdot_pred'])
            self.coef_wdotpred = list(self.coef['Mdot_pred'])
            print("MdotRated Coefficients : {} - {}".format(len(self.coef_mdotrated), self.coef_mdotrated))
            print("WdotRated Coefficients : {} - {}".format(len(self.coef_wdotrated), self.coef_wdotrated))
            print("MdotPred Coefficients : {} - {}".format(len(self.coef_mdotpred), self.coef_mdotpred))
            print("WdotPred Coefficients : {} - {}".format(len(self.coef_wdotpred), self.coef_wdotpred))

            #Map data
            for o in range(self.data.shape[0]):
                # MapDensity()
                self.data.at[self.data.index[o], "{}".format(self.MapDensity)] = CP.CoolProp.PropsSI('D', 'P', self.data[self.SuctionPressure][o] * 98.0665 * 1000, 'T', self.data[self.SuctionTemp][o] + 273.15, 'R410A')
                # EvaMapTemp(Celcius) Quality : 1
                self.data.at[self.data.index[o], "{}".format(self.EvaMapTemp)] = CP.CoolProp.PropsSI('T', 'P', self.data[self.SuctionPressure][o] * 98.0665 * 1000, 'Q', 0.5, 'R410A') - 273.15
                # CondMapTemp(Celcius) Quality : 1
                self.data.at[self.data.index[o], "{}".format(self.CondMapTemp)] = CP.CoolProp.PropsSI('T', 'P', self.data[self.DischargePressure][o] * 98.0665 * 1000, 'Q', 0.5, 'R410A') - 273.15


            # 정격 냉매 질량 유량
            self.VSENS_MdotRated()
            # 냉매 질량 유량 가상센서
            self.VSENS_MdotPred()
            # 정격 전력
            self.VSENS_WdotRated()
            # 전력 가상센서
            self.VSENS_WdotPred()
            # 증발기 용량 가상센서
            self.VSENS_CAPA_EVAP()
            # 컨덴서 용량 가상센서
            self.VSENS_CAPA_COND()
            # UA 가상센서
            self.VSENS_UA()

            self.data = self.data.sort_values(self.TIME)
            self.data = self.data.fillna(method='ffill')  # 결측값 처리

            # 저장할 총 경로
            save = "{}/VirtualSensor/{}/{}/{}".format(self.SAVE_PATH, self.target, self.folder_name, out_unit)
            self.create_folder(save)
            self.data.to_csv("{}/Before_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))  # 조건 적용 전

            """필요한 경우 조건을 적용하는 장소이다."""
            # self.data = self.data[self.data[self.onoffsignal] == 1] #작동중인 데이터만 사용
            # self.data = self.data.dropna(axis=0) # 결측값을 그냥 날리는 경우
            self.data.to_csv("{}/After_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))  # 조건 적용 후

    def VSENS_MdotRated(self):
        #Virtual Sensors : m_dot_rated
        for o in range(self.data.shape[0]):
            # m_dot_rated
            self.data.at[self.data.index[o], "{}".format(self.MdotRated)] \
            = self.data[self.MapDensity][o] * (self.coef_mdotrated[0]
            + self.coef_mdotrated[1] * self.data[self.EvaMapTemp][o]
            + self.coef_mdotrated[2] * pow(self.data[self.EvaMapTemp][o], 2)
            + self.coef_mdotrated[3] * pow(self.data[self.EvaMapTemp][o], 3)
            + self.coef_mdotrated[4] * self.data[self.CondMapTemp][o]
            + self.coef_mdotrated[5] * pow(self.data[self.CondMapTemp][o], 2)
            + self.coef_mdotrated[6] * pow(self.data[self.CondMapTemp][o], 3)
            + self.coef_mdotrated[7] * self.data[self.EvaMapTemp][o] * self.data[self.CondMapTemp][o]
            + self.coef_mdotrated[8] * self.data[self.EvaMapTemp][o] * pow(self.data[self.CondMapTemp][o], 2)
            + self.coef_mdotrated[9] * pow(self.data[self.EvaMapTemp][o], 2) * self.data[self.CondMapTemp][o]
            + self.coef_mdotrated[10] * pow(self.data[self.EvaMapTemp][o], 2) * pow(self.data[self.CondMapTemp][o], 2))

    def VSENS_MdotPred(self):
        for o in range(self.data.shape[0]):
            mdot_pred = (self.data[self.MdotRated][o] / 3600) * (self.coef_mdotpred[0]
            + self.coef_mdotpred[1] * (self.data[self.freq][o] - self.RatedFrequency)
            + self.coef_mdotpred[2] * pow(self.data[self.freq][o] - self.RatedFrequency, 2))
            if mdot_pred <= 0:
                mdot_pred = 0
            self.data.at[self.data.index[o], "{}".format(self.MdotPred)] = mdot_pred

    def VSENS_WdotRated(self):
        for o in range(self.data.shape[0]):
            # MapDensity : kg/m^3
            self.data.at[self.data.index[o], "{}".format(self.WdotRated)] \
            = self.data[self.MapDensity][o] * (self.coef_wdotrated[0]
            + self.coef_wdotrated[1] * self.data[self.EvaMapTemp][o]
            + self.coef_wdotrated[2] * pow(self.data[self.EvaMapTemp][o], 2)
            + self.coef_wdotrated[3] * pow(self.data[self.EvaMapTemp][o], 3)
            + self.coef_wdotrated[4] * self.data[self.CondMapTemp][o]
            + self.coef_wdotrated[5] * pow(self.data[self.CondMapTemp][o], 2)
            + self.coef_wdotrated[6] * pow(self.data[self.CondMapTemp][o], 3)
            + self.coef_wdotrated[7] * self.data[self.EvaMapTemp][o] * self.data[self.CondMapTemp][o]
            + self.coef_wdotrated[8] * self.data[self.EvaMapTemp][o] * pow(self.data[self.CondMapTemp][o], 2)
            + self.coef_wdotrated[9] * pow(self.data[self.EvaMapTemp][o], 2) * self.data[self.CondMapTemp][o]
            + self.coef_wdotrated[10] * pow(self.data[self.EvaMapTemp][o], 2) * pow(self.data[self.CondMapTemp][o], 2))

    def VSENS_WdotPred(self):
        for o in range(self.data.shape[0]):
            w_dot_pred = self.data[self.WdotRated][o] * (self.coef_wdotpred[0]
            + self.coef_wdotpred[1] * (self.data[self.freq][o] - self.RatedFrequency)
            + self.coef_wdotpred[2] * pow(self.data[self.freq][o] - self.RatedFrequency, 2))
            if w_dot_pred <= 0:
                w_dot_pred = 0
            self.data.at[self.data.index[o], "{}".format(self.WdotPred)] = w_dot_pred

    def VSENS_CAPA_EVAP(self):
        h_condOut_prev = 0 # Two-Phase인 경우 대비
        for o in range(self.data.shape[0]):
            try:
                h_condOut = CP.CoolProp.PropsSI('H', 'P', self.data[self.DischargePressure][o] * 98.0665 * 1000, 'T',
                                                self.data[self.LiqTemp][o] + 273.15, 'R410A')  # Condenser out
            except:
                h_condOut = h_condOut_prev
            h_condOut_prev = h_condOut  # Two-Phase인 경우 대비
            h_suc = CP.CoolProp.PropsSI('H', 'P', self.data[self.SuctionPressure][o] * 98.0665 * 1000, 'T',
                                        self.data[self.SuctionTemp][o] + 273.15, 'R410A')  # Suction Line Enthalpy

            self.data.at[self.data.index[o], "{}".format(self.hCondOut)] = h_condOut
            self.data.at[self.data.index[o], "{}".format(self.hsuc)] = h_suc
            self.data.at[self.data.index[o], "{}".format(self.del_h_evap)] = abs(h_suc - h_condOut) / 1000
            self.data.at[self.data.index[o], "{}".format(self.Qevap)] = self.data[self.MdotPred][o] * (abs(h_suc - h_condOut) / 1000)

    def VSENS_CAPA_COND(self): # 컨덴서 용량 가상센서
        h_dis_prev = 0
        for o in range(self.data.shape[0]):
            #Discharge Temperature 컬럼이 여러개 있기 때문에 큰 값을 사용한 것
            T_dis1 = self.data[self.DischargeTemp][o]
            T_dis2 = self.data[self.DischargeTemp_sub][o]
            T_dis = max(T_dis1, T_dis2)
            try:
                h_dis = CP.CoolProp.PropsSI('H', 'P', self.data[self.DischargePressure][o] * 98.0665 * 1000, 'T', T_dis + 273.15, 'R410A') #Discharge Line
            except:
                h_dis = h_dis_prev
            h_dis_prev = h_dis

            h_condOut = self.data[self.hCondOut][o]
            self.data.at[self.data.index[o], "{}".format(self.hdis)] = h_dis
            self.data.at[self.data.index[o], "{}".format(self.del_h_cond)] = abs(h_dis - h_condOut) / 1000
            self.data.at[self.data.index[o], "{}".format(self.Qcond)] = self.data[self.MdotPred][o] * (abs(h_dis - h_condOut) / 1000)

    def VSENS_UA(self): # 컨덴서 열 관류율 가상센서 W/m^2-K
        for o in range(self.data.shape[0]):
            T_CondIn = self.data[self.DischargeTemp][o]
            T_CondOut = self.data[self.CondOutTemp][o]
            MdotPred = self.data[self.MdotPred][o]
            T_c_Sat = self.data[self.LiqTemp][o]
            try:
                c_p = CP.CoolProp.PropsSI('C', 'P', self.data[self.DischargePressure][o] * 98.0665 * 1000, 'T', self.data[self.DischargeTemp][o] + 273.15, 'R410A')
            except:
                c_p = 0
            self.data.at[self.data.index[o], "{}".format(self.cp)] = c_p
            self.data.at[self.data.index[o], "{}".format(self.ua)] = MdotPred * c_p * np.log(abs(T_c_Sat - T_CondIn) / abs(T_c_Sat - T_CondOut))

    def VSENS_HEATEXCHANGER_OUTLET_TEMP(self):
        # 작성중
        for o in range(self.data.shape[0]):
            UA = self.data[self.ua][o]
            MdotPred = self.data[self.MdotPred][o]

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


TIME = 'updated_time'
start ='2022-02-14' #데이터 시작시간
end = '2022-02-14' #데이터 끝시간

freqValue = 'comp_current_frequency1'
PdisValue = 'high_pressure'
PsucValue = 'low_pressure'
TsucValue = 'suction_temp1'
TdisValue = 'discharge_temp1'
TcondOutValue = 'cond_out_temp1'
TliqValue = 'double_tube_temp'
TinAirValue = 'evain_temp'
ToutAirValue = 'evaout_temp'

TARGET = 'Power'
COMP_MODEL_NAME = 'GB066' # GB052, GB066, GB070, GB080

VS = COMPRESSORMAPMODEL(COMP_MODEL_NAME=COMP_MODEL_NAME, TIME=TIME, start=start, end=end)

for outdv in [3067]:
    VS.VSENSOR_PROCESSING(out_unit=outdv,freqValue=freqValue, PdisValue=PdisValue,
                          PsucValue=PsucValue,TsucValue=TsucValue, TdisValue=TdisValue,
                          TcondOutValue=TcondOutValue, TliqValue=TliqValue,
                          TinAirValue=TinAirValue, ToutAirValue=ToutAirValue, target=TARGET)



