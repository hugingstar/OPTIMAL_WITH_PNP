import os
import pandas as pd
import math
import CoolProp as CP
from CoolProp.Plots import PropertyPlot
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

class COMPRESSORMAPMODEL():
    def __init__(self, COMP_MODEL_NAME, TIME, start, end):
        "파일을 호출할 경로"
        self.DATA_PATH = "D:/Optimal/Data/VirtualSensor"
        self.SAVE_PATH = "D:/Optimal/Results"
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
                 "Mdot_pred": [9.90958566E-1 , 1.66658435E-2, -1.91782998E-5],
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
                           TdisValue, TcondOutValue, TliqValue, TAirinValue, TAiroutValue,
                           ToaValue, VAiroutValue, CompSignal, target):
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
        print(self._outdata)
        self.freq = list(pd.Series(list(self._outdata.columns))[
                             pd.Series(list(self._outdata.columns)).str.contains(pat=freqValue, case=False)])
        # High Pressure
        self.DischargePressure = list(pd.Series(list(self._outdata.columns))[
                                          pd.Series(list(self._outdata.columns)).str.contains(pat=PdisValue, case=False)])[0]
        self.DischargeTemp = list(pd.Series(list(self._outdata.columns))[
                                      pd.Series(list(self._outdata.columns)).str.contains(pat=TdisValue, case=False)])
        # Low Pressure
        self.SuctionPressure = list(pd.Series(list(self._outdata.columns))[
                                        pd.Series(list(self._outdata.columns)).str.contains(pat=PsucValue, case=False)])[0]
        self.SuctionTemp = list(pd.Series(list(self._outdata.columns))[
                                    pd.Series(list(self._outdata.columns)).str.contains(pat=TsucValue, case=False)])

        #Condenser Out Temperature
        self.CondOutTemp = list(pd.Series(list(self._outdata.columns))[
                                    pd.Series(list(self._outdata.columns)).str.contains(pat=TcondOutValue, case=False)])[0]
        #Evaporator In Temperature
        self.EvapInTemp = list(pd.Series(list(self._outdata.columns))[
                                    pd.Series(list(self._outdata.columns)).str.contains(pat=TliqValue, case=False)])

        #Compressor Signal
        self.CompressorSignal = list(pd.Series(list(self._outdata.columns))[
                                        pd.Series(list(self._outdata.columns)).str.contains(pat=CompSignal, case=False)])
        #Real Power
        self.RealPower = list(pd.Series(list(self._outdata.columns))[
                                        pd.Series(list(self._outdata.columns)).str.contains(pat='value', case=False)])[0]
        self._outdata[self._outdata[self.RealPower] < 0] = None # Power 실측값이 음수인 경우 처리
        self._outdata.fillna(method='ffill', inplace=True)

        #AirSide Temperature(Outdoor Units)
        self.AirInletTemperature = list(pd.Series(list(self._outdata.columns))[
                                          pd.Series(list(self._outdata.columns)).str.contains(pat=TAirinValue, case=False)])[0]
        self.AirOutletTemperature = list(pd.Series(list(self._outdata.columns))[
                                          pd.Series(list(self._outdata.columns)).str.contains(pat=TAiroutValue, case=False)])[0]
        self.OutdoorAirTemperature = list(pd.Series(list(self._outdata.columns))[
                                          pd.Series(list(self._outdata.columns)).str.contains(pat=ToaValue, case=False)])[0]

        # AirSide Volume Rate(Outdoor Units)
        self.AirOutletVolumeReal = list(pd.Series(list(self._outdata.columns))[
                                          pd.Series(list(self._outdata.columns)).str.contains(pat=VAiroutValue, case=False)])[0]
        self.FanSteps = list(pd.Series(list(self._outdata.columns))[
                                          pd.Series(list(self._outdata.columns)).str.contains(pat='fan_step', case=False)])[0]

        #Suction Line Density
        self.SuctionDensity()

        #Virtual Power Sensor
        self.coef_wdotrated = list(self.coef['Wdot_rated'])
        self.VirtualRatedPowerSensor()
        self.coef_wdotpred = list(self.coef['Mdot_pred'])
        self.VirtualPowerSensor()

        #Virtual Mass Flow Sensor
        self.coef_mdotrated = list(self.coef['Mdot_rated'])
        self.VirtualRatedMassFlowSensor()
        self.coef_mdotpred = list(self.coef['Mdot_pred'])
        self.VirtualMassFlowSensor()

        #Virtual Outdoorunit Capacity Sensor
        self.VirtualCoolingCapacitySensor()
        self.VirtualHeatingCapacitySensor()
        self.VirtualAirVolumeFlowSensor()

        self.VirtualPerformanceSensor()
        self.VirtualUASensor()

        save = "{}/VirtualSensor/{}/{}/{}".format(self.SAVE_PATH, self.target, self.folder_name, out_unit)
        self.create_folder(save)
        self._outdata.to_csv("{}/Before_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 전

        """Plot Time Range"""
        # 그림 그릴 부분의 시작시간(plt_ST) - 끝시간(plt_ET)
        st = '00:00:00'#'11:30:00'
        et = '23:59:00'#'14:30:00'
        self.gap = 60 # 60

        solve = self._outdata
        solve = solve[solve.index >= self.folder_name + ' ' + st]
        solve = solve[solve.index <= self.folder_name + ' ' + et]
        solve.index = pd.to_datetime(solve.index)
        solve = solve.resample(rule='5min').mean()
        solve = solve[(solve[self.CompressorSignal[0]] > 0.5) & (solve[self.CompressorSignal[1]] > 0.5)]  # 작동중인 데이터만 사용
        solve.to_csv("{}/After_Outdoor_{}.csv".format(save, out_unit))  # 조건 적용 후

        """Ph, Ts Diagram"""
        self.PressureEnthalpyDiagram(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)
        self.TemperatureEntropyDiagram(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)

        """biot Data with Virtual Power Sensor"""
        self.PlottingSystem(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)
        self.PlottingMeasurement(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)
        self.PlottingVirtualPowerSensor(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)
        self.PlottingVirtualPowerSensorAccuracy(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)
        self.PlottingPerformanceSensor(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)
        self.PlottingUASensor(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)

    def VirtualUASensor(self): # 컨덴서 열 관류율 가상센서 W/m^2-K
        self.VirtualUA_cond = self.DischargePressure.replace(PdisValue, 'virtual_ua_cond')
        self.VirtualUA_evap = self.DischargePressure.replace(PdisValue, 'virtual_ua_evap')
        Tdis1 = self._outdata[self.DischargeTemp[0]].tolist()
        Tdis2 = self._outdata[self.DischargeTemp[1]].tolist()
        TcondOut = self._outdata[self.CondOutTemp].tolist()  # Double tube temp
        High_p = self._outdata[self.DischargePressure].tolist()
        Low_p = self._outdata[self.SuctionPressure].tolist()
        Tsuc = self._outdata[self.SuctionTemp[0]].tolist()
        h_evapIn = self._outdata[self.EnthalpyEvapIn].tolist()

        num = 0
        while num < self._outdata.shape[0]:
            Tdis = (Tdis1[num] + Tdis2[num]) / 2
            c_p = CP.CoolProp.PropsSI('C', 'P', Low_p[num] * 100 * 1000, 'Q', 1, 'R410A') / 1000
            Tcond_sat = CP.CoolProp.PropsSI('T', 'P', High_p[num] * 100 * 1000, 'Q', 1, 'R410A') - 273.15
            ua_cond = self._outdata[self.VirtualMdotPred][num] * c_p * np.log(abs(Tcond_sat - Tdis) / abs(Tcond_sat - TcondOut[num]))

            Tcond_sat = CP.CoolProp.PropsSI('T', 'P', Low_p[num] * 100 * 1000, 'Q', 0, 'R410A') - 273.15
            ua_evap = self._outdata[self.VirtualMdotPred][num] * c_p * np.log(abs(Tcond_sat - TcondOut[num]) / abs(Tcond_sat - Tsuc[num]))

            self._outdata.at[self._outdata.index[num], "{}".format(self.VirtualUA_cond)] = abs(ua_cond)
            self._outdata.at[self._outdata.index[num], "{}".format(self.VirtualUA_evap)] = abs(ua_evap)
            num += 1

    def VirtualPerformanceSensor(self):
        self.Performance = self.DischargePressure.replace(PdisValue, 'virtual_cop')
        self.Performance_real = self.DischargePressure.replace(PdisValue, 'real_cop')
        HeatCapacity = self._outdata[self.HeatingCapacity].tolist()
        VirtualPower = self._outdata[self.VirtualPower].tolist()
        RealPower = self._outdata[self.RealPower].tolist()

        virtual_cop_prev = 0
        real_cop_prev = 0
        num = 0
        while num < self._outdata.shape[0]:
            try:
                virtual_cop = HeatCapacity[num] / VirtualPower[num] # kW/kW
            except ZeroDivisionError:
                virtual_cop = virtual_cop_prev

            real_cop = 0
            # try:
            #     print(HeatCapacity[num], RealPower[num])
            #     real_cop = (HeatCapacity[num]) / (RealPower[num])  # kW/kW
            # except ZeroDivisionError:
            #     real_cop = real_cop_prev

            real_cop_prev = real_cop
            self._outdata.at[self._outdata.index[num], "{}".format(self.Performance)] = virtual_cop
            self._outdata.at[self._outdata.index[num], "{}".format(self.Performance_real)] = real_cop
            num += 1

    def VirtualAirVolumeFlowSensor(self):
        self.OutletAirVolumeFlow = self.DischargePressure.replace(PdisValue, 'air_volume_flow_rate')

        TairOut = self._outdata[self.AirOutletTemperature].tolist()
        TairIn = self._outdata[self.AirInletTemperature].tolist()
        Toa = self._outdata[self.OutdoorAirTemperature].tolist()
        High_p = self._outdata[self.DischargePressure].tolist()
        Tdis1 = self._outdata[self.DischargeTemp[0]].tolist()
        Tdis2 = self._outdata[self.DischargeTemp[1]].tolist()
        m_dot_pred = self._outdata[self.VirtualMdotPred].tolist()
        AirVolumeFlow_real = self._outdata[self.AirOutletVolumeReal].tolist()
        TcondOut = self._outdata[self.CondOutTemp].tolist()  # Double tube temp

        num = 0
        AirVolFlow_prev = 0
        h_diff_prev = 0
        t_diff_prev = 0
        while num < self._outdata.shape[0]:
            c_p_air = CP.CoolProp.PropsSI('C', 'P', 101325, 'T', Toa[num] + 273.15, 'Air') / 1000 #1.0035
            rho_air = CP.CoolProp.PropsSI('D', 'P', 101325, 'T', Toa[num] + 273.15, 'Air')

            try:
                h_dis = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'T', (Tdis1[num]+Tdis2[num])/2 + 273.15, 'R410A') / 1000
            except ValueError:
                h_dis = 0
            try:
                h_condOut = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'T', TcondOut[num] + 273.15, 'R410A') / 1000
            except ValueError:
                h_condOut = 0
                print("Two Phase Error!")

            h_diff = h_dis - h_condOut
            t_diff = TairIn[num] - TairOut[num]

            if h_diff < 50:
                h_diff = 0

            if t_diff < 1:
                t_diff = t_diff_prev

            if t_diff != 0:
                AirVolFlow = (m_dot_pred[num] * (h_diff)) / (c_p_air * t_diff * rho_air)
            else:
                AirVolFlow = AirVolFlow_prev

            # 정방향
            if AirVolFlow <= 0:
                AirVolFlow = 0

            AirVolFlow_prev = AirVolFlow
            t_diff_prev = t_diff
            self._outdata.at[self._outdata.index[num], "{}".format(self.OutletAirVolumeFlow)] = AirVolFlow
            self._outdata.at[self._outdata.index[num], "{}".format(self.AirOutletVolumeReal)] = AirVolumeFlow_real[num] * 3 / 60
            num += 1

    def VirtualHeatingCapacitySensor(self):
        self.HeatingCapacity = self.DischargePressure.replace(PdisValue, 'heating_capacity')
        self.EnthalpyDischarge = self.DischargePressure.replace(PdisValue, 'discharge_enthalpy')
        self.DiffHeatEnthalpy = self.DischargePressure.replace(PdisValue, 'diff_heat_enthalpy')

        High_p = self._outdata[self.DischargePressure].tolist()
        Tdis1 = self._outdata[self.DischargeTemp[0]].tolist()
        Tdis2 = self._outdata[self.DischargeTemp[1]].tolist()
        TcondOut = self._outdata[self.CondOutTemp].tolist()  # Double tube temp
        num = 0
        while num < self._outdata.shape[0]:

            try:
                h_dis = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'T', (Tdis1[num]+Tdis2[num])/2 + 273.15, 'R410A') / 1000
            except ValueError:
                h_dis = 0

            try:
                h_condOut = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'T', TcondOut[num] + 273.15, 'R410A') / 1000
            except ValueError:
                h_condOut = 0
                print("Two Phase Error!")

            h_diff =  h_dis - h_condOut
            HeatCapacity = self._outdata[self.VirtualMdotPred][num] * h_diff / 10

            self._outdata.at[self._outdata.index[num], "{}".format(self.EnthalpyDischarge)] = h_dis
            self._outdata.at[self._outdata.index[num], "{}".format(self.CondenserOutlet)] = h_condOut
            self._outdata.at[self._outdata.index[num], "{}".format(self.DiffHeatEnthalpy)] = h_diff
            self._outdata.at[self._outdata.index[num], "{}".format(self.HeatingCapacity)] = HeatCapacity
            num += 1

    def VirtualCoolingCapacitySensor(self):
        self.CoolingCapacity = self.DischargePressure.replace(PdisValue, 'cooling_capacity')
        self.CondenserOutlet = self.DischargePressure.replace(PdisValue, 'cond_out_enthalpy')
        self.EnthalpySuction = self.DischargePressure.replace(PdisValue, 'suction_enthalpy')
        self.EnthalpyEvapIn = self.DischargePressure.replace(PdisValue, 'evap_in_enthalpy')
        self.DiffEvapEnthalpy = self.DischargePressure.replace(PdisValue, 'diff_evap_enthalpy')

        Low_p = self._outdata[self.SuctionPressure].tolist()
        High_p = self._outdata[self.DischargePressure].tolist()
        TcondOut = self._outdata[self.CondOutTemp].tolist()  # Double tube temp
        Tsuc = self._outdata[self.SuctionTemp[0]].tolist()

        num = 0
        h_evapIn_prev = 0
        while num < self._outdata.shape[0]:
            h_suc = CP.CoolProp.PropsSI('H', 'P', Low_p[num] * 100 * 1000, 'T', Tsuc[num] + 273.15, 'R410A') / 1000 #[kJ/kg]

            try:
                h_condOut = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'T', TcondOut[num] + 273.15,'R410A') / 1000
                h_evapIn = CP.CoolProp.PropsSI('H', 'P', Low_p[num] * 100 * 1000, 'H', h_condOut * 1000, 'R410A') / 1000
            except ValueError:
                h_evapIn = h_evapIn_prev
                print("Two Phase Error!")
            h_diff = h_suc - h_evapIn

            CoolCapacity = self._outdata[self.VirtualMdotPred][num] * h_diff / 10

            h_evapIn_prev = h_evapIn
            self._outdata.at[self._outdata.index[num], "{}".format(self.EnthalpySuction)] = h_suc # kJ/kg
            self._outdata.at[self._outdata.index[num], "{}".format(self.EnthalpyEvapIn)] = h_evapIn # kJ/kg
            self._outdata.at[self._outdata.index[num], "{}".format(self.DiffEvapEnthalpy)] = h_diff
            self._outdata.at[self._outdata.index[num], "{}".format(self.CoolingCapacity)] = CoolCapacity # kW
            num += 1

    def VirtualMassFlowSensor(self):
        self.VirtualMdotPred = self.DischargePressure.replace(PdisValue, 'm_dot_pred')
        self.RealMdot = self.DischargePressure.replace(PdisValue, 'm_dot_real')
        self.DiffCompEnthalpy = self.DischargePressure.replace(PdisValue, 'diff_comp_enthalpy')
        self.RatedFrequency = 58 #Hz

        Low_p = self._outdata[self.SuctionPressure].tolist()
        High_p = self._outdata[self.DischargePressure].tolist()
        Tsuc = self._outdata[self.SuctionTemp[0]].tolist()
        Tdis1 = self._outdata[self.DischargeTemp[0]].tolist()
        Tdis2 = self._outdata[self.DischargeTemp[1]].tolist()

        num = 0
        h_diff_prev = 0
        while num < self._outdata.shape[0]:
            #Virtual Mass Flow
            f_real = (self._outdata[self.freq[0]][num] + self._outdata[self.freq[1]][num])/2
            """Baseline"""
            h_suc = CP.CoolProp.PropsSI('H', 'P', Low_p[num] * 100 * 1000, 'T', Tsuc[num] + 273.15, 'R410A') / 1000  # [kJ/kg]
            try:
                h_dis = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'T', ((Tdis1[num] + Tdis2[num]) / 2) + 273.15, 'R410A') / 1000
            except ValueError:
                h_dis = 0

            h_diff = h_dis - h_suc

            # 엔탈피 차이가 매우 작은 경우에는 꺼진 것과 다름 없기 때문에
            # 급격한 Mass Flow 상승이 되지 않도록 0으로 만들어줌.
            if (h_diff < 5) & (h_diff > 0):
                h_diff = h_diff_prev
            elif h_diff <= 0:
                h_diff = 0

            if h_diff != 0:
                m_dot_real = self._outdata[self.RealPower][num] / h_diff
            else:
                m_dot_real = 0
            # 정방향
            if m_dot_real <= 0:
                m_dot_real = 0

            """Virtual Sensor"""
            if h_diff != 0:
                m_dot_pred = self._outdata[self.VirtualMdotRated][num] \
                             * (self.coef_mdotpred[0] + self.coef_mdotpred[1] * (f_real - self.RatedFrequency)
                                + self.coef_mdotpred[2] * pow((f_real - self.RatedFrequency), 2))
            else:
                m_dot_pred = 0
            # 정방향
            if m_dot_pred <= 0:
                m_dot_pred = 0
            h_diff_prev = h_diff
            self._outdata.at[self._outdata.index[num], "{}".format(self.DiffCompEnthalpy)] = h_diff
            self._outdata.at[self._outdata.index[num], "{}".format(self.RealMdot)] = m_dot_real
            self._outdata.at[self._outdata.index[num], "{}".format(self.VirtualMdotPred)] = m_dot_pred
            num += 1

    def VirtualRatedMassFlowSensor(self):
        self.VirtualMdotRated = self.DischargePressure.replace(PdisValue, 'm_dot_rated')
        num = 0
        while num < self._outdata.shape[0]:
            Tsuc_real = self._outdata[self.SuctionTemp[0]][num]
            Tdis_real = (self._outdata[self.DischargeTemp[0]][num] + self._outdata[self.DischargeTemp[1]][num])/2
            self._outdata.at[self._outdata.index[num], "{}".format(self.VirtualMdotRated)] \
            = (self._outdata[self.MapDensity][num] * (self.coef_mdotrated[0]
            + self.coef_mdotrated[1] * Tsuc_real
            + self.coef_mdotrated[2] * pow(Tsuc_real, 2)
            + self.coef_mdotrated[3] * pow(Tsuc_real, 3)
            + self.coef_mdotrated[4] * Tdis_real
            + self.coef_mdotrated[5] * pow(Tdis_real, 2)
            + self.coef_mdotrated[6] * pow(Tdis_real, 3)
            + self.coef_mdotrated[7] * Tsuc_real * Tdis_real
            + self.coef_mdotrated[8] * Tsuc_real * pow(Tdis_real, 2)
            + self.coef_mdotrated[9] * pow(Tsuc_real, 2) * Tdis_real
            + self.coef_mdotrated[10] * pow(Tsuc_real, 2) * pow(Tdis_real, 2)))/3600
            num += 1

    def VirtualPowerSensor(self):
        self.VirtualPower = self.DischargePressure.replace(PdisValue, 'virtual_power')
        RealP = self._outdata[self.RealPower].tolist()
        self.RatedFrequency = 58 #Hz
        num = 0
        while num < self._outdata.shape[0]:
            f_real = (self._outdata[self.freq[0]][num] + self._outdata[self.freq[1]][num])/2
            w_dot_pred = self._outdata[self.VirtualRatedPower][num] \
                         * (self.coef_wdotpred[0]
                            + self.coef_wdotpred[1] * (f_real - self.RatedFrequency)
                            + self.coef_wdotpred[2] * pow(f_real - self.RatedFrequency, 2))
            if w_dot_pred <= 0:
                w_dot_pred = 0

            # if (self._outdata[self.CompressorSignal[0]][num] == 0) and (self._outdata[self.CompressorSignal[1]][num] == 0):
            #     w_dot_pred = 0
            self._outdata.at[self._outdata.index[num], "{}".format(self.VirtualPower)] = w_dot_pred
            self._outdata.at[self._outdata.index[num], "{}".format(self.RealPower)] = RealP[num] / 1000
            num += 1

    def VirtualRatedPowerSensor(self):
        self.VirtualRatedPower = self.DischargePressure.replace(PdisValue, 'virtual_rated_power')
        num = 0
        while num < self._outdata.shape[0]:
            Tsuc_real =  self._outdata[self.SuctionTemp[0]][num]
            Tdis_real = ((self._outdata[self.DischargeTemp[0]][num] + self._outdata[self.DischargeTemp[1]][num])/2)
            # Tdis_real = max(self._outdata[self.DischargeTemp[0]][num], self._outdata[self.DischargeTemp[1]][num])
            self._outdata.at[self._outdata.index[num], "{}".format(self.VirtualRatedPower)] \
            = self._outdata[self.MapDensity][num] * (self.coef_wdotrated[0]
            + self.coef_wdotrated[1] * Tsuc_real
            + self.coef_wdotrated[2] * pow(Tsuc_real, 2)
            + self.coef_wdotrated[3] * pow(Tsuc_real, 3)
            + self.coef_wdotrated[4] * Tdis_real
            + self.coef_wdotrated[5] * pow(Tdis_real, 2)
            + self.coef_wdotrated[6] * pow(Tdis_real, 3)
            + self.coef_wdotrated[7] * Tsuc_real * Tdis_real
            + self.coef_wdotrated[8] * Tsuc_real * pow(Tdis_real, 2)
            + self.coef_wdotrated[9] * pow(Tsuc_real, 2) * Tdis_real
            + self.coef_wdotrated[10] * pow(Tsuc_real, 2) * pow(Tdis_real, 2)) * 0.7457 / 3600
            num += 1

    def SuctionDensity(self):
        """
        냉매 Density를 Coolprop에서 추정하는 함수
        밀도는 전원이 켜지나 안 켜지나 항상 물성치로 측정되도록 프로그래밍 되었다.
        SuctionPressure : 저압측 [Pa]로 환산하여 입력 [bar] --> [Pa]
        SuctionTemp : 흡입 온도 [K] = [C] + 373.15
        Desity :  R410A의 밀도 [kg/m^3]
        :return: 가상센서 값이 데이터에 합쳐짐
        """
        self.MapDensity = self.DischargePressure.replace(PdisValue, 'density') # 컬럼이름을 바꾼것
        num = 0
        while num < self._outdata.shape[0]:
            self._outdata.at[self._outdata.index[num], "{}".format(self.MapDensity)] \
                = CP.CoolProp.PropsSI('D', 'P', self._outdata[self.SuctionPressure][num] * 100 * 1000, 'T', self._outdata[self.SuctionTemp[0]][num] + 273.15, 'R410A')
            num += 1

    """ Prediction Performance"""

    def CvRMSE(self, realList, predList):
        errorList = []
        for i in range(len(realList)):
            errorList.append(pow(predList[i] - realList[i], 2))
        RMSE = np.sqrt(sum(errorList)/ len(errorList))
        error = 100 * RMSE/np.mean(realList)
        print("Cv(RMSE) : {} % ".format(round(error, 2)))
        return round(error, 2)

    def MBE(self, realList, predList):
        errorList = []
        for i in range(len(realList)):
            errorList.append(predList[i] - realList[i])
        ERR = sum(errorList) / len(errorList)
        error = 100 * ERR
        print("MBE : {} % ".format(round(error, 2)))
        return round(error, 2)

    """Plotting"""
    def PlottingSystem(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve.index = pd.to_datetime(solve.index)
        solve = solve.resample(rule='5min').mean()
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(25, 30))
        ax1 = fig.add_subplot(5, 1, 1)
        ax2 = fig.add_subplot(5, 1, 2)
        ax3 = fig.add_subplot(5, 1, 3)
        ax4 = fig.add_subplot(5, 1, 4)
        ax5 = fig.add_subplot(5, 1, 5)

        ax1.plot(tt, solve[self.DischargePressure].tolist(), 'r-', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax1.plot(tt, solve[self.SuctionPressure].tolist(), 'b--', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax2.plot(tt, solve[self.DischargeTemp[0]].tolist(), 'r-', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax2.plot(tt, solve[self.SuctionTemp[0]].tolist(), 'b--', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax3.plot(tt, solve[self.freq[0]].tolist(), 'r-', linewidth='2', alpha=0.7, drawstyle='steps-post', )
        ax3.plot(tt, solve[self.freq[1]].tolist(), 'b--', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax4.plot(tt, solve[self.CompressorSignal[0]].tolist(), 'r-', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax4.plot(tt, solve[self.CompressorSignal[1]].tolist(), 'b--', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax5.plot(tt, solve[self.RealPower].tolist(), 'k-', linewidth='2', alpha=0.7, drawstyle='steps-post')

        gap = self.gap  # 09~18 : 120 240
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax4.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax5.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=26)
        ax2.tick_params(axis="x", labelsize=26)
        ax3.tick_params(axis="x", labelsize=26)
        ax4.tick_params(axis="x", labelsize=26)
        ax5.tick_params(axis="x", labelsize=26)

        ax1.tick_params(axis="y", labelsize=26)
        ax2.tick_params(axis="y", labelsize=26)
        ax3.tick_params(axis="y", labelsize=26)
        ax4.tick_params(axis="y", labelsize=26)
        ax5.tick_params(axis="y", labelsize=26)

        ax1.set_ylabel('Pressure', fontsize=28)
        ax2.set_ylabel('Temperature', fontsize=28)
        ax3.set_ylabel('Frequency', fontsize=28)
        ax4.set_ylabel('Compressor\nSignal', fontsize=28)
        ax5.set_ylabel('Power', fontsize=28)

        ax1.set_yticks([0, 10, 20, 30, 40, 50, 60])
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax3.set_yticks([0, 25, 50, 75, 100])
        ax4.set_yticks([0, 1])

        ax1.set_ylim([0, max(solve[self.DischargePressure].tolist()) * 1.5])
        ax2.set_ylim([-10, max(solve[self.DischargeTemp[0]].tolist()) * 1.5])
        ax3.set_ylim([0, 100])
        ax4.set_ylim([0, 3])

        ax1.legend(['High Pressure($bar$)', 'Low Pressure($bar$)'], fontsize=22, loc='upper right', ncol=2)
        ax2.legend(['Discharge Temperature($^{\circ}C$)', 'Suction Temperature($^{\circ}C$)'], fontsize=18,
                   loc='upper right', ncol=2)
        ax3.legend(['Frequency1($Hz$)', 'Frequency2($Hz$)'], fontsize=22, ncol=2)
        ax4.legend(['Compressor Signal($Hz$)', 'Compressor Signal($Hz$)'], fontsize=22, ncol=2)
        ax5.legend(['Real Power($kW$)'], fontsize=22, ncol=2, loc='upper right')

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)
        ax5.autoscale(enable=True, axis='x', tight=True)
        # ax6.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()
        # ax6.grid()

        plt.tight_layout()
        plt.savefig("{}/OutdoorUnit_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingMeasurement(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve.index = pd.to_datetime(solve.index)
        solve = solve.resample(rule='5min').mean()
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(25, 18))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        ax1.plot(tt, solve[self.AirOutletTemperature].tolist(), 'b--', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax1.plot(tt, solve[self.AirInletTemperature].tolist(), 'k-', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax2.plot(tt, solve[self.AirOutletVolumeReal].tolist(), 'k-', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax2.plot(tt, solve[self.OutletAirVolumeFlow].tolist(), 'b--', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax3.plot(tt, solve[self.RealMdot].tolist(), 'k-', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax3.plot(tt, solve[self.VirtualMdotPred].tolist(), 'r--', linewidth='2', alpha=0.9, drawstyle='steps-post')

        gap = self.gap  # 09~18 : 120 240
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=26)
        ax2.tick_params(axis="x", labelsize=26)
        ax3.tick_params(axis="x", labelsize=26)

        ax1.tick_params(axis="y", labelsize=26)
        ax2.tick_params(axis="y", labelsize=26)
        ax3.tick_params(axis="y", labelsize=26)

        ax1.set_ylabel('Temperature', fontsize=28)
        ax2.set_ylabel('Volume Flow Rate', fontsize=28)
        ax3.set_ylabel('Mass Flow Rate', fontsize=28)

        ax1.set_yticks([0, 5, 10, 15, 20, 25, 30])
        # ax2.set_yticks([0, 40, 80, 120, 160, 200])
        # ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        ax1.set_ylim([0, 20])
        # ax2.set_ylim([0, 200])
        # ax3.set_ylim([0, 0.8])

        ax1.legend(['Outlet Air Temperature($^{\circ}C$)', 'Inlet Air Temperature($^{\circ}C$)'],
                   fontsize=22, ncol=2, loc='upper right')
        ax2.legend(['Outlet Real Air Volume Flow Rate($m^{3}/min$)', 'Outlet Virtual Air Volume Flow Rate($m^{3}/min$)'],
                   fontsize=22, ncol=2, loc='upper right')
        ax3.legend(['Real Mass flow Rate($kg/s$)', 'Virtual Mass flow Rate($kg/s$)'],
                   fontsize=22, ncol=2, loc='upper right')

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.tight_layout()
        plt.savefig("{}/Measurement_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingVirtualPowerSensor(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve.index = pd.to_datetime(solve.index)
        solve = solve.resample(rule='5min').mean()
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(25, 6))
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(tt, solve[self.RealPower].tolist(), 'k-', linewidth='2', alpha=0.7, drawstyle='steps-post')
        ax1.plot(tt, solve[self.VirtualPower].tolist(), 'r-', linewidth='2', alpha=0.9, drawstyle='steps-post')

        gap = self.gap  # 09~18 : 120 240
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=26)

        ax1.tick_params(axis="y", labelsize=26)

        ax1.set_ylabel('Power', fontsize=28)

        # ax1.set_yticks([0, 20, 40, 60])

        # ax1.set_ylim([0, max(solve[self.RealPower].tolist()) * 2])

        ax1.legend(['Real Power($kW$)', 'Virtual Power($kW$)'.format()], fontsize=22,  ncol=2, loc='upper right')

        ax1.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()

        plt.tight_layout()
        plt.savefig("{}/VirtualPowerSensor_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingUASensor(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve.index = pd.to_datetime(solve.index)
        solve = solve.resample(rule='5min').mean()
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(25, 12))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.plot(tt, solve[self.FanSteps].tolist(), 'g-', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax2.plot(tt, solve[self.VirtualUA_cond].tolist(), 'b--', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax2.plot(tt, solve[self.VirtualUA_evap].tolist(), 'r-', linewidth='2', alpha=0.9, drawstyle='steps-post')


        gap = self.gap  # 09~18 : 120 240
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=26)
        ax2.tick_params(axis="x", labelsize=26)

        ax1.tick_params(axis="y", labelsize=26)
        ax2.tick_params(axis="y", labelsize=26)

        ax1.set_ylabel('Fan steps', fontsize=28)
        ax2.set_ylabel('UA', fontsize=28)

        ax1.set_yticks([0, 25, 50, 75, 100])
        # ax2.set_yticks([0, 5, 10])

        ax1.set_ylim([0, 100])
        # ax2.set_ylim([0, 10])

        ax1.legend(['Fan steps'], fontsize=22, ncol=2, loc='upper right')
        ax2.legend(['UA (Outdoor, $kW/K$)','UA (Indoor, $kW/K$)'], fontsize=22,  ncol=2, loc='upper right')

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()

        plt.tight_layout()

        plt.savefig("{}/VirtualUASensor_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingPerformanceSensor(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve.index = pd.to_datetime(solve.index)
        solve = solve.resample(rule='5min').mean()
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(25, 18))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        ax1.plot(tt, solve[self.DiffHeatEnthalpy].tolist(), 'g-', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax1.plot(tt, solve[self.DiffCompEnthalpy].tolist(), 'b--', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax1.plot(tt, solve[self.DiffEvapEnthalpy].tolist(), 'k-.', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax2.plot(tt, solve[self.HeatingCapacity].tolist(), 'r-', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax2.plot(tt, solve[self.CoolingCapacity].tolist(), 'b--', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax3.plot(tt, solve[self.Performance].tolist(), 'b-', linewidth='2', alpha=0.9, drawstyle='steps-post')
        ax3.plot(tt, solve[self.Performance_real].tolist(), 'r--', linewidth='2', alpha=0.9, drawstyle='steps-post')

        gap = self.gap  # 09~18 : 120 240
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=26)
        ax2.tick_params(axis="x", labelsize=26)
        ax3.tick_params(axis="x", labelsize=26)

        ax1.tick_params(axis="y", labelsize=26)
        ax2.tick_params(axis="y", labelsize=26)
        ax3.tick_params(axis="y", labelsize=26)

        ax1.set_ylabel('Enthalpy difference', fontsize=28)
        ax2.set_ylabel('Capacity', fontsize=28)
        ax3.set_ylabel('Performance', fontsize=28)

        ax1.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        ax3.set_yticks([0, 2, 4, 6, 8, 10])

        ax1.set_ylim([0, 400]) # 400
        ax2.set_ylim([0, 100])
        ax3.set_ylim([0, 6])

        ax1.legend(['Enthalpy Difference \n between Discharge and Condenser outlet ($kJ/kg$)',
                    'Enthalpy Difference \n between Compressor inlet and Discharge ($kJ/kg$)',
                    'Enthalpy Difference \n between Evaporator inlet and Suction ($kJ/kg$)'],
                   fontsize=22,  ncol=2, loc='upper right')
        ax2.legend(['Virtual Capacity(Indoor, $kW$)','Virtual Capacity (Outdoor, $kW$)'],
                   fontsize=22, ncol=2, loc='upper right')
        ax3.legend(['Virtual Performance(COP, $kW/kW$)'],
                   fontsize=22, ncol=2, loc='upper right')

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.tight_layout()
        plt.savefig("{}/VirtualPerformanceSensor_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingVirtualPowerSensorAccuracy(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve.index = pd.to_datetime(solve.index)
        solve = solve.resample(rule='5min').mean()
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        #작동중인 데이터만 사용해서 정확도 계산을 위한 데이터만 남김
        tem = solve[(solve[self.CompressorSignal[0]] == 1) | (solve[self.CompressorSignal[1]] == 1)]
        tem.to_csv("{}/ACCData_Outdoor_{}.csv".format(save, out_unit))

        self.error_cvrmse = self.CvRMSE(realList=tem[self.RealPower].tolist(), predList=tem[self.VirtualPower].tolist())
        self.error_mbe = self.MBE(realList=tem[self.RealPower].tolist(), predList=tem[self.VirtualPower].tolist())

        errorACC = pd.DataFrame({'CvRMSE':[self.error_cvrmse], 'MBE' : [self.error_mbe]})
        errorACC.to_csv("{}/ACC_Outdoor_{}.csv".format(save, out_unit))

        """Seperation point"""
        seper_time = "14:40:00"
        tem_Normal = tem[tem.index <= self.folder_name + ' ' + seper_time]
        tem_fault = tem[tem.index > self.folder_name + ' ' + seper_time]

        # #Sensor Accuracy
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(tem_Normal[self.RealPower].tolist(), tem_Normal[self.VirtualPower].tolist(), s=60, alpha=0.7)
        ax1.scatter(tem_fault[self.RealPower].tolist(), tem_fault[self.VirtualPower].tolist(), s=60, alpha=0.7)

        ax1.tick_params(axis="x", labelsize=30)

        ax1.tick_params(axis="y", labelsize=30)

        # ax1.legend(['Normal state', 'Fault state(KT2)'], fontsize=28,  ncol=2, loc='upper right')

        ax1.set_ylabel('Virtual Power', fontsize=32)
        ax1.set_xlabel('Real Power', fontsize=32)

        ax1.set_xticks([0, 10, 20, 30, 40, 50])
        ax1.set_yticks([0, 10, 20, 30, 40, 50])

        ax1.grid()
        plt.tight_layout()
        plt.savefig("{}/VirtualPowerSensor_Acc_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def TemperatureEntropyDiagram(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        solve = solve[((solve[self.CompressorSignal[0]] != 0)
                      | (solve[self.CompressorSignal[1]] != 0))
                      & ((solve[self.freq[0]] != 0)
                      | (solve[self.freq[1]] != 0))
                      & (solve[self.AirOutletVolumeReal] != 0)]

        High_p = [statistics.mean(solve[self.DischargePressure].tolist())]

        Low_p = [statistics.mean(solve[self.SuctionPressure].tolist())]
        TcondOut = [statistics.mean(solve[self.CondOutTemp].tolist())] #Double tube temp
        TevapIn = [statistics.mean(solve[self.EvapInTemp[0]].tolist())]

        if len(self.DischargeTemp) != 1:
            Tdis1 = statistics.mean(solve[self.DischargeTemp[0]].tolist())
            Tdis2 = statistics.mean(solve[self.DischargeTemp[1]].tolist())
            Tdis = [(Tdis1 + Tdis2) / 2]
        else:
            Tdis = [statistics.mean(solve[self.DischargeTemp[0]].tolist())]
        Tsuc = [statistics.mean(solve[self.SuctionTemp[0]].tolist())]

        dicc = {"High Pressure": High_p, "Low Pressure": Low_p, "Discharge Temperature": Tdis,
                "Suction Temperature": Tsuc, "Condenser Outlet Temperature": TcondOut, "Evaporator Inlet Temperature": TevapIn}
        self.df_cycle_TS = pd.DataFrame(dicc, index=solve.index)

        num = 0
        while num == 0:
            # Suction Lin Enthalpy [kJ/kg-K]
            s_suc = CP.CoolProp.PropsSI('S', 'P', Low_p[num] * 100 * 1000, 'T', Tsuc[num] + 273.15, 'R410A')/1000
            self.df_cycle_TS['s_suc'] = s_suc
            s_suc_sat = CP.CoolProp.PropsSI('S', 'P', Low_p[num] * 100 * 1000, 'Q', 1, 'R410A') / 1000
            self.df_cycle_TS['s_suc_sat'] = s_suc_sat
            T_suc_sat = CP.CoolProp.PropsSI('T', 'P', Low_p[num] * 100 * 1000, 'Q', 1, 'R410A')
            self.df_cycle_TS['T_suc_sat'] = T_suc_sat

            #Discharge Entropy [kJ/kg-K]
            s_dis = CP.CoolProp.PropsSI('S', 'P', High_p[num] * 100 * 1000, 'T', Tdis[num] + 273.15, 'R410A') / 1000
            self.df_cycle_TS['s_dis'] = s_dis
            s_dis_sat = CP.CoolProp.PropsSI('S', 'P', High_p[num] * 100 * 1000, 'Q', 1, 'R410A') / 1000
            self.df_cycle_TS['s_dis_sat'] = s_dis_sat
            T_dis_sat = CP.CoolProp.PropsSI('T', 'P', High_p[num] * 100 * 1000, 'Q', 1, 'R410A')
            self.df_cycle_TS['T_dis_sat'] = T_dis_sat

            s_sat_liquid = CP.CoolProp.PropsSI('S', 'P', High_p[num] * 100 * 1000, 'Q', 0, 'R410A') / 1000
            self.df_cycle_TS['s_sat_liquid'] = s_sat_liquid
            T_sat_liquid = CP.CoolProp.PropsSI('T', 'P', High_p[num] * 100 * 1000, 'Q', 0, 'R410A')
            self.df_cycle_TS['T_sat_liquid'] = T_sat_liquid

            # # Condenser Outlet Enthalpy [kJ/kg-K]
            s_condOut = CP.CoolProp.PropsSI('S', 'P', High_p[num] * 100 * 1000, 'T', TcondOut[num] + 273.15, 'R410A')/1000
            self.df_cycle_TS['s_condOut'] = s_condOut

            # # Evaporator Inlet Enthalpy [kJ/kg-K]
            h_condOut = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'Q', 0, 'R410A')
            self.df_cycle_TS['h_condOut'] = h_condOut
            s_evapIn = CP.CoolProp.PropsSI('S', 'P', Low_p[num] * 100 * 1000, 'H', h_condOut, 'R410A')/1000
            self.df_cycle_TS['s_evapIn'] = s_evapIn

            pp = PropertyPlot('R410A', 'TS', unit_system='KSI')
            pp.calc_isolines(CP.iQ, num=15)
            pp.calc_isolines(CP.iP, num=15)
            pp.draw()
            plt.plot([s_suc_sat, s_suc], [T_suc_sat, Tsuc[num] + 273.15], color='r', alpha=0.8)
            plt.plot([s_suc, s_dis], [Tsuc[num] + 273.15, Tdis[num] + 273.15], color='r', alpha=0.8)
            plt.plot([s_dis_sat, s_dis], [T_dis_sat, Tdis[num] + 273.15], color='r', alpha=0.8)
            plt.plot([s_sat_liquid, s_dis_sat], [T_sat_liquid, T_dis_sat], color='r', alpha=0.8)
            plt.plot([s_condOut, s_sat_liquid], [TcondOut[num] + 273.15, T_sat_liquid], color='r', alpha=1)
            plt.plot([s_condOut, s_evapIn], [TcondOut[num] + 273.15, TevapIn[num]+ 273.15], color='r', alpha=1)
            plt.plot([s_evapIn, s_suc_sat], [TevapIn[num] + 273.15, T_suc_sat], color='r', alpha=1)

            plt.scatter(s_suc_sat, T_suc_sat, marker='o', color='g', alpha=0.8)
            plt.scatter(s_suc, Tsuc[num] + 273.15, marker='o', color='r', alpha=0.8)
            plt.scatter(s_dis, Tdis[num] + 273.15, marker='o', color='r', alpha=0.8)
            plt.scatter(s_dis_sat, T_dis_sat, marker='o', color='g', alpha=0.8)
            plt.scatter(s_sat_liquid, T_sat_liquid, marker='o', color='g', alpha=0.8)
            plt.scatter(s_condOut, TcondOut[num] + 273.15, marker='o', color='r', alpha=0.8)
            plt.scatter(s_evapIn, TevapIn[num] + 273.15, marker='o', color='r', alpha=0.8)

            plt.grid()
            plt.tight_layout()
            pp.savefig("{}/TSDiagram_Outdoor_{}.png".format(save, out_unit))
            # pp.show()
            num += 1
            plt.clf()

        """데이터 프레임에 평균을 내린 시간 길이 정보가 포함되어 있음"""
        self.df_cycle_TS.to_csv("{}/TSCycle_Outdoor_{}.csv".format(save, out_unit))

    def PressureEnthalpyDiagram(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self._outdata.fillna(0)
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index <= plt_ET]

        solve = solve[((solve[self.CompressorSignal[0]] != 0)
                      | (solve[self.CompressorSignal[1]] != 0))
                      & ((solve[self.freq[0]] != 0)
                      | (solve[self.freq[1]] != 0))
                      & (solve[self.AirOutletVolumeReal] != 0)]

        High_p = [statistics.mean(solve[self.DischargePressure].tolist())]

        Low_p = [statistics.mean(solve[self.SuctionPressure].tolist())]
        TcondOut = [statistics.mean(solve[self.CondOutTemp].tolist())] #Double tube temp

        if len(self.DischargeTemp) != 1:
            Tdis1 = statistics.mean(solve[self.DischargeTemp[0]].tolist())
            Tdis2 = statistics.mean(solve[self.DischargeTemp[1]].tolist())
            Tdis = [(Tdis1 + Tdis2)/2]
        else:
            Tdis = [statistics.mean(solve[self.DischargeTemp[0]].tolist())]
        Tsuc = [statistics.mean(solve[self.SuctionTemp[0]].tolist())]

        dicc = {"High Pressure" : High_p, "Low Pressure": Low_p, "Discharge Temperature" : Tdis, "Suction Temperature" : Tsuc, "Condenser Outlet Temperature" : TcondOut}
        self.df_cycle = pd.DataFrame(dicc, index=solve.index)

        num = 0
        while num == 0:
            # Suction Lin Enthalpy [kJ/kg]
            h_sat_vap = CP.CoolProp.PropsSI('H', 'P', Low_p[num] * 100 * 1000, 'Q', 1, 'R410A') / 1000
            self.df_cycle['h_sat_vap'] = h_sat_vap
            h_suc = CP.CoolProp.PropsSI('H', 'P', Low_p[num] * 100 * 1000, 'T', Tsuc[num] + 273.15, 'R410A')/1000
            self.df_cycle['h_suc'] = h_suc
            c_p = CP.CoolProp.PropsSI('C', 'P', Low_p[num] * 100 * 1000, 'Q', 1, 'R410A') / 1000
            self.df_cycle['c_p_low'] = c_p
            self.SuperHeat = (h_suc - h_sat_vap) / c_p
            self.df_cycle['SuperHeat'] = self.SuperHeat

            # #Discharge Enthalpy [kJ/kg]
            h_dis = CP.CoolProp.PropsSI('H', 'P', High_p[ num] * 100 * 1000, 'T', Tdis[num] + 273.15, 'R410A') / 1000
            self.df_cycle['h_dis'] = h_dis
            h_dis_sat = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'Q', 1, 'R410A') / 1000
            self.df_cycle['h_dis_sat'] = h_dis_sat

            # Condenser Outlet Enthalpy [kJ/kg]
            h_condOut = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000,'T',TcondOut[num] + 273.15,'R410A')/1000
            self.df_cycle['h_condOut'] = h_condOut
            h_sat_liq = CP.CoolProp.PropsSI('H', 'P', High_p[num] * 100 * 1000, 'Q', 0, 'R410A') / 1000
            self.df_cycle['h_sat_liq'] = h_sat_liq

            # Evaporator Inlet Enthalpy [kJ/kg]
            h_evapIn = CP.CoolProp.PropsSI('H', 'P', Low_p[num] * 100 * 1000, 'H', h_condOut * 1000, 'R410A')/1000
            self.df_cycle['h_evapIn'] = h_evapIn

            c_p = CP.CoolProp.PropsSI('C', 'P', High_p[num] * 100 * 1000, 'Q', 0, 'R410A') / 1000 #kJ/kg-K
            self.df_cycle['c_p_high'] = c_p

            self.SubCooling = (h_sat_liq - h_evapIn) / c_p
            self.df_cycle['Subcooling'] = self.SubCooling

            heatcapa = (h_dis - h_condOut)
            self.df_cycle['Heating capa'] = heatcapa
            coolcapa = (h_suc - h_evapIn)
            self.df_cycle['Cooling capa'] = coolcapa
            energyConsumption = (h_dis - h_suc)
            self.df_cycle['Energy Consumption'] = energyConsumption

            COP_Heat = heatcapa/energyConsumption
            self.df_cycle['COP_Heating'] = COP_Heat
            COP_Cool = coolcapa / energyConsumption
            self.df_cycle['COP_Cooling'] = COP_Cool

            pp = PropertyPlot('R410A', 'PH', unit_system='KSI')
            pp.calc_isolines(CP.iT, num=15)
            pp.calc_isolines(CP.iQ, num=15)
            pp.draw()

            plt.plot([h_sat_vap, h_suc], [Low_p[num] * 100, Low_p[num] * 100], color='r', alpha=0.8)
            plt.plot([h_suc, h_dis], [Low_p[num] * 100, High_p[num] * 100], color='r', alpha= 0.8)
            plt.plot([h_suc, h_dis], [Low_p[num] * 100, High_p[num] * 100], color='r', alpha=0.8)
            plt.plot([h_sat_vap, h_dis], [High_p[num] * 100, High_p[num] * 100], color='r', alpha=0.8)
            plt.plot([h_sat_liq, h_sat_vap], [High_p[num] * 100, High_p[num] * 100], color='r', alpha=0.8)
            plt.plot([h_condOut, h_sat_vap], [High_p[num] * 100, High_p[num] * 100], color='r', alpha=0.8)
            plt.plot([h_evapIn, h_condOut], [Low_p[num] * 100, High_p[num] * 100], color='r', alpha=0.8)
            plt.plot([h_evapIn, h_sat_vap], [Low_p[num] * 100, Low_p[num] * 100], color='r', alpha=0.8)

            plt.scatter(h_sat_vap, Low_p[num] * 100, marker='o', color='g', alpha=0.8)
            plt.scatter(h_suc, Low_p[num] * 100, marker='o', color='r', alpha= 0.8)
            plt.scatter(h_dis, High_p[num] * 100, marker='o', color='r', alpha=0.8)
            plt.scatter(h_dis_sat, High_p[num] * 100, marker='o', color='g', alpha=0.8)
            plt.scatter(h_sat_liq, High_p[num] * 100, marker='o', color='g', alpha=0.8)
            plt.scatter(h_condOut, High_p[num] * 100, marker='o', color='r', alpha=0.8)
            plt.scatter(h_evapIn, Low_p[num] * 100, marker='o', color='r', alpha=0.8)

            plt.grid()
            plt.tight_layout()
            pp.savefig("{}/PHDiagram_Outdoor_{}.png".format(save, out_unit))
            # pp.show()
            num += 1
            plt.clf()

        """데이터 프레임에 평균을 내린 시간 길이 정보가 포함되어 있음"""
        self.df_cycle.to_csv("{}/CompressionCycle_Outdoor_{}.csv".format(save, out_unit))

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
start ='2022-01-17' #데이터 시작시간
end = '2022-01-17' #데이터 끝시간

freqValue = 'comp_current_frequency'
PdisValue = 'high_pressure'
PsucValue = 'low_pressure'
TsucValue = 'suction_temp'
TdisValue = 'discharge_temp'
TcondOutValue = 'double_tube_temp'
TliqValue = 'cond_out_temp'
TAirinValue = 'Air_inlet_Average_Temperature'
TAiroutValue = 'Air_outlet_Average_Temperature'
VAiroutValue = 'outdoor_volume'
ToaValue = 'outdoor_temperature'
CompSignal = 'comp'

TARGET = 'Power'
COMP_MODEL_NAME = 'GB066' # GB052, GB066, GB070, GB080

VS = COMPRESSORMAPMODEL(COMP_MODEL_NAME=COMP_MODEL_NAME, TIME=TIME, start=start, end=end)

for outdv in [3069]:
    VS.VSENSOR_PROCESSING(out_unit=outdv,freqValue=freqValue, PdisValue=PdisValue,
                          PsucValue=PsucValue,TsucValue=TsucValue, TdisValue=TdisValue,
                          TcondOutValue=TcondOutValue, TliqValue=TliqValue,
                          TAirinValue=TAirinValue, TAiroutValue=TAiroutValue,
                          ToaValue=ToaValue,VAiroutValue=VAiroutValue, CompSignal=CompSignal, target=TARGET)



