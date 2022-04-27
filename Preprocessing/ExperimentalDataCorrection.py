import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os

class DataCorrection:
    def __init__(self, TIME, start, end):
        "파일을 호출할 경로"
        self.DATA_PATH = "D:/OPTIMAL/Data/Experiment"
        self.SAVE_PATH = "D:/OPTIMAL/Results"
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

        self.dido = {
            3065: [3109, 3100, 3095, 3112, 3133, 3074, 3092, 3105, 3091, 3124,
                   3071, 3072, 3123, 3125, 3106, 3099, 3081, 3131, 3094, 3084],
            3069: [3077, 3082, 3083, 3089, 3096, 3104, 3110, 3117, 3134, 3102,
                   3116, 3129, 3090],
            3066: [3085, 3086, 3107, 3128, 3108, 3121],
            3067: [3075, 3079, 3080, 3088, 3094, 3101, 3111, 3114, 3115, 3119,
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
        print(self.folder_name)
        self.create_folder('{}/Experiment'.format(self.SAVE_PATH))  # Deepmodel 폴더를 생성

    def Visualizing(self, out_unit):

        # 건물 인식(딕셔너리에서 포함된 건물로 인식한다.)
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        save = "{}/Experiment/{}".format(self.SAVE_PATH, self.folder_name)
        self.create_folder(save)
        # Outdoor unit data from biot
        self._outdUnitPath = "{}/{}/outdoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit)
        self._outunitData = pd.read_csv(self._outdUnitPath, index_col=self.TIME)
        self._outunitData.to_csv("{}/Outdoor_{}.csv".format(save, out_unit))
        # print("[Outdoor Unit biot Data] : {}".format(self._outunitData.shape))

        # Outdoor unit velocity measurement
        self._outFlowPath = "{}/{}/{}{} flow.csv".format(self.DATA_PATH, self.folder_name,self.start_month, self.start_date)
        self._outFlowData = pd.read_csv(self._outFlowPath)
        self.col1 = list(pd.Series(list(self._outFlowData.columns))[pd.Series(list(self._outFlowData.columns)).str.contains(pat='Time', case=False)])
        self.col2 = list(pd.Series(list(self._outFlowData.columns))[pd.Series(list(self._outFlowData.columns)).str.contains(pat='velocity', case=False)])
        self.flowVelocity = self._outFlowData[[self.col1[0], self.col2[0]]]
        self.flowVelocity = self.flowVelocity.rename(columns={self.col1[0]: self.TIME})
        for _o in range(len(self.col1)-1):
            tem = self._outFlowData[[self.col1[_o + 1], self.col2[_o + 1]]]
            tem = tem.rename(columns= {self.col1[_o + 1] : self.TIME, self.col2[_o + 1] : self.col2[0]})
            self.flowVelocity = pd.concat([self.flowVelocity, tem], axis=0, ignore_index=True)
        self.flowVelocity['Date'] = self.folder_name
        self.flowVelocity[self.TIME] = self.flowVelocity['Date'] + ' ' + self.flowVelocity[self.TIME]
        self.flowVelocity.index = pd.to_datetime(self.flowVelocity[self.TIME])
        self.flowVelocity = self.flowVelocity.resample('1T').mean()
        self.flowVelocity.to_csv("{}/Velocity_Outdoor_{}.csv".format(save, out_unit))
        # print("[AirFlow Velocity Measurement Data] : {}".format(self.flowVelocity.shape))

        # Outdoor Unit Temperature1 : Inlet
        self._outTemp1Path = "{}/{}/{}{} temp.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        self._outTemp1Data = pd.read_csv(self._outTemp1Path)
        self._outTemp1Data[self.TIME] = self._outTemp1Data[['Date', 'Time']].apply(' '.join, axis=1)
        self.col1 = list(pd.Series(list(self._outTemp1Data.columns))[
                             pd.Series(list(self._outTemp1Data.columns)).str.contains(pat=self.TIME, case=False)])
        self.col2 = list(pd.Series(list(self._outTemp1Data.columns))[
                             pd.Series(list(self._outTemp1Data.columns)).str.contains(pat='point', case=False)])
        self.col3 = list(pd.Series(list(self._outTemp1Data.columns))[
                             pd.Series(list(self._outTemp1Data.columns)).str.contains(pat='condout', case=False)])
        self.features = self.col1 + self.col2 + self.col3
        self._outTemp1Data = self._outTemp1Data[self.features]
        self._outTemp1Data.set_index(self.TIME, inplace=True)
        self._outTemp1Data.index = pd.to_datetime(self._outTemp1Data.index)
        self._outTemp1Data.sort_index(ascending=True)
        self._outTemp1Data = self._outTemp1Data.resample('1T').mean()
        self._outTemp1Data.to_csv("{}/Temp1_Outdoor_{}.csv".format(save, out_unit))
        # print("[Outdoor unit Measurement Temperature 1] : {}".format(self._outTemp1Data.shape))

        # Outdoor Unit Temperature2 : Outlet
        self._outTemp2Path = "{}/{}/{}{} temp2.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        self._outTemp2Data = pd.read_csv(self._outTemp2Path)
        self._outTemp2Data[self.TIME] = self._outTemp2Data[['Date', 'Time']].apply(' '.join, axis=1)

        self.col1 = list(pd.Series(list(self._outTemp2Data.columns))[
                             pd.Series(list(self._outTemp2Data.columns)).str.contains(pat=self.TIME, case=False)])
        self.col2 = list(pd.Series(list(self._outTemp2Data.columns))[
                             pd.Series(list(self._outTemp2Data.columns)).str.contains(pat='point', case=False)])
        self.col3 = list(pd.Series(list(self._outTemp2Data.columns))[
                             pd.Series(list(self._outTemp2Data.columns)).str.contains(pat='condout', case=False)])
        self.features = self.col1 + self.col2 + self.col3
        self._outTemp2Data = self._outTemp2Data[self.features]
        self._outTemp2Data.set_index(self.TIME, inplace=True)
        self._outTemp2Data.index = pd.to_datetime(self._outTemp2Data.index)
        self._outTemp2Data.sort_index(ascending=True)
        self._outTemp2Data = self._outTemp2Data.resample('1T').mean()
        self._outTemp2Data.to_csv("{}/Temp2_Outdoor_{}.csv".format(save, out_unit))
        print("[Outdoor unit Measurement Temperature 2] : {}".format(self._outTemp2Data.shape))
        # print(self._outTemp2Data) #!!

        #Outdoor unit volume data
        self._outVolume1Path = "{}/{}/{}{} volume1.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        self._outVolume2Path = "{}/{}/{}{} volume2.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        self._outVolume1Data = pd.read_csv(self._outVolume1Path)
        self._outVolume2Data = pd.read_csv(self._outVolume2Path)
        self.col1 = list(pd.Series(list(self._outVolume1Data.columns))[
                             pd.Series(list(self._outVolume1Data.columns)).str.contains(pat='Time', case=False)])
        self.col2 = list(pd.Series(list(self._outVolume2Data.columns))[
                             pd.Series(list(self._outVolume2Data.columns)).str.contains(pat='Time', case=False)])
        self._outVolume1Data = self._outVolume1Data.rename(columns={self.col1[0]: self.TIME})
        self._outVolume2Data = self._outVolume2Data.rename(columns={self.col2[0]: self.TIME})
        self._outVolumeData = pd.concat([self._outVolume1Data, self._outVolume2Data], axis=0, ignore_index=True)
        self._outVolumeData['Date'] = self.folder_name
        self._outVolumeData[self.TIME] = self._outVolumeData[['Date', self.TIME]].apply(' '.join, axis=1)
        self._outVolumeData[self.TIME] = pd.to_datetime(self._outVolumeData[self.TIME])
        self._outVolumeData.set_index(self.TIME, inplace=True)
        self.col1 = list(pd.Series(list(self._outVolumeData.columns))[
                             pd.Series(list(self._outVolumeData.columns)).str.contains(pat='volume', case=False)])
        self._outVolumeData = self._outVolumeData[self.col1]
        self._outVolumeData.sort_index(ascending=True)
        self._outVolumeData = self._outVolumeData.resample('1T').mean()
        self._outVolumeData.to_csv("{}/Volume_Outdoor_{}.csv".format(save, out_unit))
        # print(self._outVolumeData) #!!

        self.OutIntegData = self._outunitData.join(self.flowVelocity, how='left')
        self.OutIntegData = self.OutIntegData.join(self._outVolumeData, how='left')
        self.OutIntegData.to_csv("{}/OutIntegData_Outdoor_{}.csv".format(save, out_unit))

        # 그림 그릴 부분의 시작시간(plt_ST) - 끝시간(plt_ET)
        st = '09:00:00'
        et = '17:40:00'

        #Outdoor System
        self.PlottingOutdoorSystem(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save,
                             out_unit=out_unit)
        #Inlet Data
        self.OutIntegDataInlet = self.OutIntegData.join(self._outTemp1Data, how='left')
        self.PlottingOutdoorInlet(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save,
                             out_unit=out_unit)

        #Outlet
        self.OutIntegDataOutlet = self.OutIntegData.join(self._outTemp2Data, how='left')
        self.PlottingOutdoorOutlet(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save,
                             out_unit=out_unit)


        for indv in list(self.bldginfo[out_unit]):
            self._indpath = "{}/{}/{}{}_indoor_{}.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date, indv)
            self._indata = pd.read_csv(self._indpath)
            self._indata = self._indata.replace({"High": 3, "Mid": 2, "Low": 1, "Auto": 0})
            print(self._indata)



    def PlottingOutdoorOutlet(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        # 측정을 하지 않은 곳은 다 0으로 입력하였다.
        solve = self.OutIntegDataOutlet.fillna(0)
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index < plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)

        tempInlet_avg = solve[['point1', 'point2', 'point4', 'point5', 'point6', 'point7', 'point8', 'point9']].mean(axis=1)
        print(tempInlet_avg)

        ax1.plot(tt, solve['velocity1'].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax2.plot(tt, solve['volume'].tolist(), 'b-', linewidth='2', drawstyle='steps-post')
        ax3.plot(tt, solve['high_pressure'].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax3.plot(tt, solve['low_pressure'].tolist(), 'b-', linewidth='2', drawstyle='steps-post')
        ax4.plot(tt, solve['outdoor_temperature'].tolist(), 'g--', linewidth='2', drawstyle='steps-post')
        ax4.legend(['Outdoor temperature'], fontsize=18)
        for i in [1, 2, 4, 5, 6, 7, 8, 9]:
            ax4.plot(tt, solve['point{}'.format(i)].tolist(), linewidth='1.5', drawstyle='steps-post')
        ax4.plot(tt, tempInlet_avg.tolist(), 'r', linewidth='3', drawstyle='steps-post')

        ax3.legend(['High Pressure', 'Low Pressure'], fontsize=18)

        gap = 120 # 09~18 : 120
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax4.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=22)
        ax2.tick_params(axis="x", labelsize=22)
        ax3.tick_params(axis="x", labelsize=22)
        ax4.tick_params(axis="x", labelsize=22)

        ax1.tick_params(axis="y", labelsize=22)
        ax2.tick_params(axis="y", labelsize=22)
        ax3.tick_params(axis="y", labelsize=22)
        ax4.tick_params(axis="y", labelsize=22)

        ax1.set_ylabel('Velocity Measurement\n(Inlet)', fontsize=24)
        ax2.set_ylabel('Volume Measurement\n(Outlet)', fontsize=24)
        ax3.set_ylabel('Pressure\n(System Nodes)', fontsize=24)
        ax4.set_ylabel('Temp Measurement\n(Outlet)', fontsize=24)

        ax4.set_xlabel('Time', fontsize=24)

        ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax2.set_yticks([0, 500, 1000, 1500, 2000, 2500])
        ax3.set_yticks([0, 20, 40, 60, 80])
        ax4.set_yticks([0, 10, 20, 30, 40])

        ax4.set_ylim([5, 35])

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

        plt.tight_layout()
        plt.savefig("{}/OutdoorOutlet_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingOutdoorInlet(self, plt_ST, plt_ET, save, out_unit):
        # 온도 포인트만 그리는 곳
        plt.rcParams["font.family"] = "Times New Roman"
        # 측정을 하지 않은 곳은 다 0으로 입력하였다.
        solve = self.OutIntegDataInlet.fillna(0)
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index < plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)

        # Temperature Average
        tempInlet_avg = solve[['point1', 'point2', 'point3', 'point4', 'point5', 'point6', 'point7', 'point8', 'point10',
                               'point11', 'point12', 'point13', 'point14', 'point15', 'point17','point18', 'point20',
                               'point22', 'point23', 'point25', 'point26', 'point27', 'point28', 'point29', 'point30',
                               'point32', 'point33', 'point34', 'point36', 'point37', 'point39']].mean(axis=1)
        print(tempInlet_avg)

        ax1.plot(tt, solve['outdoor_temperature'].tolist(), 'g--', linewidth='2', drawstyle='steps-post')
        ax1.legend(['Outdoor temperature'], fontsize=18)
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 10]: #Except 9
            ax1.plot(tt, solve['point{}'.format(i)].tolist(), linewidth='1.5', drawstyle='steps-post')
        ax1.plot(tt, tempInlet_avg.tolist(), 'r', linewidth='3', drawstyle='steps-post')

        ax2.plot(tt, solve['outdoor_temperature'].tolist(), 'g--', linewidth='2', drawstyle='steps-post')
        ax2.legend(['Outdoor temperature'], fontsize=18)
        for i in [11, 12, 13, 14, 15, 17, 18, 20]: #Except 16 19
            ax2.plot(tt, solve['point{}'.format(i)].tolist(), linewidth='1.5', drawstyle='steps-post')
        ax2.plot(tt, tempInlet_avg.tolist(), 'r', linewidth='3', drawstyle='steps-post')

        ax3.plot(tt, solve['outdoor_temperature'].tolist(), 'g--', linewidth='2', drawstyle='steps-post')
        ax3.legend(['Outdoor temperature'], fontsize=18)
        for i in [22, 23, 25, 26, 27, 28, 29, 30]: #except 21 24
            ax3.plot(tt, solve['point{}'.format(i)].tolist(), linewidth='1.5', drawstyle='steps-post')
        ax3.plot(tt, tempInlet_avg.tolist(), 'r', linewidth='3', drawstyle='steps-post')

        ax4.plot(tt, solve['outdoor_temperature'].tolist(), 'g--', linewidth='2', drawstyle='steps-post')
        ax4.legend(['Outdoor temperature'], fontsize=18)
        for i in [32, 33, 34, 36, 37, 39]: #Except 31, 35, 38, 40
            ax4.plot(tt, solve['point{}'.format(i)].tolist(), linewidth='1.5', drawstyle='steps-post')
        ax4.plot(tt, tempInlet_avg.tolist(), 'r', linewidth='3', drawstyle='steps-post')

        gap = 120 # 09~18 : 120
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax4.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=22)
        ax2.tick_params(axis="x", labelsize=22)
        ax3.tick_params(axis="x", labelsize=22)
        ax4.tick_params(axis="x", labelsize=22)

        ax1.tick_params(axis="y", labelsize=22)
        ax2.tick_params(axis="y", labelsize=22)
        ax3.tick_params(axis="y", labelsize=22)
        ax4.tick_params(axis="y", labelsize=22)

        ax1.set_ylabel('Temp Measurement\n(Inlet)', fontsize=24)
        ax2.set_ylabel('Temp Measurement\n(Inlet)', fontsize=24)
        ax3.set_ylabel('Temp Measurement\n(Inlet)', fontsize=24)
        ax4.set_ylabel('Temp Measurement\n(Inlet)', fontsize=24)

        ax1.set_xlabel('Time', fontsize=24)

        ax1.set_yticks([-5, 0, 5, 10, 15])
        ax2.set_yticks([-5, 0, 5, 10, 15])
        ax3.set_yticks([-5, 0, 5, 10, 15])
        ax4.set_yticks([-5, 0, 5, 10, 15])

        ax1.set_ylim([0, 15])
        ax2.set_ylim([0, 15])
        ax3.set_ylim([0, 15])
        ax4.set_ylim([0, 15])

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

        plt.tight_layout()
        plt.savefig("{}/OutdoorInlet_Outdoor_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingOutdoorSystem(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        # 측정을 하지 않은 곳은 다 0으로 입력하였다.
        solve = self._outunitData.fillna(0)
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index < plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)
        ax6 = ax2.twinx()
        ax7 = ax3.twinx()
        ax8 = ax4.twinx()

        ax1.plot(tt, solve['total_indoor_capa'].tolist(), 'g-', linewidth='2', drawstyle='steps-post')
        ax2.plot(tt, solve['suction_temp1'].tolist(), 'b-', linewidth='2', drawstyle='steps-post')
        ax2.plot(tt, solve['discharge_temp1'].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax3.plot(tt, solve['eev1'].tolist(), 'k-', linewidth='2', drawstyle='steps-post')
        ax4.plot(tt, solve['value'].tolist(), 'k-', linewidth='2', drawstyle='steps-post')
        ax6.plot(tt, solve['fan_step'].tolist(), 'k-', linewidth='2', drawstyle='steps-post')
        ax7.plot(tt, solve['comp1'].tolist(), 'r-', linewidth='2',alpha=0.8, drawstyle='steps-post')
        ax7.plot(tt, solve['comp2'].tolist(), 'b--', linewidth='2',alpha=0.8, drawstyle='steps-post')
        ax8.plot(tt, solve['comp_current_frequency1'].tolist(), 'r-', linewidth='2', alpha=0.8, drawstyle='steps-post')
        ax8.plot(tt, solve['comp_current_frequency2'].tolist(), 'b--', linewidth='2', alpha=0.8, drawstyle='steps-post')

        gap = 120 # 09~18 : 120
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax4.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=22)
        ax2.tick_params(axis="x", labelsize=22)
        ax3.tick_params(axis="x", labelsize=22)
        ax4.tick_params(axis="x", labelsize=22)

        ax1.tick_params(axis="y", labelsize=22)
        ax2.tick_params(axis="y", labelsize=22)
        ax3.tick_params(axis="y", labelsize=22)
        ax4.tick_params(axis="y", labelsize=22)
        ax6.tick_params(axis="y", labelsize=22)
        ax7.tick_params(axis="y", labelsize=22)
        ax8.tick_params(axis="y", labelsize=22)

        ax1.set_ylabel('Capacity', fontsize=24)
        ax2.set_ylabel('Temperature', fontsize=24)
        ax3.set_ylabel('EEV', fontsize=24)
        ax4.set_ylabel('Power', fontsize=24)
        ax6.set_ylabel('Fan Step', fontsize=24)
        ax7.set_ylabel('Compressor Signal', fontsize=24)
        ax8.set_ylabel('Frequency', fontsize=24)

        ax4.set_xlabel('Time', fontsize=24)

        ax1.set_yticks([0, 250, 500, 750, 1000])
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        ax3.set_yticks([0, 500, 1000, 1500, 2000, 2500])
        ax4.set_yticks([0, 10000, 20000, 30000])
        ax6.set_yticks([0, 25, 50])
        ax7.set_yticks([0, 1])
        ax8.set_yticks([0, 25, 50, 75, 100])


        # ax1.set_ylim([])
        ax2.set_ylim([-100, 100])
        ax3.set_ylim([-1500, 2500])
        ax4.set_ylim([-30000, 30000])
        ax6.set_ylim([0, 200])
        ax7.set_ylim([0, 5])
        ax8.set_ylim([0, 300])

        ax1.legend(['Total indoor capacity'], fontsize=18)
        ax2.legend(['Suction Temperature', 'Discharge Temperature'], fontsize=18, loc='center right')
        ax3.legend(['Expansion value Opening'], fontsize=18)

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

        plt.tight_layout()
        plt.savefig("{}/OutdoorSystem_Outdoor_{}.png".format(save, out_unit))
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


TIME = 'updated_time'
start ='2022-02-14' #데이터 시작시간
end = '2022-02-14' #데이터 끝시간

DC = DataCorrection(TIME=TIME, start=start, end=end)
for i in [3069]:
    DC.Visualizing(out_unit=i)
