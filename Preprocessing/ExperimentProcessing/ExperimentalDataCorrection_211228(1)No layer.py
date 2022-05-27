import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
"""
이 프로그램은 
"""

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
        print(self._outdUnitPath)
        self._outunitData = pd.read_csv(self._outdUnitPath, index_col=self.TIME)
        self.col = list(pd.Series(list(self._outunitData.columns))[
                             pd.Series(list(self._outunitData.columns)).str.contains(pat='value', case=False)]) # 전력
        self._outunitData[self._outunitData[self.col] < 0] = None
        self._outunitData.fillna(method='ffill', inplace=True)
        self._outunitData.to_csv("{}/Outdoor_{}.csv".format(save, out_unit))

        # Outdoor velocity1
        self._outFlowPath = "{}/{}/{}{} flow1.csv".format(self.DATA_PATH, self.folder_name, self.start_month,self.start_date)
        self._outFlowData = pd.read_csv(self._outFlowPath)
        self.col1 = list(pd.Series(list(self._outFlowData.columns))[
                             pd.Series(list(self._outFlowData.columns)).str.contains(pat='Time', case=False)])
        self.col2 = list(pd.Series(list(self._outFlowData.columns))[
                             pd.Series(list(self._outFlowData.columns)).str.contains(pat='velocity', case=False)])

        self.OutdoorVelocity = self._outFlowData[[self.col1[0], self.col2[0]]]
        self.OutdoorVelocity = self.OutdoorVelocity.rename(columns={self.col1[0]: self.TIME, self.col2[0]:'outdoor_velocity'})
        print(self.col1[0], self.col2[0])
        for _o in [1, 2, 3, 4, 5]: # 앞쪽 3개
            print(self.col1[_o], self.col2[_o])
            tem = self._outFlowData[[self.col1[_o], self.col2[_o]]]
            tem = tem.rename(columns={self.col1[_o]: self.TIME, self.col2[_o]: 'outdoor_velocity'})
            self.OutdoorVelocity = pd.concat([self.OutdoorVelocity, tem], axis=0, ignore_index=True)
        self.OutdoorVelocity['Date'] = self.folder_name
        self.OutdoorVelocity[self.TIME] = self.OutdoorVelocity['Date'] + ' ' + self.OutdoorVelocity[self.TIME]
        self.OutdoorVelocity[self.OutdoorVelocity['outdoor_velocity'] < 0] = None
        self.OutdoorVelocity.fillna(method='ffill', inplace=True)
        self.OutdoorVelocity.index = pd.to_datetime(self.OutdoorVelocity[self.TIME])
        self.OutdoorVelocity = self.OutdoorVelocity.resample('1T').mean()

        # Outdoor velocity2
        # self._outFlowPath2 = "{}/{}/{}{} flow2.csv".format(self.DATA_PATH, self.folder_name, self.start_month,
        #                                                  self.start_date)
        # self._outFlowData2 = pd.read_csv(self._outFlowPath2)
        # self.col1 = list(pd.Series(list(self._outFlowData2.columns))[
        #                      pd.Series(list(self._outFlowData2.columns)).str.contains(pat='Time', case=False)])
        # self.col2 = list(pd.Series(list(self._outFlowData2.columns))[
        #                      pd.Series(list(self._outFlowData2.columns)).str.contains(pat='velocity', case=False)])
        # self.OutdoorVelocity2 = self._outFlowData2[[self.col1[0], self.col2[0]]]
        # self.OutdoorVelocity2 = self.OutdoorVelocity2.rename(columns={self.col1[0]: self.TIME, self.col2[0]: 'outdoor_velocity'})
        # print(self.col1[0], self.col2[0])
        # for _o in [1, 2]:  # 앞쪽 3개
        #     print(self.col1[_o], self.col2[_o])
        #     tem = self._outFlowData2[[self.col1[_o], self.col2[_o]]]
        #     tem = tem.rename(columns={self.col1[_o]: self.TIME, self.col2[_o]: 'outdoor_velocity'})
        #     self.OutdoorVelocity2 = pd.concat([self.OutdoorVelocity2, tem], axis=0, ignore_index=True)
        # self.OutdoorVelocity2['Date'] = self.folder_name
        # self.OutdoorVelocity2[self.TIME] = self.OutdoorVelocity2['Date'] + ' ' + self.OutdoorVelocity2[self.TIME]
        # self.OutdoorVelocity2[self.OutdoorVelocity2['outdoor_velocity'] < 0] = None
        # self.OutdoorVelocity2.fillna(method='ffill', inplace=True)
        # self.OutdoorVelocity2.index = pd.to_datetime(self.OutdoorVelocity2[self.TIME])
        # self.OutdoorVelocity2 = self.OutdoorVelocity2.resample('1T').mean()

        self.IntegOutdoorVelocity = pd.concat([self.OutdoorVelocity], axis=0)
        self.IntegOutdoorVelocity.to_csv("{}/Outdoor_{}_Measure_OutdoorVelocity.csv".format(save, out_unit))

        #Outdoor Volume1
        # self._outVolumePath = "{}/{}/{}{} volume1.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        # self._outVolumeData = pd.read_csv(self._outVolumePath)
        # self.col = list(pd.Series(list(self._outVolumeData.columns))[
        #                      pd.Series(list(self._outVolumeData.columns)).str.contains(pat='Time', case=False)])
        # self._outVolumeData = self._outVolumeData.rename(columns={self.col[0]: self.TIME, 'volume' : 'outdoor_volume'})
        # self._outVolumeData['Date'] = self.folder_name
        # self._outVolumeData[self.TIME] = self._outVolumeData[['Date', self.TIME]].apply(' '.join, axis=1)
        # self._outVolumeData.drop(columns=['Date'], inplace=True)
        # self._outVolumeData[self.TIME] = pd.to_datetime(self._outVolumeData[self.TIME])
        # self._outVolumeData[self._outVolumeData['outdoor_volume'] < 0] = None
        # self._outVolumeData.fillna(method='ffill', inplace=True)
        # self._outVolumeData.set_index(self.TIME, inplace=True)
        # self._outVolumeData = self._outVolumeData.resample('1T').mean()

        #Outdoor Volume2
        # self._outVolumePath2 = "{}/{}/{}{} volume2.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        # self._outVolumeData2 = pd.read_csv(self._outVolumePath2)
        # self.col = list(pd.Series(list(self._outVolumeData2.columns))[
        #                      pd.Series(list(self._outVolumeData2.columns)).str.contains(pat='Time', case=False)])
        # self._outVolumeData2 = self._outVolumeData2.rename(columns={self.col[0]: self.TIME, 'volume' : 'outdoor_volume'})
        # self._outVolumeData2['Date'] = self.folder_name
        # self._outVolumeData2[self.TIME] = self._outVolumeData2[['Date', self.TIME]].apply(' '.join, axis=1)
        # self._outVolumeData2.drop(columns=['Date'], inplace=True)
        # self._outVolumeData2[self.TIME] = pd.to_datetime(self._outVolumeData2[self.TIME])
        # self._outVolumeData2[self._outVolumeData2['outdoor_volume'] < 0] = None
        # self._outVolumeData2.fillna(method='ffill', inplace=True)
        # self._outVolumeData2.set_index(self.TIME, inplace=True)
        # self._outVolumeData2 = self._outVolumeData2.resample('1T').mean()

        # self.IntegOutdoorVolume = pd.concat([self._outVolumeData, self._outVolumeData2], axis=0)
        # self.IntegOutdoorVolume.to_csv("{}/Outdoor_{}_Measure_OutdoorVolume.csv".format(save, out_unit))

        #Outdoor Temperature Measurement
        self._outTempPath = "{}/{}/{}{} temp1.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        self._outTempData = pd.read_csv(self._outTempPath)
        self._outTempData[self.TIME] = self._outTempData[['Date', 'Time']].apply(' '.join, axis=1)
        self._outTempData.drop(columns=['Date', 'Time'], inplace=True)
        self._outTempData[self.TIME] = pd.to_datetime(self._outTempData[self.TIME])
        self._outTempData.fillna(0, inplace=True)
        self._outTempData.set_index(self.TIME, inplace=True)
        self._outTempData = self._outTempData.resample('1T').mean()

        # self._outTempPath2 = "{}/{}/{}{} temp2.csv".format(self.DATA_PATH, self.folder_name, self.start_month, self.start_date)
        # self._outTempData2 = pd.read_csv(self._outTempPath2)
        # self._outTempData2[self.TIME] = self._outTempData2[['Date', 'Time']].apply(' '.join, axis=1)
        # self._outTempData2.drop(columns=['Date', 'Time', 'sec'], inplace=True)
        # self._outTempData2[self.TIME] = pd.to_datetime(self._outTempData2[self.TIME])
        # self._outTempData2.fillna(0, inplace=True)
        # self._outTempData2.set_index(self.TIME, inplace=True)
        # self._outTempData2 = self._outTempData2.resample('1T').mean()

        self.IntegTemp = pd.concat([self._outTempData], axis=1) #self._outTempData2
        self.IntegTemp.to_csv("{}/Outdoor_{}_Measure_OutdoorTemp.csv".format(save, out_unit))

        self.OutIntegData = self._outunitData.join(self.IntegOutdoorVelocity, how='left')
        # self.OutIntegData = self.OutIntegData.join(self.IntegOutdoorVolume, how='left')
        self.OutIntegData = self.OutIntegData.join(self.IntegTemp, how='left')

        # self.OutIntegData = self.OutIntegData.fillna(0)
        print(self.OutIntegData.shape)
        if "Unnamed: 0" in self.OutIntegData:
            self.OutIntegData.drop(columns=['Unnamed: 0'], inplace=True)
        self.OutIntegData.to_csv("{}/OutIntegrationData_{}.csv".format(save, out_unit))

        """Plot Time Range"""
        # 그림 그릴 부분의 시작시간(plt_ST) - 끝시간(plt_ET)
        st = '00:00:00' #'11:00:00'
        et = '23:59:00' #'14:31:00'
        self.gap = 240
        # Plotting
        self.PlottingOutdoorSystem(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)

        # Measurement
        self.PlottingOutdoorMeasurement(plt_ST=self.folder_name + ' ' + st, plt_ET=self.folder_name + ' ' + et, save=save, out_unit=out_unit)

    def PlottingOutdoorMeasurement(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self.OutIntegData.fillna(0)
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

        ax1.plot(tt, solve['outdoor_velocity'].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax2.plot(tt, solve['fan_step'].tolist(), 'k', linewidth='2', drawstyle='steps-post')

        # Inlet
        cd_col_list = []
        for _ in [21, 22]:
            cd_col_list.append('point{}'.format(_))
            ax3.plot(tt, solve['point{}'.format(_)].tolist(), 'k', alpha=0.2, linewidth='2', drawstyle='steps-post')
        avg_temp_list = solve[cd_col_list].mean(axis=1)
        lineavg1, = ax3.plot(tt, avg_temp_list, 'k', linewidth='4', drawstyle='steps-post', label='Inlet Average Temperature')

        df1 = pd.DataFrame({'Air_inlet_Average_Temperature': avg_temp_list})
        df1.to_csv("{}/Outdoor_{}_inlet_AvgTemp.csv".format(save, out_unit))
        print(df1)

        # Outlet
        cd_col_list = []
        for _ in [1, 2, 3, 4]:
            cd_col_list.append('point{}'.format(_))
            # 가장 큰 것 : 19
            ax3.plot(tt, solve['point{}'.format(_)].tolist(), 'b', alpha=0.2, linewidth='2', drawstyle='steps-post')
        avg_temp_list = solve[cd_col_list].mean(axis=1)
        lineavg2, = ax3.plot(tt, avg_temp_list, 'b', linewidth='4', drawstyle='steps-post', label='Outlet Average Temperature')

        df2 = pd.DataFrame({'Air_outlet_Average_Temperature': avg_temp_list})
        df2.to_csv("{}/Outdoor_{}_outlet_AvgTemp.csv".format(save, out_unit))
        print(df2)

        ax1.legend(['Air Velocity ({})'.format('$m/s$')], fontsize=18, loc='upper left')
        ax2.legend(['Fans steps'], fontsize=18, loc='upper left')
        ax3.legend(handles=[lineavg1, lineavg2], fontsize=18, loc='upper left')

        gap = self.gap   # 09~18 : 120
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=24)
        ax2.tick_params(axis="x", labelsize=24)
        ax3.tick_params(axis="x", labelsize=24)

        ax1.tick_params(axis="y", labelsize=24)
        ax2.tick_params(axis="y", labelsize=24)
        ax3.tick_params(axis="y", labelsize=24)

        ax1.set_ylabel('Air Velocity', fontsize=28)
        ax2.set_ylabel('Fan steps', fontsize=28)
        ax3.set_ylabel('Temperature', fontsize=28)

        ax3.set_xlabel('Time', fontsize=28)

        ax1.set_yticks([0, 1, 2, 3, 4, 5])
        ax2.set_yticks([0, 10, 20, 30, 40, 50, 60])
        ax3.set_yticks([0, 5, 10, 15, 20, 25, 30])

        ax1.set_ylim([0, 5])
        ax2.set_ylim([0, 60])
        ax3.set_ylim([0, 30])

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.tight_layout()
        plt.savefig("{}/Outdoor_Measurement_{}.png".format(save, out_unit))
        # plt.show()
        plt.clf()

    def PlottingOutdoorSystem(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        # 측정을 하지 않은 곳은 다 0으로 입력하였다.
        solve = self.OutIntegData #.fillna(0)
        solve = solve[solve.index >= plt_ST]
        solve = solve[solve.index < plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(25, 40))
        ax1 = fig.add_subplot(8, 1, 1)
        ax2 = fig.add_subplot(8, 1, 2)
        ax3 = fig.add_subplot(8, 1, 3)
        ax4 = fig.add_subplot(8, 1, 4)
        ax5 = fig.add_subplot(8, 1, 5)
        ax6 = fig.add_subplot(8, 1, 6)
        ax7 = fig.add_subplot(8, 1, 7)
        ax8 = fig.add_subplot(8, 1, 8)

        ax1.plot(tt, solve['total_indoor_capa'].tolist(), 'g-', linewidth='2', drawstyle='steps-post')
        ax2.plot(tt, solve['fan_step'].tolist(), 'k--', linewidth='2', drawstyle='steps-post')
        ax3.plot(tt, solve['eev1'].tolist(), 'k-', linewidth='2', drawstyle='steps-post')
        ax4.plot(tt, solve['high_pressure'].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax4.plot(tt, solve['low_pressure'].tolist(), 'b--', linewidth='2', drawstyle='steps-post')
        ax5.plot(tt, solve['value'].tolist(), 'k-', linewidth='2', drawstyle='steps-post')

        ax6.plot(tt, solve['comp1'].tolist(), 'r-', linewidth='2', alpha=0.8, drawstyle='steps-post')
        ax6.plot(tt, solve['comp2'].tolist(), 'b--', linewidth='2', alpha=0.8, drawstyle='steps-post')
        ax7.plot(tt, solve['comp_current_frequency1'].tolist(), 'r-', linewidth='2', alpha=0.8, drawstyle='steps-post')
        ax7.plot(tt, solve['comp_current_frequency2'].tolist(), 'b--', linewidth='2', alpha=0.8, drawstyle='steps-post')

        ax8.plot(tt, solve['discharge_temp1'].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax8.plot(tt, solve['discharge_temp2'].tolist(), 'g-', linewidth='2', drawstyle='steps-post', alpha=0.8)
        ax8.plot(tt, solve['suction_temp1'].tolist(), 'b--', linewidth='2', drawstyle='steps-post')

        gap = self.gap   # 09~18 : 120
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax4.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax5.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax6.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax7.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax8.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=24)
        ax2.tick_params(axis="x", labelsize=24)
        ax3.tick_params(axis="x", labelsize=24)
        ax4.tick_params(axis="x", labelsize=24)
        ax5.tick_params(axis="x", labelsize=24)
        ax6.tick_params(axis="x", labelsize=24)
        ax7.tick_params(axis="x", labelsize=24)
        ax8.tick_params(axis="x", labelsize=24)

        ax1.tick_params(axis="y", labelsize=24)
        ax2.tick_params(axis="y", labelsize=24)
        ax3.tick_params(axis="y", labelsize=24)
        ax4.tick_params(axis="y", labelsize=24)
        ax5.tick_params(axis="y", labelsize=24)
        ax6.tick_params(axis="y", labelsize=24)
        ax7.tick_params(axis="y", labelsize=24)
        ax8.tick_params(axis="y", labelsize=24)

        ax1.set_ylabel('Total Capacity', fontsize=28)
        ax2.set_ylabel('Fan Steps', fontsize=28)
        ax3.set_ylabel('EEV', fontsize=28)
        ax4.set_ylabel('Pressure', fontsize=28)
        ax5.set_ylabel('Power', fontsize=28)
        ax6.set_ylabel('Compressor Signal', fontsize=28)
        ax7.set_ylabel('Frequency', fontsize=28)
        ax8.set_ylabel('Temperature', fontsize=28)

        ax8.set_xlabel('Time', fontsize=28)

        ax1.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax3.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        ax4.set_yticks([0, 10, 20, 30, 40, 50])
        # ax5.set_yticks([0, 10000, 20000, 30000, 40000, 50000])
        ax6.set_yticks([0, 1])
        ax7.set_yticks([0, 25, 50, 75, 100])
        ax8.set_yticks([0, 25, 50, 75, 100])

        ax1.set_ylim([0, max(solve['total_indoor_capa'].tolist()) * 1.1])
        ax2.set_ylim([0, 100])
        ax3.set_ylim([0, max(solve['eev1'].tolist()) * 1.2])
        ax4.set_ylim([0, max(solve['high_pressure'].tolist()) * 1.5])
        # ax5.set_ylim([0, max(solve['value'].tolist()) * 1.5])
        ax6.set_ylim([0, 2])
        ax7.set_ylim([0, 100])
        ax8.set_ylim([-10, max(solve['discharge_temp1'].tolist()) * 1.5])

        ax1.legend(['Total indoor capacity'], fontsize=18)
        ax2.legend(['Fan Steps'], fontsize=18, loc='upper right')
        ax3.legend(['Expansion value Opening'], fontsize=18)
        ax4.legend(['High Pressure', 'Low Pressure'], fontsize=18, loc='upper right', ncol=2)
        ax5.legend(['Power'], fontsize=18)
        ax8.legend(['Discharge Temperature 1', 'Discharge Temperature 2', 'Suction Temperature'], fontsize=18,
                   loc='upper right', ncol=3)

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)
        ax5.autoscale(enable=True, axis='x', tight=True)
        ax6.autoscale(enable=True, axis='x', tight=True)
        ax7.autoscale(enable=True, axis='x', tight=True)
        ax8.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()
        ax6.grid()
        ax7.grid()
        ax8.grid()

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
start ='2021-12-28' #데이터 시작시간
end = '2021-12-28' #데이터 끝시간

DC = DataCorrection(TIME=TIME, start=start, end=end)
for i in [3066]:
    DC.Visualizing(out_unit=i)
