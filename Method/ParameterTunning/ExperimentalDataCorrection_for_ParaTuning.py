import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os

class DataCorrection:
    def __init__(self, TIME, start, end):
        "파일을 호출할 경로"
        self.DATA_PATH = "D:/OPTIMAL/Data/ParameterTuning"
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
        self.create_folder('{}/ParameterTunning'.format(self.SAVE_PATH))  # Deepmodel 폴더를 생성

    def Visualizing(self, out_unit):
        # 건물 인식(딕셔너리에서 포함된 건물로 인식한다.)
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        save = "{}/ParameterTunning/{}".format(self.SAVE_PATH, self.folder_name)
        self.create_folder(save)
        # Outdoor unit data from biot
        self._outdUnitPath = "{}/{}/Outdoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit)
        print(self._outdUnitPath)
        self._outunitData = pd.read_csv(self._outdUnitPath, index_col=self.TIME)
        self.col = list(pd.Series(list(self._outunitData.columns))[
                             pd.Series(list(self._outunitData.columns)).str.contains(pat='value', case=False)]) # 전력
        self._outunitData[self._outunitData[self.col] < 0] = None
        # self._outunitData.fillna(method='ffill', inplace=True) #선택적 옵션
        self._outunitData.to_csv("{}/Outdoor_{}.csv".format(save, out_unit))

        self.OutIntegData = self._outunitData

        # self.OutIntegData = self.OutIntegData.fillna(0)
        if "Unnamed: 0" in self.OutIntegData:
            self.OutIntegData.drop(columns=['Unnamed: 0'], inplace=True)
        self.OutIntegData.to_csv("{}/OutIntegrationData_{}.csv".format(save, out_unit))

        """Plot Time Range"""
        # 그림 그릴 부분의 시작시간(plt_ST) - 끝시간(plt_ET)
        st = '2022-02-01 00:00:00' #'11:00:00'
        et = '2022-02-28 23:59:00' #'14:31:00'
        self.gap = 9000 #x축 간격

        # Plotting System
        # self.PlottingOutdoorSystem(plt_ST=self.start + ' ' + st, plt_ET=self.end + ' ' + et, save=save, out_unit=out_unit)

        self.PlottingOutdoorSystem(plt_ST=st, plt_ET=et, save=save, out_unit=out_unit)

    def PlottingOutdoorSystem(self, plt_ST, plt_ET, save, out_unit):
        """
        biot 데이터를 그리는 함수
        :param plt_ST: 그림 그릴 시작시간
        :param plt_ET: 그림 그릴 끝시간
        :param save: 저장
        :param out_unit: 실외기 번호
        :return: 그림 저장되고 따로 리턴값은 없음
        """
        plt.rcParams["font.family"] = "Times New Roman"
        # 측정을 하지 않은 곳은 다 0으로 입력하였다.
        solve = self.OutIntegData.fillna(0)
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

        self.TotIndCapa = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='total_indoor_capa', case=False)])
        self.FanStep = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='fan_step', case=False)])
        self.outdoor_eev = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='eev1', case=False)])
        self.HighPressure = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='high_pressure', case=False)])
        self.LowPressure = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='low_pressure', case=False)])
        self.Power = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='value', case=False)])
        self.CompSignal = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='comp', case=False)])
        self.frequency = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='comp_current_frequency', case=False)])
        self.DischargeTemp = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='comp_current_frequency', case=False)])
        self.SuctionTemp = list(pd.Series(list(solve.columns))[
                             pd.Series(list(solve.columns)).str.contains(pat='suction_temp', case=False)])

        ax1.plot(tt, solve[self.TotIndCapa[0]].tolist(), 'g-', linewidth='2', drawstyle='steps-post')
        ax2.plot(tt, solve[self.FanStep[0]].tolist(), 'k--', linewidth='2', drawstyle='steps-post')
        ax3.plot(tt, solve[self.outdoor_eev[0]].tolist(), 'k-', linewidth='2', drawstyle='steps-post')
        ax4.plot(tt, solve[self.HighPressure[0]].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax4.plot(tt, solve[self.LowPressure[0]].tolist(), 'b--', linewidth='2', drawstyle='steps-post')
        ax5.plot(tt, solve[self.Power[0]].tolist(), 'k-', linewidth='2', drawstyle='steps-post')

        ax6.plot(tt, solve[self.CompSignal[0]].tolist(), 'r-', linewidth='2', alpha=0.8, drawstyle='steps-post')
        ax6.plot(tt, solve[self.CompSignal[1]].tolist(), 'b--', linewidth='2', alpha=0.8, drawstyle='steps-post')
        ax7.plot(tt, solve[self.frequency[0]].tolist(), 'r-', linewidth='2', alpha=0.8, drawstyle='steps-post')
        ax7.plot(tt, solve[self.frequency[1]].tolist(), 'b--', linewidth='2', alpha=0.8, drawstyle='steps-post')

        ax8.plot(tt, solve[self.DischargeTemp[0]].tolist(), 'r-', linewidth='2', drawstyle='steps-post')
        ax8.plot(tt, solve[self.DischargeTemp[1]].tolist(), 'g-', linewidth='2', drawstyle='steps-post', alpha=0.8)
        ax8.plot(tt, solve[self.SuctionTemp[0]].tolist(), 'b--', linewidth='2', drawstyle='steps-post')

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

        ax1.set_ylim([0, max(solve[self.TotIndCapa[0]].tolist()) * 1.1])
        ax2.set_ylim([0, 100])
        ax3.set_ylim([0, max(solve[self.outdoor_eev[0]].tolist()) * 1.2])
        ax4.set_ylim([0, max(solve[self.HighPressure[0]].tolist()) * 1.5])
        # ax5.set_ylim([0, max(solve['value'].tolist()) * 1.5])
        ax6.set_ylim([0, 2])
        ax7.set_ylim([0, 100])
        ax8.set_ylim([-10, max(solve[self.DischargeTemp[0]].tolist()) * 1.5])

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
start ='2021-11-01' #데이터 시작시간
end = '2022-02-28' #데이터 끝시간

DC = DataCorrection(TIME=TIME, start=start, end=end)
for i in [3066, 3065, 3067, 3069]:
    DC.Visualizing(out_unit=i)
