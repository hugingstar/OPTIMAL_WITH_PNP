import pandas as pd
import matplotlib.pyplot as plt
import os

"""
코드 실행하기 전에 법전원 실험 데이터 정리 방법 확인바랍니다.
22-05-14 : 저장하기
"""

class DataCorrection:
    def __init__(self, TIME, start, end):
        self.DATA_PATH = "H:/삼성전자/법전원 실험 정리"
        self.SAVE_PATH = "H:/삼성전자/법전원 실험 정리/Results"
        self.TIME = TIME

        """저장할 폴더 이름 생성"""
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
        self.create_folder('{}/{}'.format(self.SAVE_PATH, self.folder_name))  # Results, 날짜 폴더 생성

        """Volume Flow Rate"""
        self._OutAirVolumePath = "{}/{}/VolumeFlowRate.csv".format(self.DATA_PATH, self.folder_name)
        self._OutAirVolumeData = pd.read_csv(self._OutAirVolumePath)
        self._OutAirVolumeData['Date'] = self.folder_name
        self._OutAirVolumeData[self.TIME] = self._OutAirVolumeData['Date'] + ' ' + self._OutAirVolumeData['Time']
        self._OutAirVolumeData.index = pd.to_datetime(self._OutAirVolumeData[self.TIME])
        self._OutAirVolumeData.drop(labels=['Date', 'Time', 'updated_time'], axis=1, inplace=True)
        self._OutAirVolumeData = self._OutAirVolumeData.resample('1T').mean()
        self._OutAirVolumeData.to_csv("{}/{}/AirVolumeData.csv".format(self.SAVE_PATH, self.folder_name))

        """Outlet Air Temperature Sensor"""
        # self._OutAirTempPath = "{}/{}/OutletAirTemp.csv".format(self.DATA_PATH, self.folder_name)
        # self._OutAirTempData = pd.read_csv(self._OutAirTempPath)
        # self._OutAirTempData[self.TIME] = self._OutAirTempData['Date'] + ' ' + self._OutAirTempData['Time']
        # self._OutAirTempData.index = pd.to_datetime(self._OutAirTempData[self.TIME])
        # self._OutAirTempData.drop(labels=['Date', 'Time', 'updated_time'], axis=1, inplace=True)
        # self._OutAirTempData = self._OutAirTempData.resample('1T').mean()
        # self._OutAirTempData.to_csv("{}/{}/OutletAirTemp.csv".format(self.SAVE_PATH, self.folder_name))

        """HOBOSensor1 : Inlet"""
        self.HOBO_inlet_Path = "{}/{}/AmbientHOBO.csv".format(self.DATA_PATH, self.folder_name)
        self.HOBO_inlet_Data = pd.read_csv(self.HOBO_inlet_Path)
        self.HOBO_inlet_Data.index = pd.to_datetime(self.HOBO_inlet_Data[self.TIME])
        self.HOBO_inlet_Data.drop(labels=['updated_time'], axis=1, inplace=True)
        self.HOBO_inlet_Data = self.HOBO_inlet_Data.resample('1T').mean()
        self.HOBO_inlet_Data.to_csv("{}/{}/InletHOBO.csv".format(self.SAVE_PATH, self.folder_name))


        """HOBOSensor2 : Outlet"""
        self.HOBO_outlet_Path = "{}/{}/OutletHOBO.csv".format(self.DATA_PATH, self.folder_name)
        self.HOBO_outlet_Data = pd.read_csv(self.HOBO_outlet_Path)
        self.HOBO_outlet_Data.index = pd.to_datetime(self.HOBO_outlet_Data[self.TIME])
        self.HOBO_outlet_Data.drop(labels=['updated_time'], axis=1, inplace=True)
        self.HOBO_outlet_Data = self.HOBO_outlet_Data.resample('1T').mean()
        self.HOBO_outlet_Data.to_csv("{}/{}/OutletHOBO.csv".format(self.SAVE_PATH, self.folder_name))

        """Data integration"""
        # self.IntegData = pd.concat([self._OutAirVolumeData, self._OutAirTempData], axis=1)
        self.IntegData = pd.concat([self._OutAirVolumeData], axis=1)
        self.IntegData = pd.concat([self.IntegData, self.HOBO_inlet_Data], axis=1)
        self.IntegData = pd.concat([self.IntegData, self.HOBO_outlet_Data], axis=1)
        self.IntegData.sort_index(inplace=True)
        self.IntegData.to_csv("{}/{}/IntegData.csv".format(self.SAVE_PATH, self.folder_name))

        print(self.IntegData)

        save = "{}/{}".format(self.SAVE_PATH, self.folder_name)
        st = '11:15:00'  # '11:00:00'
        et = '12:15:00'  # '14:31:00'
        out_unit = 'lawbldg'

        self.PlottingMesurement(plt_ST=st, plt_ET=et, save=save, out_unit=out_unit)

    def PlottingMesurement(self, plt_ST, plt_ET, save, out_unit):
        plt.rcParams["font.family"] = "Times New Roman"
        solve = self.IntegData.fillna(0)
        solve = solve[solve.index >= self.folder_name + ' ' + plt_ST]
        solve = solve[solve.index <= self.folder_name + ' ' + plt_ET]

        tt0 = solve.index.tolist()
        tt = []
        for i in range(len(tt0)):
            k = str(tt0[i])[8:16]
            tt.append(k)

        fig = plt.figure(figsize=(25, 20))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        # ax4 = fig.add_subplot(4, 1, 4)
        print(solve.columns)

        ax1.plot(tt, solve['VolumeFlowRate'].tolist(), 'g-', linewidth='2', drawstyle='steps-post')
        # ax2.plot(tt, solve['Temp1'].tolist(), 'b-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        # ax2.plot(tt, solve['Temp2'].tolist(), 'r-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        # ax2.plot(tt, solve['Temp3'].tolist(), 'b-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        # ax2.plot(tt, solve['Temp4'].tolist(), 'r-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        # ax2.plot(tt, solve['Temp5'].tolist(), 'b-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        # ax2.plot(tt, solve['Temp6'].tolist(), 'r-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        ax2.plot(tt, solve['Temp_Inlet'].tolist(), 'b-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        ax2.plot(tt, solve['Temp_Outlet'].tolist(), 'r-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        ax3.plot(tt, solve['RH_Inlet'].tolist(), 'b-', linewidth='2', drawstyle='steps-post', alpha= 0.6)
        ax3.plot(tt, solve['RH_Outlet'].tolist(), 'r-', linewidth='2', drawstyle='steps-post', alpha= 0.6)

        ax1.legend(['Air Volume Rate({})'.format('$m^{3}/h$')], fontsize=18, loc='upper right')
        # ax2.legend(['Temperature({})'.format('$^{\circ}C$')], fontsize=18, loc='upper right')
        ax2.legend(['Temperature Inlet(HOBO, {})'.format('$^{\circ}C$'), 'Temperature Outlet(HOBO, {})'.format('$^{\circ}C$')], fontsize=18, loc='upper right', ncol=2)
        ax3.legend(['Humidity Inlet (HOBO, RH %)', 'Humidity Outlet (HOBO, RH %)'], fontsize=18, loc='upper right', ncol=2)

        gap = 15 #60
        ax1.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax2.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        ax3.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])
        # ax4.set_xticks([tt[i] for i in range(len(tt)) if i % gap == 0 or tt[i] == tt[-1]])

        ax1.tick_params(axis="x", labelsize=24)
        ax2.tick_params(axis="x", labelsize=24)
        ax3.tick_params(axis="x", labelsize=24)
        # ax4.tick_params(axis="x", labelsize=24)

        ax1.tick_params(axis="y", labelsize=24)
        ax2.tick_params(axis="y", labelsize=24)
        ax3.tick_params(axis="y", labelsize=24)
        # ax4.tick_params(axis="y", labelsize=24)

        ax1.set_ylabel('Air Volume Rate', fontsize=28)
        # ax2.set_ylabel('Temperature', fontsize=28)
        ax2.set_ylabel('Air temperature', fontsize=28)
        ax3.set_ylabel('Humidity', fontsize=28)

        # ax3.set_yticks([0, 10, 200])

        ax2.set_ylim([0, 50])
        ax3.set_ylim([0, 55])
        # ax4.set_ylim([0, 60])

        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)
        # ax4.autoscale(enable=True, axis='x', tight=True)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        # ax4.grid()

        plt.tight_layout()
        plt.savefig("{}/Measurement_{}_{}.png".format(save, out_unit, plt_ST[0:2]))
        plt.show()
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
start = '2022-05-22'
end = '2022-05-22'

DC = DataCorrection(TIME=TIME, start=start, end=end)
