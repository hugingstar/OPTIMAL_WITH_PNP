import pymysql
import pandas as pd
import datetime
import os

class ACQUISITION():
    def __init__(self, start, end, AnalysisObject):
        """원하는 데이터"""
        self.DATA_PATH = "D:/OPTIMAL/Data"
        self.AnalysisObject = AnalysisObject

        self.TIME = "updated_time"
        self.indoor_data = ["indoor_power",
                            "indoor_set_temp",
                            "current_room_temp",
                            "evain_temp",
                            "evaout_temp",
                            "indoor_fan_speed",
                            "eev",
                            "relative_capa_code"]

        # self.outdoor_data = ["total_indoor_capa",
        #                      "comp1", "comp2",
        #                      "comp_current_frequency1", "comp_current_frequency2",
        #                      "suction_temp1", "cond_out_temp1",
        #                      "discharge_temp1", "discharge_temp2", "discharge_temp3",
        #                      "double_tube_temp",
        #                      "outdoor_temperature",
        #                      "high_pressure", "low_pressure",
        #                      "eev1", "eev2", "eev3",
        #                      "ct1", "ct2", "ct3",
        #                      "value",
        #                      "fan_step"]

        self.outdoor_data = ["total_indoor_capa", "comp1", "comp2", "cond_out_temp1", "cond_out_temp2", "suction_temp1",
                             "suction_temp2","discharge_temp1", "discharge_temp2", "discharge_temp3",
                             "outdoor_temperature", "high_pressure", "low_pressure", "eev1", "eev2", "eev3",
                             "eev4", "eev5", "ct1", "ct2", "ct3", "double_tube_temp", "hot_gas_valve1", "hot_gas_valve2",
                             "outdoor_capacity", "liquid_bypass_valve", "evi_bypass_valve", "comp_current_frequency1",
                             "comp_current_frequency2", "comp_desired_frequency1",
                             "comp_desired_frequency2", "main_cooling_valve", "evi_eev", "fan_step", "comp_ipm1", "value"]

        # Dummy in outdoor
        # cond_out_temp2, suction_temp2, evi_eev

        #옵티멀 스타트에 필요한것만 모아논 것
        # self.indoor_data = ["indoor_power", "indoor_set_temp", "current_room_temp", "relative_capa_code"]
        # self.outdoor_data = ["total_indoor_capa", "outdoor_temperature", "value"]

        """
        실내기 모든 컬럼(참고용)
        ['updated_time', 'indoor_power', 'evain_temp', 'evaout_temp',
       'indoor_set_temp', 'current_room_temp', 'eev', 'indoor_fan_speed',
       'relative_capa_code']
        
        실외기 모든 컬럼(참고용)
        ['updated_time', 'total_indoor_capa', 'comp1', 'comp2', 'cond_out_temp1',
       'cond_out_temp2', 'suction_temp1', 'suction_temp2', 'discharge_temp1',
       'discharge_temp2', 'discharge_temp3', 'outdoor_temperature',
       'high_pressure', 'low_pressure', 'eev1', 'eev2', 'eev3', 'eev4', 'eev5',
       'ct1', 'ct2', 'ct3', 'value', 'double_tube_temp', 'hot_gas_valve1',
       'hot_gas_valve2', 'outdoor_capacity', 'liquid_bypass_valve',
       'evi_bypass_valve', 'comp_current_frequency1',
       'comp_current_frequency2', 'comp_desired_frequency1',
       'comp_desired_frequency2', 'main_cooling_valve', 'evi_eev', 'fan_step',
       'comp_ipm1']
       """

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

        """날짜 정보"""
        self.start_year = start[:4]
        self.start_month = start[5:7]
        self.start_date = start[8:10]
        self.end_year = end[:4]
        self.end_month = end[5:7]
        self.end_date = end[8:10]

        """디렉토리 생성"""
        self.folder_name = "{}-{}-{}".format(self.start_year, self.start_month, self.start_date)
        self.create_folder('{}/{}'.format(self.DATA_PATH, self.AnalysisObject, self.folder_name)) # 없으면 생성

        """MYSQL 접속"""
        self.db_conn = pymysql.connect(host="192.168.0.33",
                                       user='bsmart',
                                       password='bsmartlab419',
                                       database='biot',
                                       autocommit=True)
        self.cursor = self.db_conn.cursor()

        """(Appendix) SQL 내부에 TABLES 이름 확인"""
        # self.SHOW_DATABASE()

        # 현재재 날짜 확인
        self.today = datetime.datetime.now()
        print("Today : {}".format(self.today))

    def get_indoor_with_Fullsentences(self, out_unit):

        # 건물 이름
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        # Indoor Unit이 순서대로 반복
        for i in self.bldginfo[out_unit]:
            sql1 = "SELECT * FROM indoor_{} WHERE updated_time >= '{}-{}-{} {}:00:00' and updated_time <= '{}-{}-{} {}:59:00' AND time(updated_time) BETWEEN '00:00:00' AND '23:59:00'"\
                .format(i, self.start_year, self.start_month, self.start_date, '00', self.end_year, self.end_month, self.end_date, 23)
            df_org = pd.read_sql(sql1, self.db_conn)
            # print(df_org.columns)
            df1 = df_org[self.indoor_data]
            for j in df1.columns:
                df1.rename(columns={'{}'.format(j) : 'Bldg_{}/Outd_{}/Ind_{}/{}'.format(self.bldg_name, out_unit, i, j)}, inplace=True)
            df1 = df1.set_index(df_org[self.TIME])
            save = "{}/{}/{}/{}".format(self.DATA_PATH, self.AnalysisObject, self.folder_name, out_unit)
            self.create_folder(save)
            df1.to_csv(save + '\Outdoor_{}_Indoor_{}.csv'.format(out_unit, i))

    def get_outdoor_with_Fullsentences(self, out_unit):
        """
        :param dev: 실외기 디바이스 넘버 List로 넣기
        :param start: 시작일
        :param end: 종료일
        :param occ: 0시부터 23시까지
        :return: 실외기 데이터가 저장
        """
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        sql = "SELECT * FROM outdoor_{} WHERE updated_time >= '{}-{}-{} {}:00:00' and updated_time <= '{}-{}-{} {}:59:00'"\
            .format(out_unit, self.start_year, self.start_month, self.start_date, '00', self.end_year, self.end_month, self.end_date, '23')
        df_org = pd.read_sql(sql, self.db_conn)
        # print(df_org.columns)
        df = df_org[self.outdoor_data]

        #Meta data
        for j in df.columns:
            df.rename(columns={'{}'.format(j): 'Bldg_{}/Outd_{}/{}'.format(self.bldg_name, out_unit, j)}, inplace=True)

        df = df.set_index(df_org[self.TIME])
        df.to_csv("{}/{}/{}/Outdoor_{}.csv".format(self.DATA_PATH, self.AnalysisObject, self.folder_name, out_unit))

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: creating directory. ' + directory)

    def CLOSE_DATABASE(self):
        self.db_conn.close()

    def SHOW_DATABASE(self):
        with self.db_conn:
            with self.cursor as cur:
                cur.execute('SHOW TABLES')
                for data in cur:
                    print(data)

    def TIME_INDEX(self):
        # 시간 인덱스 컬럼명을 확인하고 싶을 때 사용
        return self.TIME

    def BLDG_INFO(self):
        # 현재 건물이 어떤 건물인가를 확인하고 싶을 때 사용
        return self.bldginfo

    def INDOOR_FEATURES(self):
        # 실내기 컬럼 이름을 확인할 때 사용
        return self.indoor_data

    def OUTDOOR_FEATURES(self):
        # 실외기 컬럼 이름을 확인할 때 사용
        return self.outdoor_data

    def ALL_FEATURES(self):
        # 실내기 + 실외기 컬럼을 하나로 합칠 때 사용
        return self.indoor_data + self.outdoor_data

    def DATA_SAVE_PATH(self):
        # 데이터를 저장할 경로 출력
        return self.DATA_PATH

    def BUILDING_NAME(self):
        # 데이터를 저장할 경로 출력
        return self.bldg_name


"""Indoor data"""
start ='2022-01-07'
end = '2022-01-07'
AnalysisObject = 'VirtualSensor' # Optimal/VirtualSensor

"""진리관"""
# Indoor
# cooo = ACQUISITION(start=start, end=end, AnalysisObject=AnalysisObject)
# for i in [909, 910, 921, 920, 919, 917, 918, 911]:
#     cooo.get_indoor_with_Fullsentences(out_unit=i)
# cooo.CLOSE_DATABASE()

# Outdoor
# cooo = ACQUISITION(start=start, end=end, AnalysisObject=AnalysisObject)
# for i in [909, 910, 921, 920, 919, 917, 918, 911]:
#     cooo.get_outdoor_with_Fullsentences(out_unit=i)
# cooo.CLOSE_DATABASE()


"""디지털 도서관"""
# Indoor
cooo = ACQUISITION(start=start, end=end, AnalysisObject=AnalysisObject)
for i in [3065, 3066, 3067, 3069]:
    cooo.get_indoor_with_Fullsentences(out_unit=i)
cooo.CLOSE_DATABASE()

# Outdoor
cooo = ACQUISITION(start=start, end=end, AnalysisObject=AnalysisObject)
for i in [3065, 3066, 3067, 3069]:
    cooo.get_outdoor_with_Fullsentences(out_unit=i)
cooo.CLOSE_DATABASE()
