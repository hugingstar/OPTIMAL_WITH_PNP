import pymysql
import pandas as pd
import datetime
import os

class ACQUISITION():
    def __init__(self, start_year, start_month, start_date, end_year, end_month, end_date):
        """진리관 정보"""
        self.indoor_data = "indoor_power, indoor_set_temp, current_room_temp, eev, relative_capa_code"
        self.outdoor_data = "total_indoor_capa, comp1, comp2, cond_out_temp1, cond_out_temp2, suction_temp1, suction_temp2, " \
                       "discharge_temp1, discharge_temp2, discharge_temp3, outdoor_temperature, high_pressure, low_pressure, eev, ct, meter"
        # 미터 ID NUMBER
        self.meter_id = [1040, 1046, 1551, 1041, 1045, 1552, 1553, 1549, 1042, 1047, 1043, 1550, 1548, 1044]
        # 미터 ID NUMBER
        self.outdoor_id = [908, 907, 909, 910, 921, 920, 919, 917, 918, 911, 915, 913, 914, 916]
        # 미터 ID NUMBER
        self.indoor_id = [922, 924, 925, 926, 928, 929, 930, 931, 933, 934, 935, 936, 937, 938, 939, 940,
                     941, 942, 943, 944, 947, 948, 950, 951, 953, 954, 955, 956, 957, 958, 959, 960,
                     961, 962, 963, 964, 966, 967, 968, 970, 971, 972, 974, 975, 976, 977, 978, 979, 980,
                     981, 983, 984, 985, 986, 988, 990, 991, 992, 993, 994, 996, 997, 998, 999, 1000,
                     1002, 1004, 1005, 1006, 1007, 1008, 1009, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1019, 1020,
                     1021, 1022, 1023, 1024, 1025]

        # 진리관 실외기
        self.jinli_out = [909, 910, 921, 920, 919, 917, 918, 911]

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
        # 진리관 미터기 : 실외기의 Value에 들어있음.
        self.jinli_meter = [1551, 1041, 1045, 1552, 1553, 1549, 1042, 1047]

        # 필요없는 NUMBER
        self.nong = [636, 620, 615, 651, 640, 596, 610, 602, 564, 611, 595, 573]

        """디지털 도서관 정보"""
        #실외기 유닛 번호
        self.dido_out = [3065, 3066, 3067, 3069]
        #각 실외기에 연결된 실내기 번호
        self.dido = {
            3065: [3109, 3100, 3095, 3112, 3133, 3074, 3092, 3105, 3091, 3124, 3071, 3072, 3123, 3125, 3106, 3099, 3081,
                   3131, 3094, 3084],
            3069: [3077, 3082, 3083, 3089, 3096, 3104, 3110, 3117, 3134, 3102, 3116, 3129, 3090],
            3066: [3085, 3086, 3107, 3128, 3108, 3121],
            3067: [3075, 3079, 3080, 3088, 3094, 3101, 3111, 3114, 3115, 3119, 3120, 3122, 3130]}

        """
        1. 데이터 저장할 디렉토리 생성
        예를들어, 오늘 날짜로 폴더가 생성됨 2022-03-07 
        """
        today = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        self.start_year = start_year
        self.start_month = start_month
        self.start_date = start_date
        self.end_year = end_year
        self.end_month = end_month
        self.end_date = end_date

        self.folder_name = "{}-{}-{}".format(self.start_year, self.start_month, self.start_date) #today[0:10]
        print(self.folder_name)
        self.create_folder('D:/OPTIMAL/Data/{}'.format(self.folder_name))

        # MYSQL 접속
        self.db_conn = pymysql.connect(host="192.168.0.33",
                                       user='bsmart',
                                       password='bsmartlab419',
                                       database='biot',
                                       autocommit=True)

        self.cursor = self.db_conn.cursor()

        """(Appendix) SQL 내부에 TABLES 이름 확인"""
        # with self.db_conn:
        #     with self.cursor as cur:
        #         cur.execute('SHOW TABLES')
        #         for data in cur:
        #             print(data)

        """
        2.IndoorUnit data를 Biot에서 가져오기
        이 프로그램은 시작 날짜를 입력하고 현재 날짜까지 데이터를 잘라오는 프로그램임.
        즉, Start 월/일부터 End 월/일까지 입력  
        indoor_start_month : 시작 달
        indoor_start_date : 시작 일
        indoor_month : 끝나는 달
        indoor_date : 끝나는 일
        단, 용량이 있다보니 하나씩 돌려야함.
        저장 위치: 날짜/OutdoorUnit/IndoorUnit
        예를 들어, 2022-03-37/3065/0224
        """
        self.today = datetime.datetime.now()

        """
        3.Outdoor unit
        start_month : 데이터 시작 달
        start_date : 데이터 시작 일
        end_month : 데이터 시작 일
        end_date : 데이터 끝 일
        dev : 디지털 도서관 실외기 정보
        """

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: creating directory. ' + directory)

    def get_indoor(self, unit):
        """
        :param year1: 시작년도
        :param month1: 시작월
        :param date1: 시작일
        :param year: 끝년도
        :param month: 끝월
        :param date: 끝일
        :param unit: '실외기 번호' 입력하면 실내기 맵하고 연결되어 실내기 데이터 나옴
        :return: 실내기 데이터 저장
        """
        if unit in self.jinli_out:
            self.bldginfo = self.jinli
        elif unit in self.dido_out:
            self.bldginfo = self.dido

        for i in self.bldginfo[unit]:
            sql1 = "SELECT * FROM indoor_{} WHERE updated_time >= '{}-{}-{} {}:00:00' and updated_time <= '{}-{}-{} {}:59:00' AND time(updated_time) BETWEEN '00:00:00' AND '23:59:00'".format(
                i, self.start_year, self.start_month, self.start_date, '00', self.end_year, self.end_month, self.end_date, 23)
            # cursor.execute(sql1)
            # rows = cursor.fetchall()
            # print(rows)
            df1 = pd.read_sql(sql1, self.db_conn)
            print(df1)
            save = "D:/OPTIMAL/Data/{}/{}/{}{}".format(self.folder_name, unit, self.end_month, self.end_date)
            self.create_folder(save)
            df1.to_csv(save + '\{}{}_indoor_{}.csv'.format(self.end_month, self.end_date, i))
        # len(df1)
        self.db_conn.close()

    def get_outdoor(self, out_unit, occ=False):
        """
        :param dev: 실외기 디바이스 넘버 List로 넣기
        :param start: 시작일
        :param end: 종료일
        :param occ: 0시부터 23시까지
        :return: 실외기 데이터가 저장
        """
        if occ == True:
            hour1 = '00'
            hour2 = '23'
        elif occ == False:
            hour1 = '00'
            hour2 = '23'

        # if start == end:
        #     period = start
        # elif start != end:
        #     period = start + end

        for i in out_unit:
            # sql = "SELECT * FROM outdoor_{} WHERE updated_time >= '{}-{}-{} {}:00:00' and updated_time <= '{}-{}-{} {}:59:00' AND time(updated_time) BETWEEN '00:00:00' AND '23:59:00'"\
            #     .format(i, start[:4], start[4:6], start[6:8], hour1, end[:4], end[4:6], end[6:8], hour2)
            sql = "SELECT * FROM outdoor_{} WHERE updated_time >= '{}-{}-{} {}:00:00' and updated_time <= '{}-{}-{} {}:59:00' AND time(updated_time) BETWEEN '00:00:00' AND '23:59:00'"\
                .format(i, self.start_year, self.start_month, self.start_date, '00', self.end_year, self.end_month, self.end_date, 23)
            df = pd.read_sql(sql, self.db_conn)
            df.to_csv("D:/OPTIMAL/Data/{}/outdoor_{}.csv".format(self.folder_name, i))
        self.db_conn.close()

    def get_indoor_with_Fullsentences(self, out_unit):
        if out_unit in self.jinli_out:
            bldg_name ="jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido_out:
            bldg_name = "dido"
            self.bldginfo = self.dido

        # Indoor Unit이 순서대로 반복
        for i in self.bldginfo[out_unit]:
            sql1 = "SELECT * FROM indoor_{} WHERE updated_time >= '{}-{}-{} {}:00:00' and updated_time <= '{}-{}-{} {}:59:00' AND time(updated_time) BETWEEN '00:00:00' AND '23:59:00'".format(
                i, self.start_year, self.start_month, self.start_date, '00', self.end_year, self.end_month, self.end_date, 23)
            # cursor.execute(sql1)
            # rows = cursor.fetchall()
            # print(rows)
            df1 = pd.read_sql(sql1, self.db_conn)
            for j in df1.columns:
                df1.rename(columns={'{}'.format(j) : 'Bldg_{}/Outdoor Unit_{}/Indoor Unit_{}/{}'.format(bldg_name, out_unit, i, j)}, inplace=True)
            save = "D:/OPTIMAL/Data/{}/{}".format(self.folder_name, out_unit)
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
        if out_unit in self.jinli_out:
            bldg_name = "jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido_out:
            bldg_name = "dido"
            self.bldginfo = self.dido

        sql = "SELECT * FROM outdoor_{} WHERE updated_time >= '{}-{}-{} {}:00:00' and updated_time <= '{}-{}-{} {}:59:00'"\
            .format(out_unit, self.start_year, self.start_month, self.start_date, '00', self.end_year, self.end_month, self.end_date, '23')
        df = pd.read_sql(sql, self.db_conn)
        for j in df.columns:
            df.rename(columns={'{}'.format(j): 'Bldg_{}/Outdoor Unit_{}/{}'.format(bldg_name, out_unit, j)}, inplace=True)
        df.to_csv("D:/OPTIMAL/Data/{}/Outdoor_{}.csv".format(self.folder_name, out_unit))

    def CLOSE_DATABASE(self):
        return self.db_conn.close()



"""Indoor data"""
start ='2022-01-03'
end = '2022-01-03'
indoor_start_year = start[:4]
indoor_start_month = start[5:7]
indoor_start_date = start[8:10]
indoor_end_year = end[:4]
indoor_end_month = end[5:7]
indoor_end_date = end[8:10]


"""진리관"""
# Indoor
cooo = ACQUISITION(start_year=indoor_start_year, start_month=indoor_start_month, start_date=indoor_start_date,
                   end_year=indoor_end_year, end_month=indoor_end_month, end_date=indoor_end_date)
for i in [909, 910, 921, 920, 919, 917, 918, 911]:
    cooo.get_indoor_with_Fullsentences(out_unit=i)
cooo.CLOSE_DATABASE()

"""진리관 Outdoor"""
cooo = ACQUISITION(start_year=indoor_start_year, start_month=indoor_start_month, start_date=indoor_start_date,
                   end_year=indoor_end_year, end_month=indoor_end_month, end_date=indoor_end_date)
for i in [909, 910, 921, 920, 919, 917, 918, 911]:
    cooo.get_outdoor_with_Fullsentences(out_unit=i)
cooo.CLOSE_DATABASE()




"""디지털 도서관"""
# Indoor
# for i in [3065, 3066, 3067, 3069]:
#     cooo.get_indoor_with_Fullsentences(out_unit=i)

# Outdoor
# dido_out = [3065, 3066, 3067, 3069]
# cooo.get_outdoor(out_unit=dido_out, occ=False)
# cooo.get_outdoor_with_Fullsentences(out_unit=dido_out)
