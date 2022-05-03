# -*- coding: utf-8 -*-
import pandas as pd
import requests
import json
import time
import datetime
import logging
# from apscheduler.schedulers.background import BackgroundScheduler
# import psycopg2 as pg2
from datetime import datetime

"""이 코드는 참고용임"""

class biotControl:
    def __init__(self, **kwargs):
        self.room_num = []
        self.ctrl_room = {}
        self.login_url = 'https://168.131.141.115/api/login'
        self.indoor_id = []
        self.indoor_data = "device_id, indoor_set_temp, current_room_temp, indoor_power, relative_capa_code"
        self.df = pd.DataFrame()

        self.ctrl_room = {
                "room_961": "00:FF:0C:02:10:01:SFAF9114-990W-39", #201-1
                "room_999": "00:FF:0C:02:11:01:SFAF9114-990W-39", #201-2
                "room_985": "00:FF:0C:02:12:01:SFAF9114-990W-39", #202-1
                "room_1019": "00:FF:0C:02:13:01:SFAF9114-990W-39", #202-2
                "room_1021": "00:FF:0C:02:14:01:SFAF9114-990W-39", #홀
                "room_1009": "00:FF:0C:02:15:01:SFAF9114-990W-39", #휴게공간1
                "room_939": "00:FF:0C:02:16:01:SFAF9114-990W-39",  #휴게공간2
                "room_940": "00:FF:0C:03:17:01:SFAF9114-990W-39", #203
                "room_954": "00:FF:0C:03:18:01:SFAF9114-990W-39", #204-1
                "room_958": "00:FF:0C:03:19:01:SFAF9114-990W-39", #204-2
                "room_938": "00:FF:0C:03:1B:01:SFAF9114-990W-39", #205-1
                "room_944": "00:FF:0C:03:1A:01:SFAF9114-990W-39", #205-2

                "room_992": "00:FF:0C:04:1C:01:SFAF9114-990W-39", #301-1
                "room_991": "00:FF:0C:04:1D:01:SFAF9114-990W-39", #301-2
                "room_977": "00:FF:0C:04:1E:01:SFAF9114-990W-39", #302-1
                "room_959": "00:FF:0C:04:1F:01:SFAF9114-990W-39", #302-2
                "room_980": "00:FF:0C:04:20:01:SFAF9114-990W-39", #303-1
                "room_964": "00:FF:0C:04:21:01:SFAF9114-990W-39", #303-2
                "room_1000": "00:FF:0C:04:22:01:SFAF9114-990W-39", #3층 휴게실
                "room_1007": "00:FF:0C:04:23:01:SFAF9114-990W-39", #3층 휴게실
                "room_1022": "00:FF:0C:05:26:01:SFAF9114-990W-39", #304
                "room_1011": "00:FF:0C:05:27:01:SFAF9114-990W-39", #305-1
                "room_998": "00:FF:0C:05:28:01:SFAF9114-990W-39", #305-2
                "room_981": "00:FF:0C:05:29:01:SFAF9114-990W-39", #306-1
                "room_1005": "00:FF:0C:05:2A:01:SFAF9114-990W-39", #306-2
                "room_924": "00:FF:0C:05:24:01:SFAF9114-990W-39", #홀3
                "room_1017": "00:FF:0C:05:25:01:SFAF9114-990W-39", #홀3

                "room_984": "00:FF:0C:06:2B:01:SFAF9114-990W-39", #401-1
                "room_988": "00:FF:0C:06:2C:01:SFAF9114-990W-39", #401-2
                "room_993": "00:FF:0C:06:2D:01:SFAF9114-990W-39", #402-1
                "room_950": "00:FF:0C:06:2E:01:SFAF9114-990W-39", #402-2
                "room_976": "00:FF:0C:06:2F:01:SFAF9114-990W-39", #403-1
                "room_956": "00:FF:0C:06:30:01:SFAF9114-990W-39", #403-2
                "room_971": "00:FF:0C:07:33:01:SFAF9114-990W-39", #404
                "room_955": "00:FF:0C:07:34:01:SFAF9114-990W-39", #405-1
                "room_1002": "00:FF:0C:07:35:01:SFAF9114-990W-39", #405-2
                "room_1023": "00:FF:0C:07:37:01:SFAF9114-990W-39", #406-1
                "room_1016": "00:FF:0C:07:36:01:SFAF9114-990W-39", #406-2
                "room_922": "00:FF:0C:07:32:01:SFAF9114-990W-39", #홀4
                "room_934": "00:FF:0C:07:31:01:SFAF9114-990W-39", #홀4

                "room_963": "00:FF:0C:09:00:01:SFAF9114-990W-39", #501-1
                "room_986": "00:FF:0C:09:01:01:SFAF9114-990W-39", #501-2
                "room_996": "00:FF:0C:09:02:01:SFAF9114-990W-39", #502-1
                "room_1012": "00:FF:0C:09:03:01:SFAF9114-990W-39", #502-2
                "room_1024": "00:FF:0C:09:04:01:SFAF9114-990W-39", #503-1
                "room_1015": "00:FF:0C:09:05:01:SFAF9114-990W-39", #503-2
                "room_943": "00:FF:0C:09:06:01:SFAF9114-990W-39", #5층 휴게실
                "room_966": "00:FF:0C:09:08:01:SFAF9114-990W-39", #5층 휴게실
                "room_970": "00:FF:0C:08:0A:01:SFAF9114-990W-39", #504
                "room_974": "00:FF:0C:08:0B:01:SFAF9114-990W-39", #505-1
                "room_931": "00:FF:0C:08:0C:01:SFAF9114-990W-39", #505-2
                "room_948": "00:FF:0C:08:0D:01:SFAF9114-990W-39", #506-1
                "room_1014": "00:FF:0C:08:0E:01:SFAF9114-990W-39", #506-2
                "room_930": "00:FF:0C:08:09:01:SFAF9114-990W-39", #홀5
                "room_968": "00:FF:0C:08:07:01:SFAF9114-990W-39", #홀5

                "room_978": "00:FF:0C:0B:00:01:SFAF9114-990W-39", #601-1
                "room_960": "00:FF:0C:0B:01:01:SFAF9114-990W-39", #601-2
                "room_953": "00:FF:0C:0B:02:01:SFAF9114-990W-39", #602-1
                "room_935": "00:FF:0C:0B:03:01:SFAF9114-990W-39", #602-2
                "room_925": "00:FF:0C:0B:04:01:SFAF9114-990W-39", #603-1
                "room_1025": "00:FF:0C:0B:05:01:SFAF9114-990W-39", #603-2
                "room_994": "00:FF:0C:0B:08:01:SFAF9114-990W-39", #6층 휴게실(607)
                "room_1013": "00:FF:0C:0B:06:01:SFAF9114-990W-39", #6층 휴게실(607)
                "room_929": "00:FF:0C:0A:0A:01:SFAF9114-990W-39", #604
                "room_972": "00:FF:0C:0A:0B:01:SFAF9114-990W-39", #605-1
                "room_951": "00:FF:0C:0A:0C:01:SFAF9114-990W-39", #605-2
                "room_962": "00:FF:0C:0A:0D:01:SFAF9114-990W-39", #606-1
                "room_1004": "00:FF:0C:0A:0E:01:SFAF9114-990W-39", #606-2
                "room_936": "00:FF:0C:0A:07:01:SFAF9114-990W-39", #홀6
                "room_975": "00:FF:0C:0A:09:01:SFAF9114-990W-39" #홀6
            }
        self.schedule_db = {}
        indoor_id = ["961", "999", "985", "1019", "1021", "1009", "939", "940", "954", "958", "938", "944",
                  "992", "991", "977", "959", "980", "964", "1000", "1007", "1022", "1011", "998", "981", "1005", "924", "1017",
                  "984", "988", "993", "950", "976", "956", "971", "955", "1002", "1023", "1016", "922", "934",
                  "963", "986", "996", "1012", "1024", "1015", "943", "966", "970", "974", "931", "948", "1014", "930", "968",
                  "978", "960", "953", "935", "925", "1025", "994", "1013", "929", "972", "951", "962", "1004", "936", "975"]

        # indoor_id = [
        #              "984", "988", "993", "950", "976", "956", "971", "955", "1002", "1023", "1016", "922", "934",
        #              "963", "986", "996", "1012", "1024", "1015", "943", "966", "970", "974", "931", "948", "1014", "930", "968",
        #              "929", "972", "951", "962", "1004", "936", "975"]
        
        # # 2, 3, 4, 5, 6층
        # indoor_id = ["961", "999", "985", "1019", "1021", "1009", "939", "940", "954", "958", "938", "944",
        #           "992", "991", "977", "959", "980", "964", "1000", "1007", "1022", "1011", "998", "981", "1005", "924", "1017",
        #           "984", "988", "993", "950", "976", "956", "971", "955", "1002", "1023", "1016", "922", "934",
        #           "963", "986", "996", "1012", "1024", "1015", "943", "966", "970", "974", "931", "948", "1014", "930", "968",
        #           "929", "972", "951", "962", "1004", "936", "975"]
        #

        for i in indoor_id:
            room_name = 'room_' + str(i)
            a = pd.read_csv("D:/실증결과/2019/Dec/database/{}.csv".format('indoor_'+str(i)))
            # a = pd.read_csv("D:/실증결과/20210120/indoor_{}.csv".format(i))
            for k in range(len(a)):
                a['updated_time'][k] = a['updated_time'][k].split('.')[0].split(':')[0]+":"+a['updated_time'][k].split('.')[0].split(':')[1]
            self.schedule_db[room_name] = a

        self.login()
        print('Control Start')

    def starting_base(self):
        now = datetime.now().strftime('2019-12-10 %H:%M')
        print(now)
        control_room = {}
        if int(now.split(' ')[1].split(':')[0]) >= 8 and int(now.split(' ')[1].split(':')[0]) <= 17:
            for room_num in self.schedule_db.keys():
                control_value = self.schedule_db[room_num].loc[self.schedule_db[room_num]['updated_time'] == now]['indoor_power'].tolist()
                try:
                    if control_value[0] is True:
                        control_room[room_num] = 1
                    elif control_value[0] is False:
                        control_room[room_num] = 0
                    else:
                        control_room[room_num] = 2
                except:
                        control_room[room_num] = 2

            onlist = []
            offlist = []
            for ctrl in control_room.keys():
                id = self.ctrl_room[ctrl]
                if control_room[ctrl] == 1:
                    onlist.append(id)
                elif control_room[ctrl] == 0:
                    offlist.append(id)
                else:
                    pass

            control_url = "https://168.131.141.115/dms/devices/multiControl/"

            if len(onlist) > 0:
                on_payload = {"dms_devices_ids": onlist,
                               'control': {"operations": [{"id": "AirConditioner.Indoor.General", "power": "On"}]}}
                res_on = requests.put(url=control_url, json=on_payload, headers=self.header, verify=False)
                print(on_payload)
                print("room: {}, on_status_code: {}".format(onlist, res_on.status_code))

            if len(offlist) > 0:
                off_payload = {"dms_devices_ids": offlist,
                               "control": {"operations": [{"id": "AirConditioner.Indoor.General", "power": "Off"}]}}
                res_off = requests.put(url=control_url, json=off_payload, headers=self.header, verify=False)
                print(off_payload)
                print("room: {}, off_status_code: {}".format(offlist, res_off.status_code))

            else:
                print("nothing")

        else:
            pass

    def login(self):
        sess = requests.Session()
        sess.auth = ("blab419", "419lab@@")
        header = {
            'Accept': "application/json",
            'Content-Type': "application/vnd.samsung.biot.v1+json"}
        login = sess.post(self.login_url, headers=header, verify=False)
        Token = login.json()['access_token']

        self.header = {
            'Accept': "application/vnd.samsung.biot.v1+json",
            'Authorization': "Bearer {}".format(Token),
            'Content-Type': "application/vnd.samsung.biot.v1+json"
        }
        print(login.status_code)


biot = biotControl()


# sched = BackgroundScheduler()
# sched.start()
# sched.add_job(biot.starting_base, 'cron', hour='*', minute='*', second='00')
#
# count = 0
#
# while True:
#     print(datetime.now(), ":Running")
#     time.sleep(20)