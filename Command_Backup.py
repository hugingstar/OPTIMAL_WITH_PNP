import time
from Config.load_config import JsonConfig
import json
import datetime
import pandas as pd
import requests
import pause

config_path = '/Config/identification.json'

class biotOptimalControl:
    def __init__(self, config_path, **kwargs):
        self.config = JsonConfig(config_path)
        self.room_num = []
        self.ctrl_room = {}
        self.login_url = self.config.values.login_url
        self.control_url = self.config.values.control_url

        with open("Config/mapping.json", 'r') as f:
            self.ctrl_room = json.load(f)
        # print(ctrl_room)
        self.schedule_db = {}
        indoor_id = ["961", "999", "985", "1019", "1021", "1009", "939", "940", "954", "958", "938", "944",
                     "992", "991", "977", "959", "980", "964", "1000", "1007", "1022", "1011", "998", "981", "1005",
                     "924", "1017",
                     "984", "988", "993", "950", "976", "956", "971", "955", "1002", "1023", "1016", "922", "934",
                     "963", "986", "996", "1012", "1024", "1015", "943", "966", "970", "974", "931", "948", "1014",
                     "930", "968",
                     "978", "960", "953", "935", "925", "1025", "994", "1013", "929", "972", "951", "962", "1004",
                     "936", "975"]
        self.biotlogin()
        print('Login Completed!!!')

    def biotlogin(self):
        with requests.Session() as sess:
            sess.auth = (self.config.values.id, self.config.values.password)
            header = {
                "Accept" : "application/json",
                "Content-Type" : "application/vnd.samsung.biot.v1+json"}
            print(header)
            login = sess.post(self.login_url,
                              headers=header,
                              verify=False)
            Token = login.json()['access_token']

            self.header = {
                "Accept": "application/vnd.samsung.biot.v1+json",
                "Authorization": "Bearer {}".format(Token),
                "Content-Type": "application/vnd.samsung.biot.v1+json"
            }
            print(login.status_code)

    def OPT_Signal_ON(self, opt_target, zone):
        now = datetime.datetime.now().strftime('%y-%m-%d %H:%M')
        # print(type(now), now)
        opt_target = opt_target
        # print(type(opt_target), opt_target)
        self.control_url = "https://168.131.141.115/dms/devices/multiControl/"
        target_zone = [self.ctrl_room['room_{}'.format(zone)]]
        command_message = {"dms_devices_ids": target_zone,
                           "control": {"operations":[{"id" : "AirConditioner.Indoor.General", "power" : "On"}]}}
        """Until Optimal start time"""
        print("Please, wait for On until {}".format(opt_target))
        pause.until(opt_target)
        print("Start optimal start : {}".format(now))
        res_on = requests.put(url=self.control_url,
                             json=command_message,
                             headers=self.header,
                             verify=False)
        print("room: {}, on_status_code: {}".format(target_zone, res_on.status_code))

    def OPT_Signal_OFF(self, opt_target, zone):
        now = datetime.datetime.now().strftime('%y-%m-%d %H:%M')
        # print(type(now), now)
        opt_target = opt_target
        # print(type(opt_target), opt_target)
        self.control_url = "https://168.131.141.115/dms/devices/multiControl/"
        target_zone = [self.ctrl_room['room_{}'.format(zone)]]
        command_message = {"dms_devices_ids": target_zone,
                           "control": {"operations":[{"id" : "AirConditioner.Indoor.General", "power" :"Off"}]}}
        """Until Optimal start time"""
        print("Please,wait for Off until {}".format(opt_target))
        pause.until(opt_target)
        print("Start optimal start")
        res_on = requests.put(url=self.control_url,
                             json=command_message,
                             headers=self.header,
                             verify=False)
        print("room: {}, on_status_code: {}".format(target_zone, res_on.status_code))

"""Login at biot for optimal control"""
biot = biotOptimalControl(config_path=config_path)

"""opt_target must  be later than datetime.now"""
opt_on_target = datetime.datetime(2021, 5, 14, 2, 54)
print("Optimal Start (ON) : {}".format(opt_on_target))
biot.OPT_Signal_ON(opt_target=opt_on_target, zone='961')

opt_off_target = datetime.datetime(2021, 5, 14, 2, 55)
print("Optimal Stop (OFF) : {}".format(opt_off_target))
biot.OPT_Signal_OFF(opt_target=opt_off_target, zone='961')