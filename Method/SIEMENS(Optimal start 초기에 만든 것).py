import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class OPTIMALSTART():
    def __init__(self,time, path_ind, path_outd, features_in, features_out, mode, signal, scaler):
        """
        self.w_adj : 보정 Weight 사용자 설정 가능
        self.del_t : 전 날의 작동 시간
        self.T1 : 전 날의 초기 실내 온도
        self.T2 : 전 날의 Occpancy 때 실내 온도
        :param mode:
        """
        self.onoffsignal = signal # 온 오프 시그널 컬럼명 ON : 1/OFF : 0
        self.mode = mode #Cooling(c)/Heating(h)

        """Indoor data"""
        self.data_org = pd.read_csv(path_ind, index_col=time)
        self.data_org = self.data_org[features_in]

        """Outdoor data"""
        self.data_outd = pd.read_csv(path_outd, index_col=time)
        self.data_outd = self.data_outd[features_out]

        """Preprocessing"""
        self.data = pd.concat([self.data_outd, self.data_org], axis=1)
        self.data = self.data[self.data[self.onoffsignal] == 1] # 작동중인 데이터만 뽑기
        self.del_t = len(self.data) # 작동 시간
        self.data['period'] = int(self.del_t)

        print(self.del_t)
        if self.del_t != 0:
            """모델 변수"""
            self.T1 = self.data['current_room_temp'].tolist()[0] # 10
            self.T2 = self.data['current_room_temp'].tolist()[-1] # 24 Occupancy 데이터
            self.Tsp = self.data['indoor_set_temp'].tolist()[0]  # 설정 온도
            self.Toa = self.data_outd['outdoor_temperature'].tolist()[0]

            """작동 시작 시간, 작동 끝 시간 확인"""
            self.operStart = self.data.index[0]
            self.operEnd = self.data.index[0]

            print("=========== Data summary ===========\n"
                  "Initial Zone temp(T1) : {} - Occupancy Zone temp : {}\n"
                  "Set temp : {} - Outdoor temp : {}\n"
                  "Mode : {} - Operation periods : {}\n"
                  "operStart : {} - operEnd : {}\n"
                  "===================================="
                  .format(self.T1, self.T2, self.Tsp, self.Toa, self.mode, self.del_t, self.operStart, self.operEnd))

            #One-hot-encoding : 모델에 사용할 때 상태를 인식할 떄는 원핫인코딩 활용
            self.data = pd.get_dummies(self.data)

            # Weight 값을 일단 그냥 설정 단 너무 많이 작동해버리면 최적화가 필요
            self.w_adj = 5

            if self.mode in ['C', 'c', 'COOLING', 'Cooling', 'cooling', 'COOL', 'Cool', 'cool']:
                a = int(self.cooling_opt()) #a 자체가 음수로나옴
                b = int(self.tuned_adj(opt_time=a)) #즉, 냉방일때 온도 도달 안대면 b는 양수로 나옴/도달 되면 음수로나옴
                # b = self.time_adjust()
                self.optimal_period = a - b
                logging.error("Optimal_period : {} - Cooling optimal period : {} - Adjust period : {}".format(self.optimal_period, a, -b))

            elif self.mode in ['H', 'h', 'HEATING', 'Heating','heating', 'HEAT', 'Heat', 'heat']:
                a = int(self.heating_opt()) #a 자체가 음수로나옴
                b = int(self.tuned_adj(opt_time=a)) #즉, 난방일 때 온도 도달 안대면 b는 음수로 나옴 도달 하면
                # b = self.time_adjust()
                self.optimal_period = a + b
                logging.error("Optimal_period : {} - Heating optimal period : {} - Adjust period : {}".format(self.optimal_period, a, b))
            else:
                pass
                logging.error("Please, select mode cooling or heating ")
        else:
            self.optimal_period = 40

    def cooling_opt(self):
        logging.error("Cooling model start")
        opt_period = int(self.C1() * (self.T1 - self.Tsp) + self.C2() * (self.T1 - self.Tsp) * (self.Toa - self.Tsp) * (1/25))
        return opt_period

    def heating_opt(self):
        logging.error("Heating model start")
        opt_period = int(self.C1() * (self.T1 - self.Tsp) + self.C2() * (self.T1 - self.Tsp) * (self.Toa - self.Tsp) * (1/25))
        return opt_period

    def time_adjust(self):
        if self.w_adj == None:
            logging.error("Adjustment weight is missed, Setting 5 min")
            self.w_adj = 5
        else:
            pass
        return int(self.w_adj * (self.T2 - self.Tsp))

    def tuned_adj(self, opt_time):
        """
        :param opt_time: Optimals start 계산한 본체의 값
        :return:
        """
        x1 = self.data['current_room_temp'].tolist()[-1]
        x2 = self.data['indoor_set_temp'].tolist()[-1]
        y = self.data['period'].tolist()[-1]

        """튜닝할 변수선언"""
        torch.manual_seed(1)
        var1 = torch.Tensor([x1]).unsqueeze(1) # 방온도
        var2 = torch.Tensor([x2]).unsqueeze(1) # 설정 온도
        var3 = torch.Tensor([opt_time]).unsqueeze(1)
        tar = torch.Tensor([y]).unsqueeze(1)
        w_adj = torch.zeros(1, requires_grad=True)

        optimizer = optim.SGD([w_adj], lr=0.01)

        nb_epochs = 100
        for epoch in range(nb_epochs + 1):

            # Hypothesis
            hypothesis = w_adj * abs(var1 - var2) + var3

            # Cost Function
            cost = torch.mean((hypothesis - tar) ** 2)

            # Cost 함수를 기반으로 H(x)를 최적화한다.
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Epoch {:4d}/{} - w_adj: {:.3f} - Cost: {:.6f}'.format(epoch, nb_epochs, w_adj.item(),cost.item()))
        return w_adj.item()


    def C1(self):
        ans = round((self.del_t)/(self.T2 - self.T1), 2)
        logging.error("C1 : {}".format(ans))
        return ans

    def C2(self):
        ans = round((self.del_t)/((self.T2-self.T1) * (self.T2-self.T1)), 2)
        logging.error("C2 : {}".format(ans))
        return ans

    def __repr__(self):
        """OPTIMAL class에서 계산된 결과를 출력"""
        # self.w_adj, self.del_t, self.T1, self.T2
        return self.optimal_period

class OPTIMALSTOP():
    def __init__(self, mode, w_adj, del_t, T1, T2, Tsp, Toa):
        self.w_adj = w_adj # min
        self.T1 = T1
        self.T2 = T2
        self.del_t = del_t
        self.Tsp = Tsp
        self.Toa = Toa

        if mode in ['C', 'c', 'COOLING', 'Cooling', 'cooling', 'COOL', 'Cool', 'cool']:
            a = self.cooling_opt()
            b = self.time_adjust()
            self.optimal_period = a - b
            logging.error("Optimal_period : {} - Cooling optimal period : {} - Adjust period : {}".format(self.optimal_period, a, -b))

        elif mode in ['H', 'h', 'HEATING', 'Heating','heating', 'HEAT', 'Heat', 'heat']:
            a = self.heating_opt()
            b = self.time_adjust()
            self.optimal_period = a - b
            logging.error("Optimal_period : {} - Heating optimal period : {} - Adjust period : {}".format(self.optimal_period, a, b))
        else:
            logging.error("Please, select mode cooling or heating ")

    def cooling_opt(self):
        logging.error("Cooling model stop")
        opt_period = int(10 * self.C() * (self.T1 - self.Tsp) / (self.Toa - self.Tsp))
        return opt_period

    def heating_opt(self):
        logging.error("Heating model stop")
        opt_period = int(25 * self.D() * (self.T1 - self.Tsp) / (self.Toa - self.Tsp))
        return opt_period

    def time_adjust(self):
        if self.w_adj == None:
            logging.error("Adjustment weight is missed, Setting 5 min")
            self.w_adj = 5
        else:
            pass
        return int(self.w_adj * (self.T2 - self.Tsp))

    def C(self):
        ans = round((self.del_t)/((self.T2-self.T1) * (self.T2-self.T1)), 2)
        logging.error("C : {}".format(ans))
        return ans

    def D(self):
        ans = round((self.del_t)/((self.T2-self.T1) * (self.T2-self.T1)), 2)
        logging.error("D : {}".format(ans))
        return ans

    def __repr__(self):
        """OPTIMAL class에서 계산된 결과를 출력"""
        return self.optimal_period



"""
Optimal start test : 테스트용
"""
# path_ind = 'D:/OPTIMAL/Data/2022-03-10/909/0103/0103_indoor_961.csv'
# path_outd = 'D:/OPTIMAL/Data/2022-03-10/outdoor_909.csv'
# time = 'updated_time'
# mode = 'h'
# signal = 'indoor_power'
# features_in = ['indoor_power', 'evain_temp',
#        'evaout_temp', 'indoor_set_temp', 'current_room_temp', 'eev',
#        'indoor_fan_speed', 'relative_capa_code']
#
# features_out = ['total_indoor_capa', 'comp1','comp2', 'suction_temp1',
#             'discharge_temp1', 'discharge_temp2', 'discharge_temp3',
#             'outdoor_temperature', 'high_pressure', 'low_pressure', 'eev1',
#             'value','double_tube_temp', 'evi_eev','fan_step']
#
# # Scaler option : StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler
# scaler = 'MinMaxScaler'


# optimal_period = OPTIMALSTART(path=path_ind,
#                               path2=path_outd,
#                               time=time,
#                               features_in=features_in,
#                               features_out=features_out,
#                               mode=mode,
#                               signal=signal,
#                               scaler=scaler).__repr__()
