import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error


class ENSEMBLE():
    def __init__(self, time, TRAIN_SIZE, GRAD_CLIP, BATCH_SIZE, LEARNING_RATE, N_JOBS, RANDOM_STATE,
                 space, start, end):

        self.DATA_PATH = "D:/OPTIMAL/Data"
        self.SAVE_PATH = "D:/OPTIMAL/Results"
        self.TIME = time

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

        """Hyperparameters"""
        self.TRAIN_SIZE = TRAIN_SIZE
        self.GRAD_CLIP = GRAD_CLIP
        self.BATCH_SIZE = BATCH_SIZE
        self.N_ESTIMATION = space['N_ESTIMATION']
        self.LEARNING_RATE = LEARNING_RATE
        self.MAX_DEPTH = space['MAX_DEPTH']
        self.MAX_FEATURES = space['MAX_FEATURES']
        self.MAX_LEAF_NODES = space['MAX_LEAF_NODES']
        self.N_JOBS = N_JOBS
        self.RANDOM_STATE = RANDOM_STATE
        self.space = space

        """날짜 정보"""
        self.start_year = start[:4]
        self.start_month = start[5:7]
        self.start_date = start[8:10]
        self.end_year = end[:4]
        self.end_month = end[5:7]
        self.end_date = end[8:10]

        self.folder_name = "{}-{}-{}".format(self.start_year, self.start_month, self.start_date)
        self.create_folder('{}/Ensemble'.format(self.SAVE_PATH))

    def CLASSIFIER(self, out_unit, signal, meterValue, TspValue, TzValue, ToaValue, target, method):
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        """Numbers"""
        self.out_unit = out_unit

        """classifier method """
        self.method = method

        """실외기 데이터"""
        self._outdpath = "{}/{}/Outdoor_{}.csv" \
            .format(self.DATA_PATH, self.folder_name, out_unit)
        self._outdata = pd.read_csv(self._outdpath, index_col=self.TIME)

        """실내기 데이터"""
        for self.i in list(self.bldginfo[out_unit]):
            self._indpath = "{}/{}/{}/Outdoor_{}_Indoor_{}.csv"\
                .format(self.DATA_PATH, self.folder_name, out_unit, out_unit, self.i)
            self._indata = pd.read_csv(self._indpath, index_col=self.TIME)
            # print(self._indata)

            """실내기와 실외기 데이터 합친거"""
            self.data = pd.concat([self._outdata, self._indata], axis=1)

            """문자열로 되어 있는 정보는 숫자로 대체"""
            self.data = self.data.replace({"High": 3, "Mid" : 2, "Low" : 1, "Auto" : 4})

            # 관련 컬럼 불러 내기
            self.onoffsignal = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=signal, case=False)])[0]
            self.meter_value = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=meterValue, case=False)])[0]
            self.set_temp = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=TspValue, case=False)])[0]
            # self.zone_temp = list(pd.Series(list(self.data.columns))[
            #                             pd.Series(list(self.data.columns)).str.contains(pat=TzValue, case=False)])[0]
            self.outdoor_temp =list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=ToaValue, case=False)])[0]

            # 작동시간 값을 입력
            num = 0
            for o in range(self.data.shape[0] - 1):
                a_o = int(self.data[self.onoffsignal][o]) # 전원 현재 값
                b_o = int(self.data[self.onoffsignal][o + 1]) # 전원 다음 값
                c_o = round(self.data[self.meter_value][o + 1] - self.data[self.meter_value][o], 3) # 미터 값의 차이
                # d_o = round(self.data[self.zone_temp][o + 1] - self.data[self.set_temp][o], 3) # 설정온도_구역온도 차이
                # e_o = round(self.data[self.zone_temp][o + 1] - self.data[self.outdoor_temp][o], 3) # 외기온도_구역온도 차이
                # f_o = round(self.data[self.zone_temp][o + 1] - self.data[self.zone_temp][o], 3)  # 구역온도2_구역온도1 차이
                g_o = round(self.data[self.set_temp][o + 1] - self.data[self.outdoor_temp][o], 3) # 외기온도_구역온도 차이

                if  (a_o == 0) and (b_o != 0):
                    num += 1
                    self.data.at[self.data.index[o], "{}_duration".format(self.onoffsignal)] = num
                    self.data.at[self.data.index[o], "{}_difference".format(self.meter_value)] = c_o
                    # self.data.at[self.data.index[o], "{}_and_set_difference".format(self.zone_temp)] = d_o
                    # self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.zone_temp)] = e_o
                    # self.data.at[self.data.index[o], "{}_and_zone_difference".format(self.zone_temp)] = f_o
                    self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.set_temp)] = g_o
                elif (a_o != 0) and (b_o != 0):
                    num += 1
                    self.data.at[self.data.index[o], "{}_duration".format(self.onoffsignal)] = num
                    self.data.at[self.data.index[o], "{}_difference".format(self.meter_value)] = c_o
                    # self.data.at[self.data.index[o], "{}_and_set_difference".format(self.zone_temp)] = d_o
                    # self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.zone_temp)] = e_o
                    # self.data.at[self.data.index[o], "{}_and_zone_difference".format(self.zone_temp)] = f_o
                    self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.set_temp)] = g_o
                elif (a_o != 0) and (b_o == 0):
                    num = 0
                    self.data.at[self.data.index[o], "{}_duration".format(self.onoffsignal)] = num
                    self.data.at[self.data.index[o], "{}_difference".format(self.meter_value)] = c_o
                    # self.data.at[self.data.index[o], "{}_and_set_difference".format(self.zone_temp)] = d_o
                    # self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.zone_temp)] = e_o
                    # self.data.at[self.data.index[o], "{}_and_zone_difference".format(self.zone_temp)] = f_o
                    self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.set_temp)] = g_o
                else:
                    num = 0
                    self.data.at[self.data.index[o], "{}_duration".format(self.onoffsignal)] = num
                    self.data.at[self.data.index[o], "{}_difference".format(self.meter_value)] = c_o
                    # self.data.at[self.data.index[o], "{}_and_set_difference".format(self.zone_temp)] = d_o
                    # self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.zone_temp)] = e_o
                    # self.data.at[self.data.index[o], "{}_and_zone_difference".format(self.zone_temp)] = f_o
                    self.data.at[self.data.index[o], "{}_and_oa_difference".format(self.set_temp)] = g_o
            # 가장 마지막 값은 이전 값을 받음
            self.data.at[self.data.index[-1], "{}_duration".format(self.onoffsignal)] = num
            self.data.at[self.data.index[-1], "{}_difference".format(self.meter_value)] = c_o
            # self.data.at[self.data.index[-1], "{}_and_set_difference".format(self.zone_temp)] = d_o
            # self.data.at[self.data.index[-1], "{}_and_oa_difference".format(self.zone_temp)] = e_o
            # self.data.at[self.data.index[-1], "{}_and_zone_difference".format(self.zone_temp)] = f_o
            self.data.at[self.data.index[-1], "{}_and_oa_difference".format(self.set_temp)] = g_o

            self.save = "{}/Ensemble/{}/{}/{}".format(self.SAVE_PATH, self.method, self.folder_name, out_unit)
            self.create_folder(self.save)
            self.data.to_csv("{}/Before_Outdoor_{}_Indoor_{}.csv".format(self.save, out_unit, self.i)) # 조건 적용 전

            """
            조건을 적용
            """
            # self.data = self.data[self.data[self.onoffsignal] == 1] #작동중인 데이터만 사용
            self.data = self.data.dropna(axis=0)
            self.data.to_csv("{}/After_Outdoor_{}_Indoor_{}.csv".format(self.save, out_unit, self.i)) # 조건 적용 후
            print(out_unit, self.i, self.data.shape)

            #해당 문자열이 포함되면 그것이 타겟이 된다.
            self.target = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=target, case=False)])[0]
            print("[Target column name] {}".format(self.target))

            # 타겟을 제외한 나머지는 독립변수
            self.features = list(self.data.columns.difference([self.target]))

            X = self.data[self.data.columns.difference([self.target])]
            y = self.data[self.target] # 타겟컬럼

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                    train_size=float(self.TRAIN_SIZE),
                                                                                    random_state=self.RANDOM_STATE)
            self.save_rdf = "{}/Ensemble/{}/{}/{}".format(self.SAVE_PATH, self.method, self.folder_name, self.out_unit)
            self.create_folder(self.save_rdf)

            self.X_train.to_csv("{}/X_train_Outdoor_{}_Indoor_{}.csv".format(self.save_rdf, self.out_unit, self.i))
            self.X_test.to_csv("{}/X_test_Outdoor_{}_Indoor_{}.csv".format(self.save_rdf, self.out_unit, self.i))
            self.y_train.to_csv("{}/y_train_Outdoor_{}_Indoor_{}.csv".format(self.save_rdf, self.out_unit, self.i))
            self.y_test.to_csv("{}/y_test_Outdoor_{}_Indoor_{}.csv".format(self.save_rdf, self.out_unit, self.i))

            self.optimize(FN=self.MTHD, MAX_EVALS=50)
            self.IMP_LIST()


    def MTHD(self):
            if self.method  == "Randomforest":
                self.forest = RandomForestClassifier(criterion='entropy', max_depth=self.MAX_DEPTH,
                                                n_estimators=self.N_ESTIMATION, n_jobs=self.N_JOBS,
                                                random_state=self.RANDOM_STATE)
            elif self.method == "Adaboosting":
                self.decision = DecisionTreeClassifier(max_depth=self.MAX_DEPTH, max_features=self.MAX_FEATURES,
                                                  class_weight='balanced', random_state=self.RANDOM_STATE)
                self.forest = AdaBoostClassifier(base_estimator=self.decision, n_estimators=self.N_ESTIMATION,
                                            learning_rate=self.LEARNING_RATE, random_state=self.RANDOM_STATE)
            elif self.method == "Gradientboosting":
                self.forest = GradientBoostingClassifier(learning_rate=self.LEARNING_RATE, n_estimators=self.N_ESTIMATION,
                                                    max_depth=self.MAX_DEPTH, max_features=self.MAX_FEATURES,
                                                    random_state=self.RANDOM_STATE)
            elif self.method == "DecisionTree":
                self.forest = DecisionTreeClassifier(max_depth=self.MAX_DEPTH, max_features=self.MAX_FEATURES,
                                                  class_weight='balanced', random_state=self.RANDOM_STATE)
            try:
                # evaluation = [(self.X_train, self.y_train),(self.X_test, self.y_test)]
                self.forest.fit(self.X_train, self.y_train,) # eval_set=evaluation, eval_metric='rmse',
                           # early_stopping_rounds=20, verbose=0)
                self.pred = self.forest.predict(self.X_test)
                self.rmse = self.RMSE(self.y_test, self.pred)
                print("[Train accuracy] : {} - [Test accuracy] : {}".format(round(self.forest.score(self.X_train, self.y_train), 3),
                                                                        round(self.forest.score(self.X_test, self.y_test), 3)))
                acc = pd.DataFrame(columns=['Train_accuracy', 'Test_accuracy'])
                self.trA_ = round(float(self.forest.score(self.X_train, self.y_train)) * 100, 1)
                self.tesA_ = round(float(self.forest.score(self.X_test, self.y_test)) * 100, 1)
                acc['Train_accuracy'] = [self.trA_]
                acc['Test_accuracy'] = [self.tesA_]
                acc.to_csv("{}/Acc_Outdoor_{}_Indoor_{}.csv".format(self.save_rdf, self.out_unit, self.i))
                self.imp_list = np.array(self.forest.feature_importances_)
                self.imp_list = np.multiply(self.imp_list, 100)
                print("[Feature Importance Outdoor_{}_Indoor_{}] {}".format(self.out_unit, self.i, self.imp_list))

                IMP = pd.DataFrame([self.imp_list], columns=self.data.columns.difference([self.target]))
                IMP.to_csv("{}/IMP_Outdoor_{}_Indoor_{}.csv".format(self.save_rdf, self.out_unit, self.i))

            except ValueError as ve:
                print("[Value Error] {}".format(ve))
                # 에러났을 경우에는 초기화해준다.
                acc = pd.DataFrame(columns=['Train_accuracy','Test_accuracy'])
                trA_ = float(0)
                tesA_ = float(0)
                acc['Train_accuracy'] = [trA_]
                acc['Test_accuracy'] = [tesA_]
                acc.to_csv("{}/Acc_Outdoor_{}_Indoor_{}.csv".format(self.save_rdf, self.out_unit, self.i))
                self.imp_list = [0] * int(len(self.features)) #np.multiply(self.imp_list, 0)
                self.rmse = 0

            return {'loss' : self.rmse, 'status': STATUS_OK, 'model': self.forest}

    def IMP_LIST(self):
            dxf = pd.Series(self.imp_list, index=self.features).sort_values(ascending=False)
            plt.rcParams["font.family"] = "Times New Roman"

            ftsize = 40
            plt.figure(figsize=(100, 50), dpi=300)
            plt.grid()
            plt.barh(dxf.index, dxf.values, align='center', height=0.7, edgecolor='k')
            for k, v in enumerate(dxf.index):
                plt.text(dxf.values[k], v, "{} %".format(round(dxf.values[k], 1)), fontsize=ftsize*2.5,
                         horizontalalignment='left', verticalalignment='center')
            plt.xlabel("Features Importances", fontsize=float(ftsize*3.5))
            plt.xticks([0, 25, 50, 75, 100], size=float(ftsize*3))
            plt.ylabel("Features", fontsize=float(ftsize*3.5))
            plt.yticks(size=float(ftsize * 2))
            plt.title("Prediction Target :\n{}\n(Method : {} / Train : {} % / Test : {} %)".format(self.target, self.method, self.trA_, self.tesA_), fontsize=float(ftsize*3.5))
            plt.tight_layout()
            plt.savefig("{}/FIGIMP_Outdoor_{}_Indoor_{}_Batch_{}.png".format(self.save_rdf, self.out_unit, self.i, self.BATCH_SIZE))
            plt.clf()
            return

    def RMSE(self, y_test, pred ):
        return np.sqrt(mean_squared_error(y_test, pred))

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: creating directory. ' + directory)

    def optimize(self, FN, MAX_EVALS):
        self.trials = Trials()
        best = fmin(fn=FN, space=space, algo=tpe.suggest, max_evals=MAX_EVALS,
                    trials=self.trials, rstate=np.random.default_rng(0))
        print(best)
        return best


start = '2021-01-01'
end = '2021-03-31'

"""Hyperparameters"""
time ="updated_time" #시계열 인덱스
TARGET = "room_temp" #타켓 컬럼"Duration" #"relative_capa_code"
# N_ESTIMATION = 1000
# MAX_DEPTH = 9
# MAX_FEATURES = 5
# MAX_LEAF_NODES = 10
TRAIN_SIZE = 0.7
GRAD_CLIP = 2.5
BATCH_SIZE = 500
LEARNING_RATE = 0.01
RANDOM_STATE = 0
N_JOBS = -1

# reg_candidate = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 100]

#수정위치
space = {
    "MAX_DEPTH" : hp.uniform("max_depth", 5, 9),
    "MAX_LEAF_NODES" : hp.uniform("max_leaf_nodes", 5, 10),
    "MAX_FEATURES" : hp.uniform("max_features", 3, 10),
    "N_ESTIMATION": hp.uniform("n_estimators", 10, 1000),
}


SIGNAL = 'indoor_power' # 실내기 파워 온/오프
meterValue = 'value' # 미터기 값
TspValue = 'set_temp' # 설정 온도
TzValue =  'room_temp' # 방 온도
ToaValue = 'outdoor_temp' # 외기 온도도
METHOD = "Randomforest" #Randomforest, Adaboosting, Gradientboosting, Decisiontree

# for j in BATCH_SIZE:
ens = ENSEMBLE(time=time,TRAIN_SIZE=TRAIN_SIZE, GRAD_CLIP=GRAD_CLIP, BATCH_SIZE=BATCH_SIZE,
               LEARNING_RATE=LEARNING_RATE, RANDOM_STATE=RANDOM_STATE, N_JOBS=N_JOBS,
               space=space,  start=start, end=end)

# for i in [909, 910, 921, 920, 919, 917, 918, 911]:
ens.CLASSIFIER(out_unit=909, signal=SIGNAL, target=TARGET, meterValue=meterValue,
               TspValue=TspValue, TzValue=TzValue, ToaValue=ToaValue, method=METHOD)

