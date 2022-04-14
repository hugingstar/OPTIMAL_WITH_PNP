import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import os
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope, array_ops
from tensorflow.python.framework import dtypes
import copy

# gpu 사용하기
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# cpu만 사용하기
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class DEEPMODEL():
    def __init__(self, time, LEARNING_RATE, LAMBDA_L2_REG, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, BATCH_SIZE,
                 NUM_STACK_LAYERS, HIDDEN_DIM, GRADIENT_CLIPPING, KEEP_RATE, TOTAL_ITERATION,
                 DROPOUT, FEED_PREVIOUS, start, end, test_start_time):

        "파일을 호출할 경로"
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

        self.LEARNING_RATE = LEARNING_RATE
        self.LAMBDA_L2_REG = LAMBDA_L2_REG
        self.INPUT_SEQ_LEN = INPUT_SEQ_LEN
        self.OUTPUT_SEQ_LEN = OUTPUT_SEQ_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_STACK_LAYERS = NUM_STACK_LAYERS
        self.HIDDEN_DIM = HIDDEN_DIM
        self.FEED_PREVIOUS = FEED_PREVIOUS
        self.DROPOUT = DROPOUT
        self.GRADIENT_CLIPPING = GRADIENT_CLIPPING
        self.KEEP_RATE = KEEP_RATE
        self.TOTAL_ITERATION = TOTAL_ITERATION

        self.start = start
        self.end = end
        self.test_start_time = test_start_time

        self.start_year = start[:4]
        self.start_month = start[5:7]
        self.start_date = start[8:10]
        self.end_year = end[:4]
        self.end_month = end[5:7]
        self.end_date = end[8:10]
        self.test_year = test_start_time[:4]
        self.test_month = test_start_time[5:7]
        self.test_date = test_start_time[8:10]

        self.folder_name = "{}-{}-{}".format(self.start_year, self.start_month, self.start_date)
        self.create_folder('{}/Deepmodel'.format(self.SAVE_PATH))


    def generate_train_samples(self, x, y, input_seq_len, output_seq_len, batch_size):
        total_start_points = len(x) - input_seq_len - output_seq_len
        # print("[generate_train_samples] total_start_points : {}".format(total_start_points))
        start_x_idx = np.random.choice(range(total_start_points), batch_size, replace=False)
        # print("[generate_train_samples] start_x_idx : {}".format(start_x_idx))
        input_batch_idxs = [list(range(i, i + input_seq_len)) for i in start_x_idx]
        input_seq = np.take(x, input_batch_idxs, axis=0)
        output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in start_x_idx]
        output_seq = np.take(y, output_batch_idxs, axis=0)
        return input_seq, output_seq  # in shape: (batch_size, time_steps, feature_dim)

    def generate_test_samples(self, x, y, input_seq_len, output_seq_len):
        total_samples = x.shape[0]
        input_batch_idxs = [list(range(i, i + input_seq_len)) for i in
                            range((total_samples - input_seq_len - output_seq_len))]
        input_seq = np.take(x, input_batch_idxs, axis=0)
        output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in
                             range((total_samples - input_seq_len - output_seq_len))]
        output_seq = np.take(y, output_batch_idxs, axis=0)
        return input_seq, output_seq

    def build_graph(self, INPUT_DIM, OUTPUT_DIM):
        print("[Build Graph] Input Dimension : {} - Output dimension : {}".format(INPUT_DIM, OUTPUT_DIM))
        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM

        tf.compat.v1.reset_default_graph()
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.compat.v1.GraphKeys.GLOBAL_STEP, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])
        weights = {
            'out': tf.compat.v1.get_variable('Weights_out',
                                             shape=[self.HIDDEN_DIM, self.OUTPUT_DIM],
                                             dtype=tf.float32,
                                             initializer=tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.compat.v1.get_variable('Biases_out',
                                             shape=[self.OUTPUT_DIM],
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(0.)),
        }

        with tf.compat.v1.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.compat.v1.placeholder(tf.float32, shape=(None, self.INPUT_DIM), name="inp_{}".format(t))
                for t in range(self.INPUT_SEQ_LEN)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.compat.v1.placeholder(tf.float32, shape=(None, self.OUTPUT_DIM), name="y".format(t))
                for t in range(self.OUTPUT_SEQ_LEN)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO")] + target_seq[:-1]

            with tf.compat.v1.variable_scope('LSTMCell'):
                cells = []
                for i in range(self.NUM_STACK_LAYERS):
                    with tf.compat.v1.variable_scope('RNN_{}'.format(i)):
                        cell = tf.contrib.rnn.LSTMCell(self.HIDDEN_DIM)
                        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.DROPOUT)
                        cells.append(cell)
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs, initial_state,cell, loop_function=None, scope=None):
                """
                이 함수는 Seq2seq 모델의 RNN 모델을 나타낸다.
                (RNN decoder for the sequence-to-sequence model.)
                """
                with variable_scope.variable_scope(scope or "rnn_decoder"):
                    state = initial_state
                    outputs = []
                    prev = None
                    for i, inp in enumerate(decoder_inputs):
                        print(inp)
                        if loop_function is not None and prev is not None:
                            with variable_scope.variable_scope("loop_function", reuse=True):
                                inp = loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()

                        output, state = cell(inp, state)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = output
                return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                   decoder_inputs,
                                   cell,
                                   feed_previous,
                                   dtype=dtypes.float32,
                                   scope=None):
                """Basic RNN sequence-to-sequence model.
                """
                with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                    enc_cell = copy.deepcopy(cell)
                    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                    if feed_previous:
                        return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                    else:
                        return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
                '''Naive implementation of loop function for _rnn_decoder. '''
                return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(enc_inp, dec_inp, cell, feed_previous=self.FEED_PREVIOUS)
            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.compat.v1.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.compat.v1.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + self.LAMBDA_L2_REG * reg_loss

        with tf.compat.v1.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=self.LEARNING_RATE,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=self.GRADIENT_CLIPPING)

        saver = tf.compat.v1.train.Saver()
        print("[Build Graph Return]{}".format(dict(enc_inp=enc_inp, target_seq=target_seq, train_op=optimizer,
                    loss=loss, saver=saver, reshaped_outputs=reshaped_outputs)))
        return dict(enc_inp=enc_inp, target_seq=target_seq, train_op=optimizer,
                    loss=loss, saver=saver, reshaped_outputs=reshaped_outputs)

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: creating directory. ' + directory)

    def DEEP_PROCESSING(self, out_unit, signal, meterValue, TspValue, ToaValue,
                        target, method, IMP_METHOD, NumOfFeatures):

        self.method = method
        self.imp_method = IMP_METHOD
        self.target = target

        """건물을 인식"""
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        """실외기 데이터"""
        self._outdpath = "{}/{}/Outdoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit)
        self._outdata = pd.read_csv(self._outdpath, index_col=self.TIME)
        for i in list(self.bldginfo[out_unit]):
            self._indpath = "{}/{}/{}/Outdoor_{}_Indoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit, out_unit, i)
            self._indata = pd.read_csv(self._indpath, index_col=self.TIME)


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
            self.outdoor_temp =list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=ToaValue, case=False)])[0]

            # 작동시간 값을 입력
            num = 0
            for o in range(self.data.shape[0] - 1):
                a_o = int(self.data[self.onoffsignal][o]) # 전원 현재 값
                b_o = int(self.data[self.onoffsignal][o + 1]) # 전원 다음 값

                c_o = round(self.data[self.meter_value][o + 1] - self.data[self.meter_value][o], 3) # 미터 값의 차이

                # d_o = round(self.data[self.zone_temp][o + 1] - self.data[self.set_temp][o], 3) # 설정온도_구역온도 차이
                #
                # e_o = round(self.data[self.zone_temp][o + 1] - self.data[self.outdoor_temp][o], 3) # 외기온도_구역온도 차이
                #
                # f_o = round(self.data[self.zone_temp][o + 1] - self.data[self.zone_temp][o], 3)  # 구역온도2_구역온도1 차이

                g_o = round(self.data[self.set_temp][o + 1] - self.data[self.outdoor_temp][o], 3)  # 외기온도_구역온도 차이

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

            #저장할 총 경로
            save = "{}/Deepmodel/{}({})/{}/{}".format(self.SAVE_PATH, self.method, self.imp_method, self.folder_name, out_unit)
            self.create_folder(save)
            self.data.to_csv("{}/Before_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))  # 조건 적용 전

            """
            조건을 적용
            """
            # self.data = self.data[self.data[self.onoffsignal] == 1] #작동중인 데이터만 사용
            # self.data = self.data.dropna(axis=0)

            self.data.to_csv("{}/After_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))  # 조건 적용 후
            # print(out_unit, i, self.data.shape)

            #해당 문자열이 포함되면 그것이 타겟이 된다.
            self.target = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=target, case=False)])[0]
            print("[Target column name] {}".format(self.target))

            # Feature importance 출력값을 사용하여 특징 중요도가 높은 순서대로
            # 모델 학습에 사용
            self._imppath = "{}/Ensemble/{}/{}/{}/IMP_Outdoor_{}_Indoor_{}.csv".format(self.SAVE_PATH, self.imp_method, self.folder_name, out_unit, out_unit, i)
            self._impdata = pd.read_csv(self._imppath, index_col='Unnamed: 0')
            self._impdata= self._impdata.transpose()

            #Feature importance를 고려한다면,
            self.features_all = list(self._impdata.sort_values(by=0, ascending=False).index)
            self.features = self.features_all[:NumOfFeatures] # 중요도가 큰 것 NumOfFeatures개만 사용한다.
            print("[Selected Features] {} - {}".format(len(self.features), self.features))

            # 타겟을 제외한 나머지는 독립변수
            # self.features = list(self.data.columns.difference([self.target]))

            """필요한 컬럼만을 넣어서 데이터 셋을 만든다."""

            df = self.data[self.features]

            df.to_csv("{}/After_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))  # 조건 적용 후
            df.index = pd.to_datetime(df.index)
            df.loc[:, 'hr_sin'] = np.round(np.sin(df.index.hour * (2. * np.pi / 24)), decimals=2) # 시계열 특성을 포함하기 위한 컬럼
            df.loc[:, 'hr_cos'] = np.round(np.cos(df.index.hour * (2. * np.pi / 24)), decimals=2)
            df.loc[:, 'wk_sin'] = np.round(np.sin((df.index.weekday - 1) * (2. * np.pi / 12)), decimals=2)
            df.loc[:, 'wk_cos'] = np.round(np.cos((df.index.weekday - 1) * (2. * np.pi / 12)), decimals=2)
            df.loc[:, self.target] = self.data.loc[:, self.target].copy()
            df.loc[:, 'inputy'] = self.data.loc[:, self.target].copy()

            # TST : 테스트 셋을 시작하는 시간 길이
            TST =  datetime.datetime(int(self.test_year), int(self.test_month), int(self.test_date))
            df_train = df.loc[df.index <= TST]
            df_test = df.loc[df.index >= TST]

            #시계열 기준으로 정렬
            df_train = df_train.sort_values(self.TIME)
            df_test = df_test.sort_values(self.TIME)

            # 중복값 제거
            df_train = df_train.drop_duplicates(keep='first') #중복 값이 있다면, 첫번째 값을 남기고 제거
            df_test = df_test.drop_duplicates(keep='first')

            # 데이터 확인
            df_train.to_csv("{}/BldgRawData_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i), sep=',', float_format='%.2f')
            df_test.to_csv("{}/BldgRawData_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i), sep=',', float_format='%.2f')

            # 시계열을 제거했다.
            df_train = df_train.reset_index().drop(self.TIME, 1) #TST 이전
            df_test = df_test.reset_index().drop(self.TIME, 1) # TST 이후

            #x_col: 최종적으로 학습에 들어가는 Feature list
            x_col = list(df_train.columns)
            x_col.remove(self.target)

            # 시계열 인덱스는 제거되었다.
            X_train = df_train[x_col].values.copy() # X_train : TST 데이터 이전의 데이터
            y_train = df_train[self.target].values.copy().reshape(-1, 1) # y_train : TST 이전의 타겟 값
            X_test = df_test[x_col].values.copy() # X_test : TST 이후의 데이터
            y_test = df_test[self.target].values.copy().reshape(-1, 1) #y_test : TST 이후의 타켓ㅅ값
            print("[Train and Test dataset] X_train : {} - X_test : {} - y_train : {} - y_test : {}"
                  .format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

            # Parameter for normalizing
            mean = []
            std = []
            param_col = []
            for _s in range(X_train.shape[1] - 4):
                if X_train[:, _s].std() <= 0:
                    print(_s)
                else:
                    temp_mean = X_train[:, _s].mean()
                    temp_std = X_train[:, _s].std()
                    X_train[:, _s] = (X_train[:, _s] - temp_mean) / temp_std
                    X_test[:, _s] = (X_test[:, _s] - temp_mean) / temp_std
                param_col.append(x_col[_s])
                mean.append(temp_mean)
                std.append(temp_std)

            # z-score transform y
            self.y_mean = y_train.mean()
            self.y_std = y_train.std()
            print('y_std : {} - y_mean : {}'.format(round(self.y_std, 3), round(self.y_mean,3)))

            y_train = (y_train - self.y_mean) / self.y_std
            y_test = (y_test - self.y_mean) / self.y_std
            param_col.append(self.target)

            mean.append(self.y_mean)
            std.append(self.y_std)
            param = [mean, std]
            print("[param check] mean : {} ".format(mean))
            print("[param check] std : {}".format(std))
            norm_param = pd.DataFrame(param, index=['mean', 'std'], columns=param_col)
            norm_param.to_csv("{}/Normalize_parameter_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))

            pd.DataFrame(X_train).to_csv("{}/Xtrain_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))
            pd.DataFrame(y_train).to_csv("{}/ytrain_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))
            pd.DataFrame(X_test).to_csv("{}/Xtest_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))
            pd.DataFrame(y_test).to_csv("{}/ytrain_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, i))

            # x, y = self.generate_train_samples(x=X_train, y=y_train,
            #                                    input_seq_len=self.INPUT_SEQ_LEN,
            #                                    output_seq_len=self.OUTPUT_SEQ_LEN,
            #                                    batch_size=self.BATCH_SIZE)
            # train_x, train_y = self.generate_test_samples(x=X_train, y=y_train,
            #                                               input_seq_len=self.INPUT_SEQ_LEN,
            #                                               output_seq_len=self.OUTPUT_SEQ_LEN)
            # test_x, test_y = self.generate_test_samples(x=X_test, y=y_test,
            #                                             input_seq_len=self.INPUT_SEQ_LEN,
            #                                             output_seq_len=self.OUTPUT_SEQ_LEN)

            #Build graph
            rnn_model = self.build_graph(INPUT_DIM=int(X_train.shape[1]), OUTPUT_DIM=int(y_train.shape[1]))
            saver = tf.compat.v1.train.Saver()
            self.TRAIN_PROCESS(rnn_model=rnn_model, X_train=X_train, y_train=y_train, save=save, out_unit=out_unit, iterNum=i) #save : 저장경로

            test_model = self.build_graph(INPUT_DIM=int(X_test.shape[1]), OUTPUT_DIM=int(y_test.shape[1]))
            saver = tf.compat.v1.train.Saver()
            self.TEST_PROCESS(test_model=test_model, X_test=X_test, y_test=y_test, save=save, out_unit=out_unit, iterNum=i)


    def TRAIN_PROCESS(self, rnn_model, X_train, y_train, out_unit, save, iterNum):
            saver = tf.compat.v1.train.Saver()
            init = tf.compat.v1.global_variables_initializer()
            loss_fun = []
            with tf.compat.v1.Session() as sess:
                sess.run(init) # 초기화
                for oo_ in range(self.TOTAL_ITERATION):
                    # X_train : TST 이전의 데이터
                    # y_train : TST 이전의 타겟
                    # print("[Model Training Sample] X_train : {} - y_train : {}".format(X_train.shape, y_train.shape))
                    batch_input, batch_output = self.generate_train_samples(x=X_train,
                                                                            y=y_train,
                                                                            input_seq_len=self.INPUT_SEQ_LEN,
                                                                            output_seq_len=self.OUTPUT_SEQ_LEN,
                                                                            batch_size=self.BATCH_SIZE)
                    # print("[Batch input] batch_input : {} - batch_output : {}".format(batch_input.shape, batch_output.shape))
                    # 배치 사이즈 만큼 잘라서 모델에 입력한다.
                    # rnn model의 enc_inp의 t번째에 입력 배치값을 순서대로 넣는다.
                    feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t] for t in range(self.INPUT_SEQ_LEN)}
                    # rnn model의 targer_seq에 출려 배치값을 순서대로 업데이트한다.
                    # update 함수는
                    feed_dict.update({rnn_model['target_seq'][t]: batch_output[:, t] for t in range(self.OUTPUT_SEQ_LEN)})
                    # print("{} - {}".format(rnn_model['train_op'], rnn_model['loss']))
                    _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
                    if i % 100 == 0:
                        print("[step: {}] loss: {} - {}".format(i, loss_t, datetime.datetime.now()))
                        loss_fun.append([i, loss_t])
                    if loss_t < 0.001:
                        print("========== loss break < 0.001 ==========\n")
                        break
                    if pd.isnull(loss_t) == True:
                        print("========== loss is null break ==========\n")
                        break

                temp_saver = rnn_model['saver']()
                save_path = temp_saver.save(sess, "{}/Model_Outdoor_{}_Indoor_{}".format(save, out_unit, iterNum))
            print("[Checkpoint saved at] {}".format(save))
            loss_result = pd.DataFrame(loss_fun, columns=['iteration', 'loss'])
            loss_result.to_csv('{}/lossResult_Outdoor_{}_Indoor_{}.csv'.format(save, out_unit, iterNum))

    def TEST_PROCESS(self, test_model, X_test, y_test, save, out_unit, iterNum):
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            saver = test_model['saver']().restore(sess, "{}/Model_Outdoor_{}_Indoor_{}".format(save, out_unit, iterNum))
            feed_dict = {test_model['enc_inp'][t]: X_test[:, t, :] for t in
                         range(self.INPUT_SEQ_LEN)}  # batch prediction
            feed_dict.update(
                {test_model['target_seq'][t]: np.zeros([X_test.shape[0], y_test.shape[1]], dtype=np.float32) for t in
                 range(self.OUTPUT_SEQ_LEN)})
            final_preds = sess.run(test_model['reshaped_outputs'], feed_dict=feed_dict)  # type : list
            final_preds = [np.expand_dims(pred, axis=1) for pred in final_preds]
            final_preds2 = np.array(final_preds)
            final_preds2 = final_preds2.reshape(final_preds2.shape[0],
                                                final_preds2.shape[1] * final_preds2.shape[2])  # (8620, 10)
            np.savetxt("{}/Model_Outdoor_{}_Indoor_{}_testresult_y2.csv".format(save, out_unit, iterNum), final_preds2,
                       fmt='%.2f', delimiter=',')
            final_preds = np.concatenate(final_preds, axis=1)  # (8620, 10, 1)
            final_preds3 = np.array(final_preds)  # (8620, 10, 1)
            final_preds3 = final_preds3.reshape(final_preds3.shape[0],
                                                final_preds3.shape[1] * final_preds3.shape[2])  # (8620, 10)
            np.savetxt("{}/Model_Outdoor_{}_Indoor_{}_testresult_y3.csv".format(save, out_unit, iterNum), final_preds3,
                       fmt='%.2f', delimiter=',')

        # remove duplicate hours and concatenate into one long array
        test_y_expand = np.concatenate([y_test[i].reshape(-1) for i in range(0, y_test.shape[0], self.OUTPUT_SEQ_LEN)],
                                       axis=0)
        final_preds_expand = np.concatenate(
            [final_preds[i].reshape(-1) for i in range(0, final_preds.shape[0], self.OUTPUT_SEQ_LEN)], axis=0)
        np.savetxt("{}/Model_Outdoor_{}_Indoor_{}_final_preds_expand.csv".format(save, out_unit, iterNum),
                   final_preds_expand,
                   fmt='%.2f', delimiter=',')
        """
        결과데이터 출력 과정(Nomalization 복구 과정 포함)
        """
        final_preds_expand2 = final_preds_expand * self.y_std + self.y_mean
        print('y_std, y_mean', self.y_std, self.y_mean)
        test_y_expand2 = test_y_expand * self.y_std + self.y_mean
        Y_pred = pd.DataFrame(final_preds_expand2)
        Y = pd.DataFrame(test_y_expand2)
        Y_result = pd.concat([Y, Y_pred], ignore_index=True, axis=1)
        Y_result.columns = ['Test', 'Prediction']
        print("Y_result : {} - {}".format(type(Y_result), Y_result.shape))
        Y_result.to_csv("{}/Model_Outdoor_{}_Indoor_{}_RNN_Test.csv".format(save, out_unit, iterNum))

TIME = 'updated_time'
TARGET = "room_temp"

INPUT_SEQ_LEN = 10
OUTPUT_SEQ_LEN = 10
BATCH_SIZE = 500
LEARNING_RATE = 0.01
LAMBDA_L2_REG = 0.003

HIDDEN_DIM = 128
NUM_STACK_LAYERS = 4
DROPOUT = 0.5

KEEP_RATE = 0.7
GRADIENT_CLIPPING = 2.5
FEED_PREVIOUS = False

TOTAL_ITERATION = 5000

start ='2021-01-01'
end = '2021-03-31'
test_start_time = '2021-03-27'

SIGNAL = 'indoor_power'
meterValue = 'value'
TspValue = 'set_temp'
# TzValue =  'room_temp'
ToaValue = 'outdoor_temp'
METHOD = "Seq2seq"
IMP_METHOD = "Adaboosting"

NumOfFeatures = 5

DML = DEEPMODEL(time=TIME, LEARNING_RATE=LEARNING_RATE, LAMBDA_L2_REG=LAMBDA_L2_REG, BATCH_SIZE=BATCH_SIZE,
                INPUT_SEQ_LEN=INPUT_SEQ_LEN, OUTPUT_SEQ_LEN=OUTPUT_SEQ_LEN, NUM_STACK_LAYERS=NUM_STACK_LAYERS,
                HIDDEN_DIM=HIDDEN_DIM, GRADIENT_CLIPPING=GRADIENT_CLIPPING, KEEP_RATE=KEEP_RATE,
                TOTAL_ITERATION=TOTAL_ITERATION, FEED_PREVIOUS=FEED_PREVIOUS, DROPOUT=DROPOUT,
                start=start, end=end, test_start_time=test_start_time)

for i in [909]:#, 910, 921, 920, 919, 917, 918, 911]:
    DML.DEEP_PROCESSING(out_unit=i, signal=SIGNAL, meterValue=meterValue,
                        TspValue=TspValue, ToaValue=ToaValue,
                        target=TARGET, method=METHOD, IMP_METHOD=IMP_METHOD,
                        NumOfFeatures=NumOfFeatures)