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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# cpu만 사용하기
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class DEEPMODEL():
    def __init__(self, time, LEARNING_RATE, LAMBDA_L2_REG, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, BATCH_SIZE,
                 NUM_STACK_LAYERS, HIDDEN_DIM, GRADIENT_CLIPPING, KEEP_RATE, TOTAL_ITERATION,
                 DROPOUT, LAYERS, PERCENTAGE, FEED_PREVIOUS, start, end, test_start_time):

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

        #HyperPrameter
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

        # AutoEncoder HyperParameter
        self.LAYERS = LAYERS
        self.PERCENTAGE = PERCENTAGE

        #Time range Setting
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

        # 시작전에 폴더를 생성
        self.folder_name = "{}-{}-{}".format(self.start_year, self.start_month, self.start_date)
        self.create_folder('{}/Deepmodel'.format(self.SAVE_PATH)) # Deepmodel 폴더를 생성


    def generate_train_samples(self, x, y, input_seq_len, output_seq_len, batch_size):
        #이 함수는 Training Sample을 생성하는 코드이다.
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
        # 이 함수는 테스트 셋을 만드는 함수이다.
        total_samples = x.shape[0]
        input_batch_idxs = [list(range(i, i + input_seq_len)) for i in
                            range((total_samples - input_seq_len - output_seq_len))]
        input_seq = np.take(x, input_batch_idxs, axis=0)
        output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in
                             range((total_samples - input_seq_len - output_seq_len))]
        output_seq = np.take(y, output_batch_idxs, axis=0)
        return input_seq, output_seq

    def build_graph(self, INPUT_DIM, OUTPUT_DIM):
        """
        이 메소드는 Seq2seq 모델 그래프이다. 입력 값으로는 만들어진 데이터의 차원(X 변수 개수) 값이 입력된다.
        :param INPUT_DIM: 입력 디멘젼
        :param OUTPUT_DIM: 출력 디멘젼
        :return: dictionary 형태의 파라미터 모아둔 것
        """
        print("[Build Graph] Input Dimension : {} - Output dimension : {} - Feed Previous : {} - Dropout : {}"
              .format(INPUT_DIM, OUTPUT_DIM, self.FEED_PREVIOUS, self.DROPOUT))

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
                for nst in range(self.NUM_STACK_LAYERS):
                    with tf.compat.v1.variable_scope('RNN_{}'.format(nst)):
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
                        # print(inp)
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
            reshaped_outputs = [tf.matmul(ds, weights['out']) + biases['out'] for ds in dec_outputs]

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

        saver = tf.compat.v1.train.Saver
        print("[Build Graph Return]{}".format(dict(enc_inp=enc_inp, target_seq=target_seq, train_op=optimizer,
                    loss=loss, saver=saver, reshaped_outputs=reshaped_outputs)))

        return dict(enc_inp=enc_inp, target_seq=target_seq, train_op=optimizer,
                    loss=loss, saver=saver, reshaped_outputs=reshaped_outputs)

    def build_graph_With_Attention(self, INPUT_DIM, OUTPUT_DIM):
        """
        :param INPUT_DIM: 입력값 차원 (X 변수의 개수)
        :param OUTPUT_DIM: 출력 차원(예측 하고자하는 것 ex. 실내 온도 차원 1)
        :return: 그래프 관련 값의 딕셔너리
        """
        print("[Build Graph With Attention] Input Dimension : {} - Output dimension : {} - Feed Previous : {} - Dropout : {}"
              .format(INPUT_DIM, OUTPUT_DIM, self.FEED_PREVIOUS, self.DROPOUT))

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
        print("weights: {}".format(weights))
        biases = {
            'out': tf.compat.v1.get_variable('Biases_out',
                                             shape=[self.OUTPUT_DIM],
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(0.)),
        }
        print("biases: {}".format(biases))

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

            """decoder initial value, GO value"""
            dec_inp = [tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO")] + target_seq[:-1]

            """Encoder"""
            with variable_scope.variable_scope("rnn_encoder"):
                print("==========[Encoder] LSTMCell information ==========")
                with tf.compat.v1.variable_scope('LSTMCell'):
                    cells = []
                    for i in range(self.NUM_STACK_LAYERS):  # num_stacked_layers=4 이고
                        scope_message = 'RNN_{}'.format(i)
                        with tf.compat.v1.variable_scope(scope_message):
                            cell = tf.contrib.rnn.LSTMCell(num_units=self.HIDDEN_DIM)  # LSTM cell 하나 넣었고,
                            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.DROPOUT)
                            cells.append(cell)  # 리스트에 추가
                            print(
                                "scope_message: {} - hidden_dim : {} - cell:{}".format(scope_message, self.HIDDEN_DIM, cell))
                    print("[cells] len : {} - {} - {}".format(len(cells), type(cells), cells))
                    enc_cell = tf.contrib.rnn.MultiRNNCell(cells)
                    print("[enc_cell] {} - {}".format(type(enc_cell), enc_cell))
                    # Dynamic RNN으로 바꾸기
                enc_outputs, enc_state = rnn.static_rnn(enc_cell, enc_inp, dtype=tf.float32)
                print("==========================================\n")

                print("==========[Encoder] Static RNN ==========")
                print("[enc_inp] {} - {} - {}".format(len(enc_inp), type(enc_inp), enc_inp))
                print("[enc_outputs] {} - {} - {}".format(len(enc_inp), type(enc_outputs), enc_outputs))
                print("[enc_state] {} - {} - {}".format(array_ops.shape(enc_state)[0], type(enc_state), enc_state))
                print("==========================================\n")

            def _loop_function(prev, _):
                '''Naive implementation of loop function for _rnn_decoder. '''
                return tf.matmul(prev, weights['out']) + biases['out']

            # dec_states = tf.ones_like(enc_state) #tf.ones_like(enc_state)

            """Decoder"""
            feed_previous = self.FEED_PREVIOUS  # train은 False, test는 True
            if feed_previous:
                with variable_scope.variable_scope("rnn_decoder"):
                    print("==========[Decoder] LSTMCell information ==========")
                    with tf.compat.v1.variable_scope('LSTMCell'):
                        cells = []
                        for i in range(self.NUM_STACK_LAYERS):  # num_stacked_layers=4
                            scope_message = 'RNN_{}'.format(i)
                            with tf.compat.v1.variable_scope(scope_message):
                                cell = tf.contrib.rnn.LSTMCell(num_units=self.HIDDEN_DIM)
                                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.DROPOUT)
                                print("scope_message: {} - hidden_dim : {} - cell:{}".format(scope_message, self.HIDDEN_DIM,
                                                                                             cell))
                                cells.append(cell)
                            # print("[cells] len : {} - {} - {}".format(len(cells), type(cells), cells))
                        dec_cell = tf.contrib.rnn.MultiRNNCell(cells)
                        print("[dec_cell] {} - {}".format(type(dec_cell), enc_cell))
                    print("==========================================\n")
                    prev = None
                    dec_outputs = []
                    for i, de_inp in enumerate(dec_inp):
                        # print(i, de_inp)
                        if _loop_function is not None and prev is not None:
                            with variable_scope.variable_scope("loop_function", reuse=True):
                                de_inp = _loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()
                        dec_output, dec_state = dec_cell(de_inp, enc_state)  # enc_state
                        dec_outputs.append(dec_output)
                        if _loop_function is not None:
                            prev = dec_output
            else:
                with variable_scope.variable_scope("rnn_decoder"):
                    print("==========[Decoder] LSTMCell information ==========")
                    with tf.compat.v1.variable_scope('LSTMCell'):
                        cells = []
                        for i in range(self.NUM_STACK_LAYERS):  # num_stacked_layers=4 이고
                            scope_message = 'RNN_{}'.format(i)
                            with tf.compat.v1.variable_scope(scope_message):
                                cell = tf.contrib.rnn.LSTMCell(num_units=self.HIDDEN_DIM)
                                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.DROPOUT)
                                print("scope_message: {} - hidden_dim : {} - cell:{}".format(scope_message, self.HIDDEN_DIM,
                                                                                             cell))
                                cells.append(cell)
                            # print("[cells] len : {} - {} - {}".format(len(cells), type(cells), cells))
                        dec_cell = tf.contrib.rnn.MultiRNNCell(cells)
                        print("[dec_cell] {} - {}".format(type(dec_cell), dec_cell))
                    print("==========================================\n")

                    prev = None
                    dec_outputs = []
                    for i, de_inp in enumerate(dec_inp):
                        print(i, de_inp)
                        if _loop_function is not None and prev is not None:
                            with variable_scope.variable_scope("loop_function", reuse=True):
                                de_inp = _loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()
                        dec_output, dec_state = dec_cell(de_inp, enc_state)
                        dec_outputs.append(dec_output)
                        if _loop_function is not None:
                            prev = dec_output

            print("==========[Decoder] Static RNN ==========")
            print("[dec_inp] {} - {} - {}".format(len(dec_inp), type(dec_inp), dec_inp))
            print("[dec_outputs] {} - {} - {}".format(len(dec_outputs), type(dec_outputs), dec_outputs))
            print("[dec_state] {} - {} - {}".format(len(dec_state), type(dec_state), dec_state))
            print("==========================================\n")

            """Encoder - cell state/hidden state"""
            enc_cell_states = tf.concat([tf.expand_dims(state, 1) for state in enc_state], axis=1)[0]
            print("[enc_cell_states] {} - {} - {}".format(enc_cell_states.get_shape(), type(enc_cell_states),
                                                          enc_cell_states))  # (4, ?, 128)
            enc_hidden_states = tf.concat([tf.expand_dims(state, 1) for state in enc_state], axis=1)[1]
            print("[enc_hidden_states] {} - {} - {}".format(enc_hidden_states.get_shape(), type(enc_hidden_states),
                                                            enc_hidden_states))  # (4, ?, 128)

            """Decoder - cell state/hidden state"""
            dec_cell_states = tf.concat([tf.expand_dims(state, 1) for state in dec_state], axis=1)[0]
            print("[dec_cell_states] {} - {} - {}".format(dec_cell_states.get_shape(), type(dec_cell_states),
                                                          dec_cell_states))  # (4, ?, 128)
            dec_hidden_states = tf.concat([tf.expand_dims(state, 1) for state in dec_state], axis=1)[1]
            print("[dec_hidden_states] {} - {} - {}".format(dec_hidden_states.get_shape(), type(dec_hidden_states),
                                                            dec_hidden_states))  # (4, ?, 128)

            """Convert to tensor"""
            print("==========[Enc-Dec] Convert to tensor ==========")
            enc_state = tf.convert_to_tensor(enc_state)
            print("[enc_state] {} - {} - {}".format(enc_state.get_shape(), type(enc_state), enc_state))
            dec_state = tf.convert_to_tensor(dec_state)
            print("[dec_state] {} - {} - {}".format(dec_state.get_shape(), type(dec_state), dec_state))

            enc_outputs = tf.convert_to_tensor(enc_outputs)
            print("[enc_outputs] {} - {} - {}".format(enc_outputs.get_shape(), type(enc_outputs), enc_outputs))
            dec_outputs = tf.convert_to_tensor(dec_outputs)
            print("[dec_outputs] {} - {} - {}".format(dec_outputs.get_shape(), type(dec_outputs), dec_outputs))

            """Attention layer"""
            with variable_scope.variable_scope("attn_mechanism"):  # attention mechanism
                with tf.variable_scope("attn_score"):  # attention score
                    trs_dec_outputs = tf.transpose(dec_outputs, perm=[0, 2, 1])
                    print("[trs_dec_outputs] {} - {} - {}".format(trs_dec_outputs.get_shape(), type(trs_dec_outputs), trs_dec_outputs))
                    score = tf.matmul(enc_outputs, trs_dec_outputs)  # dot product : 마지막 state,
                    # score = tf.multiply(score, tf.math.sqrt(float(self.INPUT_DIM)))
                    # score = tf.nn.relu(score, name="score")
                    print("[score] {} - {} - {}".format(score.get_shape(), type(score), score))
                with tf.variable_scope("attn_align"):  # softmax - attention distribution - attention weight
                    alphas = tf.nn.softmax(score, name="alphas")  # 총합이 1이 되도록 alignment 실행
                    print("[alphas] {} - {} - {}".format(type(alphas), alphas.shape, alphas))  # 시간의 가중치
                with tf.variable_scope("context_vector"):  # attention outputs
                    context_vec = tf.reduce_sum(tf.matmul(alphas, enc_outputs), axis=1, name="context")  # Transpose, 곱하는 순서
                    print("[context_vec] {} - {} - {}".format(type(context_vec), context_vec.shape, context_vec))
                    context_vec = tf.expand_dims(context_vec, axis=1, name="context_vec") # 연산을 위해 차원을 정렬해준다
                    print("[context_vec] {} - {} - {}".format(type(context_vec), context_vec.shape, context_vec))
                    print("[dec_outputs] {} - {} - {}".format(type(dec_outputs), dec_outputs.shape, dec_outputs))
                with tf.variable_scope("attn_outputs"):
                    attn_outputs = tf.multiply(context_vec, dec_outputs)
                    # attn_outputs = tf.nn.tanh(attn_outputs, name="attn_outputs")
                    print("[attn_outputs] {} - {} - {}".format(type(attn_outputs), attn_outputs.get_shape(),
                                                               attn_outputs))
                    attn_dec_outputs_list = []
                    for j in range(attn_outputs.get_shape()[0]):
                        print(attn_outputs[j])
                        attn_dec_outputs_list.append(attn_outputs[j])
                    print("[attn_dec_outputs_list] {} - {} - {}".format(len(attn_dec_outputs_list),
                                                                        type(attn_dec_outputs_list),
                                                                        attn_dec_outputs_list))
            print("[weights['out']] {} - {}".format(type(weights['out']), weights['out']))
            print("[biases['out']] {} - {}".format(type(biases['out']), biases['out']))
            for k in attn_dec_outputs_list:  # 확인용
                print(k)
            # reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in attn_dec_outputs_list]  # attention
            reshaped_outputs = [tf.nn.tanh(tf.matmul(i, weights['out']) + biases['out']) for i in attn_dec_outputs_list]
            print("[reshaped_outputs] {} - {} - {}".format(len(reshaped_outputs), type(reshaped_outputs),
                                                           reshaped_outputs))

            """Loss step"""
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
                optimizer = tf.contrib.layers.optimize_loss(loss=loss,
                                                            learning_rate=self.LEARNING_RATE,
                                                            global_step=global_step,
                                                            optimizer='Adam',
                                                            clip_gradients=self.GRADIENT_CLIPPING)
            saver = tf.compat.v1.train.Saver
            print("[Build Graph With Attention Return]{}".format(
                dict(enc_inp=enc_inp, target_seq=target_seq, train_op=optimizer,
                     loss=loss, saver=saver, reshaped_outputs=reshaped_outputs, attn_outputs=attn_outputs)))

            return dict(enc_inp=enc_inp, target_seq=target_seq, train_op=optimizer,
                        loss=loss, saver=saver, reshaped_outputs=reshaped_outputs, attn_outputs=attn_outputs)

    def build_Autoencoder(self, INPUT_DIM):
        """
        오토인코더 그래프이다. 오토인코더 그래프로 입력 데이터의 차원을 낮추거나 높여서 예측 성능을 높일 수 있는
        입력 데이터를 만든다. 이후에 Seq2seq를 후행으로 배치하여 모델을 학습하고 테스트해보는 것이다.
        즉, 아래에 DEEP_PROCESSING 함수에서 AutoEncoder 란을 확인하면 self.method가 AutoEncoder가 실행된 후에
        바로 Seq2seq가 실행되도록 되어 있다.
        :param INPUT_DIM: 입력값으로는 입력 데이터 차원(X 개수)이 있다.
        :return: 오토인코더의 파라미터 딕셔너리
        """
        print("[Build AutoEncoder] Input Dimension : {} - Layers : {} - Percentage : {} - Feed Previous : {} - Dropout : {}"
              .format(INPUT_DIM, self.LAYERS, self.PERCENTAGE, self.FEED_PREVIOUS, self.DROPOUT))

        self.INPUT_DIM = INPUT_DIM

        tf.compat.v1.reset_default_graph()
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.compat.v1.GraphKeys.GLOBAL_STEP, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])

        _training = tf.global_variables_initializer()
        _session = tf.Session()
        val = int(self.INPUT_DIM * (1 - PERCENTAGE) / self.LAYERS)
        print('t', (self.INPUT_DIM - val * (self.LAYERS - 1)))

        enc_inp = [
            tf.compat.v1.placeholder(tf.float32, shape=(None, self.INPUT_DIM), name="inp_{}".format(t))
            for t in range(self.INPUT_SEQ_LEN)
        ]

        weight = tf.compat.v1.get_variable('Weights',
                                           shape=[(self.INPUT_DIM - val), self.INPUT_DIM],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer())
        bias = tf.compat.v1.get_variable('Biases',
                                         shape=[self.INPUT_DIM],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer())

        with tf.compat.v1.variable_scope('Auto_encoder'):
            # Encoder
            def encoder(enc_inp, LAYERS, PERCENTAGE):
                enc_weights = []
                enc_biases = []
                enc_hiddens = []
                for layer in range(LAYERS - 1):
                    enc_weights.append(tf.compat.v1.get_variable('Enc_Weights_{}'.format(layer),
                                                                 shape=[(INPUT_DIM - val * (layer)),
                                                                        (INPUT_DIM - val * (layer + 1))],
                                                                 dtype=tf.float32,
                                                                 initializer=tf.truncated_normal_initializer()
                                                                 ))
                    enc_biases.append(tf.compat.v1.get_variable('Enc_Biases_{}'.format(layer),
                                                                shape=[(INPUT_DIM - val * (layer + 1))],
                                                                dtype=tf.float32,
                                                                initializer=tf.constant_initializer()
                                                                ))
                    if layer == 0:
                        enc_hiddens.append(tf.nn.relu(tf.add(tf.matmul(enc_inp, enc_weights[-1]), enc_biases[-1])))
                    else:
                        enc_hiddens.append(
                            tf.nn.relu(tf.add(tf.matmul(enc_hiddens[-1], enc_weights[-1]), enc_biases[-1])))

                enc_weights.append(tf.compat.v1.get_variable('Enc_Weights',
                                                             shape=[(INPUT_DIM - val * (LAYERS - 1)),
                                                                    int(INPUT_DIM * PERCENTAGE)],
                                                             dtype=tf.float32,
                                                             initializer=tf.truncated_normal_initializer()
                                                             ))
                enc_biases.append(tf.compat.v1.get_variable('Enc_Biases',
                                                            shape=[int(INPUT_DIM * PERCENTAGE)],
                                                            dtype=tf.float32,
                                                            initializer=tf.constant_initializer()
                                                            ))
                enc_hiddens.append(tf.nn.relu(tf.add(tf.matmul(enc_hiddens[-1], enc_weights[-1]), enc_biases[-1])))
                return enc_hiddens[-1], decoder(enc_hiddens[-1], LAYERS, PERCENTAGE)

            def decoder(enc_hidden, LAYERS, PERCENTAGE):
                dec_weights = []
                dec_biases = []
                dec_hiddens = []
                for layer in range(LAYERS - 1):
                    if layer == 0:
                        dec_weights.append(tf.compat.v1.get_variable('dec_Weights_{}'.format(layer),
                                                                     shape=[int(INPUT_DIM * PERCENTAGE),
                                                                            (INPUT_DIM - val * (LAYERS - layer - 1))],
                                                                     dtype=tf.float32,
                                                                     initializer=tf.truncated_normal_initializer()
                                                                     ))
                    else:
                        dec_weights.append(tf.compat.v1.get_variable('dec_Weights_{}'.format(layer),
                                                                     shape=[(INPUT_DIM - val * (LAYERS - layer)),
                                                                            (INPUT_DIM - val * (LAYERS - layer - 1))],
                                                                     dtype=tf.float32,
                                                                     initializer=tf.truncated_normal_initializer()
                                                                     ))
                    dec_biases.append(tf.compat.v1.get_variable('dec_Biases_{}'.format(layer),
                                                                shape=[(INPUT_DIM - val * (LAYERS - layer - 1))],
                                                                dtype=tf.float32,
                                                                initializer=tf.constant_initializer()
                                                                ))
                    if layer == 0:
                        dec_hiddens.append(tf.nn.relu(tf.add(tf.matmul(enc_hidden, dec_weights[-1]), dec_biases[-1])))
                    else:
                        dec_hiddens.append(
                            tf.nn.relu(tf.add(tf.matmul(dec_hiddens[-1], dec_weights[-1]), dec_biases[-1])))
                return dec_hiddens[-1]

            enc_output, dec_output = encoder(enc_inp=enc_inp, LAYERS=self.LAYERS, PERCENTAGE=self.PERCENTAGE)
            reshaped_output = tf.matmul(dec_output, weight) + bias

        with tf.compat.v1.variable_scope('Loss'):
            # L2 loss
            output_loss = tf.reduce_mean(tf.pow(reshaped_output - enc_inp, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.compat.v1.trainable_variables():
                if 'Bias' in tf_var.name or 'Weight' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + self.LAMBDA_L2_REG * reg_loss

        with tf.compat.v1.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(loss=loss, learning_rate=self.LEARNING_RATE, global_step=global_step,
                                                        optimizer='Adam', clip_gradients=self.GRADIENT_CLIPPING)
        saver = tf.compat.v1.train.Saver
        return dict(enc_inp=enc_inp, train_op=optimizer, loss=loss, reshaped_output=reshaped_output,
                    enc_output=enc_output, saver=saver)

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

    def DEEP_PROCESSING(self, out_unit, signal, meterValue, TspValue, ToaValue,
                        target, method, IMP_METHOD, NumOfFeatures):
        """
        이 함수는 그래프를 호출, 모델 트레이닝 진행, 테스트 진행을 관장하는 함수이다.
        :param out_unit: 실외기 넘버 (실외기에 연결된 실내기는 위의 딕셔너리에서 호출한다.)
        :param signal: ON-OFF 시그널
        :param meterValue: 미터 값 어구
        :param TspValue: 설정 온도 어구
        :param ToaValue: 실외 온도 어구
        :param target: 타겟 값
        :param method: 딥러닝 학습 방법 (Seq2seq ,Attention, AutoEncoder)
        :param IMP_METHOD: 특징중요도 학습 방법
        (Randomforest, Randomforest, Adaboosting, Gradientboosting)
        :param NumOfFeatures: 특징중요도 결과에서 상위 몇 개를 선택할지를 결정할 숫자
        :return:
        """
        #딥러닝 모델 생성 method 입력값
        self.method = method

        #특징중요도 방법 입력값
        self.imp_method = IMP_METHOD

        #예측 대상
        self.target = target

        # 건물 인식(딕셔너리에서 포함된 건물로 인식한다.)
        if out_unit in self.jinli.keys():
            self.bldg_name ="Jinli"
            self.bldginfo = self.jinli
        elif out_unit in self.dido.keys():
            self.bldg_name = "Dido"
            self.bldginfo = self.dido

        #실외기 데이터
        self._outdpath = "{}/{}/Outdoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit)
        self._outdata = pd.read_csv(self._outdpath, index_col=self.TIME)
        for indv in list(self.bldginfo[out_unit]):
            #실내기 데이터
            self._indpath = "{}/{}/{}/Outdoor_{}_Indoor_{}.csv".format(self.DATA_PATH, self.folder_name, out_unit, out_unit, indv)
            self._indata = pd.read_csv(self._indpath, index_col=self.TIME)

            #실내기 및 실외기의 데이터 통합
            self.data = pd.concat([self._outdata, self._indata], axis=1)
            self.data.index.names = [self.TIME] # 인덱스 컬럼명이 없는 경우를 대비하여 보완

            #문자열로 된 원본 데이터의 '모드'를 숫자로 변환
            self.data = self.data.replace({"High": 3, "Mid" : 2, "Low" : 1, "Auto" : 4})

            # 메타데이터의 풀 네임은 아주 길기 때문에 해당 어구가 포함된 컬럼 찾아서 모델 학습에 사용한다.
            # ON-OFF 시그널
            self.onoffsignal = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=signal, case=False)])[0]
            # 미터 값
            self.meter_value = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=meterValue, case=False)])[0]
            #설정 온도
            self.set_temp = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=TspValue, case=False)])[0]
            # 외기 온도
            self.outdoor_temp =list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=ToaValue, case=False)])[0]

            # 작동시간 값을 입력
            num = 0
            for o in range(self.data.shape[0] - 1):
                a_o = int(self.data[self.onoffsignal][o]) # 전원 현재 값
                b_o = int(self.data[self.onoffsignal][o + 1]) # 전원 다음 값

                c_o = round(self.data[self.meter_value][o + 1] - self.data[self.meter_value][o], 3) # 미터 값의 차이
                # 구역 온도를 예측해야하므로 주석처리 해놓음
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

            # 데이터 생성된것을 시계열을 기준으로 정렬 하고 결측값 처리
            self.data = self.data.sort_values(self.TIME)
            self.data = self.data.fillna(method='ffill') #결측값 처리

            #저장할 총 경로
            save = "{}/Deepmodel/{}({})/{}/{}".format(self.SAVE_PATH, self.method, self.imp_method, self.folder_name, out_unit)
            self.create_folder(save)
            self.data.to_csv("{}/Before_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))  # 조건 적용 전

            """필요한 경우 조건을 적용하는 장소이다."""
            # self.data = self.data[self.data[self.onoffsignal] == 1] #작동중인 데이터만 사용
            # self.data = self.data.dropna(axis=0) # 결측값을 그냥 날리는 경우

            self.data.to_csv("{}/After_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))  # 조건 적용 후
            # print(out_unit, i, self.data.shape)

            #예측할 타겟 결정 : 해당 문자열이 포함되면 그것이 타겟이 된다.
            self.target = list(pd.Series(list(self.data.columns))[
                                        pd.Series(list(self.data.columns)).str.contains(pat=target, case=False)])[0]
            print("[Target column name] {}".format(self.target))

            # 조건 1 : Feature importance 출력값을 사용하여 특징 중요도가 높은 순서대로 모델 학습에 사용
            self._imppath = "{}/Ensemble/{}/{}/{}/IMP_Outdoor_{}_Indoor_{}.csv".format(self.SAVE_PATH, self.imp_method, self.folder_name, out_unit, out_unit, indv)
            self._impdata = pd.read_csv(self._imppath, index_col='Unnamed: 0')
            self._impdata= self._impdata.transpose() #결과값은 가로로 되어있으므로 전치
            #Feature importance를 고려한다면
            self.features_all = list(self._impdata.sort_values(by=0, ascending=False).index)
            # 이 함수는 NumOfFeatures 개수를 입력값으로 받는다.
            # NumOfFeatures=5라는 것은 상위 5개의 컬럼을 모델 학습에 사용한다는 의미이다.
            self.features = self.features_all[:NumOfFeatures] # 중요도가 큰 것 NumOfFeatures개만 사용한다.
            print("[Selected Features] {} - {}".format(len(self.features), self.features))

            # 조건 2 :Feature importance 고려안함 : 타겟을 제외한 나머지는 독립변수(현재는 주석 처리)
            # self.features = list(self.data.columns.difference([self.target]))

            # 필요한 컬럼만을 넣어서 데이터 셋을 만든다
            df = self.data[self.features]

            # 적용한 조건 확인용
            df.to_csv("{}/After_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))

            # 주기성 적용 : 시간은 연속적이며 일정한 주기를 가지고 있는 데이터이다.
            # 2*pi/x에서
            # month : x=12 / week : x=52
            # hour : x=24 / minute : x=60
            df.index = pd.to_datetime(df.index)
            df.loc[:, 'week_sin'] = np.round(np.sin((df.index.weekday() - 1) * (2. * np.pi / 12)), decimals=2)
            df.loc[:, 'week_cos'] = np.round(np.cos((df.index.weekday() - 1) * (2. * np.pi / 12)), decimals=2)
            df.loc[:, 'hour_sin'] = np.round(np.sin(df.index.hour * (2. * np.pi / 24)), decimals=2)
            df.loc[:, 'hour_cos'] = np.round(np.cos(df.index.hour * (2. * np.pi / 24)), decimals=2)
            df.loc[:, self.target] = self.data.loc[:, self.target].copy()
            df.loc[:, 'inputy'] = self.data.loc[:, self.target].copy()

            #원본 데이터 중에서 TrainSet과 TestSet을 분리한다.
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
            df_train.to_csv("{}/BldgRawData_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv), sep=',', float_format='%.2f')
            df_test.to_csv("{}/BldgRawData_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv), sep=',', float_format='%.2f')

            # 시계열을 제거
            df_train = df_train.reset_index().drop(self.TIME, 1) # TST 이전
            df_test = df_test.reset_index().drop(self.TIME, 1) # TST 이후

            #x_col: 최종적으로 학습에 들어가는 Feature list
            # 시계열 인덱스는 제거되었다. 대신 주기성이 포함되어 있으므로 시간의 주기성을 고려하였음.
            x_col = list(df_train.columns)
            x_col.remove(self.target)

            # self.X_train, self.X_test, self.y_train, self.y_test 를 전역변수로 사용하였다는 것은 많은 의미를 가지고 있다.
            # 만약 오토 인코더를 적용해야 한다면, 이 변수들을 인코딩 된 값으로 갱신해줘야 하기 때문이다.
            # 알고리즘 내에서 코딩 길이를 줄이려면 이 부분을 잘 만져주어야 한다.
            self.X_train = df_train[x_col].values.copy() # X_train : TST 데이터 이전의 데이터
            y_train = df_train[self.target].values.copy().reshape(-1, 1) # y_train : TST 이전의 타겟 값
            self.X_test = df_test[x_col].values.copy() # X_test : TST 이후의 데이터
            y_test = df_test[self.target].values.copy().reshape(-1, 1) #y_test : TST 이후의 타겟값
            print("[Train and Test dataset] X_train : {} - X_test : {} - y_train : {} - y_test : {}"
                  .format(self.X_train.shape, self.X_test.shape, y_train.shape, y_test.shape))

            # Parameter for normalizing
            mean = []
            std = []
            param_col = []
            for _s in range(self.X_train.shape[1] - 4):
                if self.X_train[:, _s].std() <= 0:
                    print(_s)
                else:
                    temp_mean = self.X_train[:, _s].mean()
                    temp_std = self.X_train[:, _s].std()
                    self.X_train[:, _s] = (self.X_train[:, _s] - temp_mean) / temp_std
                    self.X_test[:, _s] = (self.X_test[:, _s] - temp_mean) / temp_std
                param_col.append(x_col[_s])
                mean.append(temp_mean)
                std.append(temp_std)

            # z-score transform y
            self.y_mean = y_train.mean()
            self.y_std = y_train.std()
            print('y_std : {} - y_mean : {}'.format(round(self.y_std, 3), round(self.y_mean, 3)))

            self.y_train = (y_train - self.y_mean) / self.y_std
            self.y_test = (y_test - self.y_mean) / self.y_std
            param_col.append(self.target)

            mean.append(self.y_mean)
            std.append(self.y_std)
            param = [mean, std]
            print("[param check] mean : {} ".format(mean))
            print("[param check] std : {}".format(std))
            norm_param = pd.DataFrame(param, index=['mean', 'std'], columns=param_col)
            norm_param.to_csv("{}/Normalize_parameter_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))

            # 입력 데이터를 저장
            pd.DataFrame(self.X_train).to_csv("{}/Xtrain_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))
            pd.DataFrame(self.y_train).to_csv("{}/ytrain_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))
            pd.DataFrame(self.X_test).to_csv("{}/Xtest_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))
            pd.DataFrame(self.y_test).to_csv("{}/ytrain_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, indv))

            # 훈련샘플 만들어졌는지 데이터 확인용
            # x, y = self.generate_train_samples(x=self.X_train, y=self.y_train,
            #                                    input_seq_len=self.INPUT_SEQ_LEN,
            #                                    output_seq_len=self.OUTPUT_SEQ_LEN,
            #                                    batch_size=self.BATCH_SIZE)

            # 훈련 셋이 만들어지는 단계이다. 이 단계에서는 self.X_train, self.X_test 값을 사용하기 때문에
            # 입력되는 값이 어떤 처리 과정을 거쳐서 만들어진 입력값인지 분명히 할 필요가 있다.(오토인코더)
            self.test_x, self.test_y = self.generate_test_samples(x=self.X_test, y=self.y_test,
                                                        input_seq_len=self.INPUT_SEQ_LEN,
                                                        output_seq_len=self.OUTPUT_SEQ_LEN)

            # Train 데이터 사용하여 테스트용으로 한번더 할 때 사용
            self.train_x, self.train_y = self.generate_test_samples(x=self.X_train, y=self.y_train,
                                                          input_seq_len=self.INPUT_SEQ_LEN,
                                                          output_seq_len=self.OUTPUT_SEQ_LEN)

            if self.method in ["Seq2seq", "seq2seq"]:
                self.method = "Seq2seq"
                #Normal Seq2seq Model : 시퀀스-투-시퀀스 모델 기초형태는 그래프를 그리고 테스트를 하면 종료가된다.
                self.FEED_PREVIOUS = False
                rnn_model = self.build_graph(INPUT_DIM=int(self.X_train.shape[1]), OUTPUT_DIM=int(self.y_train.shape[1]))
                self.TRAIN_PROCESS(model=rnn_model, save=save, out_unit=out_unit, ind_unit=indv)

                # Test setting value
                self.FEED_PREVIOUS = True
                self.DROPOUT = 0
                test_model = self.build_graph(INPUT_DIM=int(self.X_test.shape[1]), OUTPUT_DIM=int(self.y_test.shape[1]))
                self.TEST_PROCESS(model=test_model, save=save, out_unit=out_unit, ind_unit=indv)

            elif self.method in ["AutoEncoder", "autoencoder"]:
                self.method = "AutoEncoder"
                #AutoEncoder Model : 오토인코더 모델은 오토인코더 모델을 만들고 그 모델을 사용하여 테스트한다.
                # 그 후에 오토인코더로 입력 데이터를 만들어 진 것을 가지고 Seq2seq 모델을 돌린다.
                self.FEED_PREVIOUS = False
                AE_model = self.build_Autoencoder(INPUT_DIM=int(self.X_train.shape[1]))
                self.TRAIN_PROCESS(model=AE_model, save=save, out_unit=out_unit, ind_unit=indv)

                # Autoencoder Test setting value
                self.FEED_PREVIOUS = True
                self.DROPOUT = 0
                AE_test_model = self.build_Autoencoder(INPUT_DIM=int(self.X_test.shape[1]))
                # 오토인코더를 달았을 경우에는 테스트 프로세스에서 오토인코더를 만들고 끝나고 seq2seq를 시작할 준비를 해준다.
                self.TEST_PROCESS(model=AE_test_model, save=save, out_unit=out_unit, ind_unit=indv)

                # 데이터 업데이트 : 갱신된 데이터셋을 입력하고 그것을 출력
                self.train_x, self.train_y = self.generate_test_samples(self.X_train, self.y_train, self.INPUT_SEQ_LEN, self.OUTPUT_SEQ_LEN)
                self.test_x, self.test_y = self.generate_test_samples(self.X_test, self.y_test, self.INPUT_SEQ_LEN, self.OUTPUT_SEQ_LEN)

                #Normal Seq2seq Model
                self.method = "Seq2seq" # 오토인코더가 완료된 후 TRAIN_PROCESS, TEST_PROCESS에 사용된다.
                self.FEED_PREVIOUS = False
                self.DROPOUT = 0.5
                rnn_model = self.build_graph(INPUT_DIM=int(self.X_train.shape[1]), OUTPUT_DIM=int(self.y_train.shape[1]))
                self.TRAIN_PROCESS(model=rnn_model, save=save, out_unit=out_unit, ind_unit=indv)

                # Test setting value
                self.FEED_PREVIOUS = True
                self.DROPOUT = 0
                test_model = self.build_graph(INPUT_DIM=int(self.X_test.shape[1]), OUTPUT_DIM=int(self.y_test.shape[1]))
                self.TEST_PROCESS(model=test_model, save=save, out_unit=out_unit, ind_unit=indv)
                self.method = "AutoEncoder" # 오토인코더 저장위치를 정확히 지정하기 위해서 초기화시켜준다.

            elif self.method in ["Attention", "attention"]:
                self.method = "Attention"
                #Normal Seq2seq Model : 시퀀스-투-시퀀스 모델 기초형태는 그래프를 그리고 테스트를 하면 종료가된다.
                self.FEED_PREVIOUS = False
                rnn_model = self.build_graph_With_Attention(INPUT_DIM=int(self.X_train.shape[1]), OUTPUT_DIM=int(self.y_train.shape[1]))
                self.TRAIN_PROCESS(model=rnn_model, save=save, out_unit=out_unit, ind_unit=indv)

                # Test setting value
                self.FEED_PREVIOUS = True
                self.DROPOUT = 0
                test_model = self.build_graph_With_Attention(INPUT_DIM=int(self.X_test.shape[1]), OUTPUT_DIM=int(self.y_test.shape[1]))
                self.TEST_PROCESS(model=test_model, save=save, out_unit=out_unit, ind_unit=indv)

    def TRAIN_PROCESS(self, model, out_unit, save, ind_unit):
        init = tf.compat.v1.global_variables_initializer()
        loss_fun = []
        if self.method in ["Seq2seq", "seq2seq"]:
            self.method = "Seq2seq"
            print("Seq2seq Start! (Train Process)")
            with tf.compat.v1.Session() as sess:
                sess.run(init)  # 초기화
                for oo_ in range(self.TOTAL_ITERATION):
                    # X_train : TST 이전의 데이터
                    # y_train : TST 이전의 타겟
                    # print("[Model Training Sample] X_train : {} - y_train : {}".format(X_train.shape, y_train.shape))
                    batch_input, batch_output = self.generate_train_samples(x=self.X_train,
                                                                            y=self.y_train,
                                                                            input_seq_len=self.INPUT_SEQ_LEN,
                                                                            output_seq_len=self.OUTPUT_SEQ_LEN,
                                                                            batch_size=self.BATCH_SIZE)
                    # print("[Batch input] batch_input : {} - batch_output : {}".format(batch_input.shape, batch_output.shape))
                    # 배치 사이즈 만큼 잘라서 모델에 입력한다.
                    # rnn model의 enc_inp의 t번째에 입력 배치값을 순서대로 넣는다.
                    feed_dict = {model['enc_inp'][t]: batch_input[:, t] for t in range(self.INPUT_SEQ_LEN)}
                    feed_dict.update(
                        {model['target_seq'][t]: batch_output[:, t] for t in range(self.OUTPUT_SEQ_LEN)})

                    # rnn model의 targer_seq에 출려 배치값을 순서대로 업데이트한다.
                    # update 함수는
                    # print("{} - {}".format(rnn_model['train_op'], rnn_model['loss']))
                    _, loss_t = sess.run([model['train_op'], model['loss']], feed_dict)
                    if oo_ % 100 == 0:
                        print("[step: {}] Loss : {} - Time : {}".format(oo_, round(float(loss_t), 4),
                                                                        datetime.datetime.now().strftime(
                                                                            '%Y-%m-%d %H:%M:%S')))
                        loss_fun.append([oo_, loss_t])
                    if loss_t < 0.001:
                        print("[{}] Loss cut : {}".format(oo_, loss_t))
                        break
                    if pd.isnull(loss_t) == True:
                        print("Loss is null")
                        break

                temp_saver = model['saver']()
                save_path = temp_saver.save(sess, "{}/Model_Outdoor_{}_Indoor_{}".format(save, out_unit, ind_unit))
            print("[Checkpoint saved at] {}".format(save))
            loss_result = pd.DataFrame(loss_fun, columns=['iteration', 'loss'])
            loss_result.to_csv('{}/lossResult_Outdoor_{}_Indoor_{}.csv'.format(save, out_unit, ind_unit))
            print("Seq2seq Completed! (Train Process)")

        elif self.method in ["AutoEncoder", "autoencoder"]:
            self.method = "AutoEncoder"
            print("AutoEncoder Start! (Train Process)")
            #오토인코더를 만든 다음에 그 데이터를 Seq2seq로 집어넣는 개념이다.
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                for oo_ in range(self.TOTAL_ITERATION):
                    batch_input, _ = self.generate_train_samples(x=self.X_train,
                                                                 y=self.y_train,
                                                                 input_seq_len=self.INPUT_SEQ_LEN,
                                                                 output_seq_len=self.OUTPUT_SEQ_LEN,
                                                                 batch_size=self.BATCH_SIZE)
                    feed_dict = {model['enc_inp'][t]: batch_input[:, t] for t in range(self.INPUT_SEQ_LEN)}
                    _, loss_t = sess.run([model['train_op'], model['loss']], feed_dict)

                    if oo_ % 100 == 0:
                        print("[step: {}] Loss : {} - Time : {}".format(oo_, round(float(loss_t), 4),
                                                                        datetime.datetime.now().strftime(
                                                                            '%Y-%m-%d %H:%M:%S')))
                        loss_fun.append([oo_, loss_t])
                    if loss_t < 0.001:
                        print("[{}] Loss cut : {}".format(oo_, loss_t))
                        break
                    if pd.isnull(loss_t) == True:
                        print("Loss is null")
                        break
                temp_saver = model['saver']()
                save_path = temp_saver.save(sess, "{}/Model_AutoEncoder_Outdoor_{}_Indoor_{}".format(save, out_unit, ind_unit))
                loss_result = pd.DataFrame(loss_fun, columns=['iteration', 'loss'])
                loss_result.to_csv('{}/loss_AutoEncoder_Outdoor_{}_Indoor_{}.csv'.format(save, out_unit, ind_unit))
                print("AutoEncoder Completed! (Train Process)")


        elif self.method in ["Attention", "attention"]:
            self.method = "Attention"
            print("Attention Start! (Train Process)")
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                for oo_ in range(self.TOTAL_ITERATION):
                    batch_input, batch_output = self.generate_train_samples(x=self.X_train,
                                                                            y=self.y_train,
                                                                            input_seq_len=self.INPUT_SEQ_LEN,
                                                                            output_seq_len=self.OUTPUT_SEQ_LEN,
                                                                            batch_size=self.BATCH_SIZE)
                    feed_dict = {model['enc_inp'][t]: batch_input[:, t] for t in range(self.INPUT_SEQ_LEN)}
                    feed_dict.update({model['target_seq'][t]: batch_output[:, t] for t in range(self.OUTPUT_SEQ_LEN)})
                    attn_outputs = sess.run(model['attn_outputs'], feed_dict)
                    _, loss_t = sess.run([model['train_op'], model['loss']], feed_dict)

                    if oo_ % 100 == 0:
                        print("[step: {}] Loss : {} - Time : {}".format(oo_, round(float(loss_t), 4),
                                                                        datetime.datetime.now().strftime(
                                                                            '%Y-%m-%d %H:%M:%S')))
                        loss_fun.append([oo_, loss_t])
                        # print('attn_outputs shape: {}'.format(attn_outputs.shape))
                    if loss_t < 0.001:
                        print("[{}] Loss cut : {}".format(oo_, loss_t))
                        break
                    if pd.isnull(loss_t) == True:
                        print("Loss is null")
                        break

                temp_saver = model['saver']()
                save_path = temp_saver.save(sess, "{}/Model_Outdoor_{}_Indoor_{}".format(save, out_unit, ind_unit))
            print("[Checkpoint saved at] {}".format(save))
            loss_result = pd.DataFrame(loss_fun, columns=['iteration', 'loss'])
            loss_result.to_csv('{}/lossResult_Outdoor_{}_Indoor_{}.csv'.format(save, out_unit, ind_unit))
            print("Seq2seq With Attention Completed! (Train Process)")

    def TEST_PROCESS(self, model, save, out_unit, ind_unit):
        init = tf.compat.v1.global_variables_initializer()
        if self.method in ["Seq2seq"]:
            print("Seq2seq start! (Test Process)")
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                saver = model['saver']().restore(sess, "{}/Model_Outdoor_{}_Indoor_{}".format(save, out_unit, ind_unit))
                feed_dict = {model['enc_inp'][t]: self.test_x[:, t, :] for t in
                             range(self.INPUT_SEQ_LEN)}  # batch prediction
                feed_dict.update(
                    {model['target_seq'][t]: np.zeros([self.test_x.shape[0], self.y_test.shape[1]],
                                                           dtype=np.float32) for t in range(self.OUTPUT_SEQ_LEN)})
                final_preds = sess.run(model['reshaped_outputs'], feed_dict=feed_dict)  # type : list
                final_preds = [np.expand_dims(pred, axis=1) for pred in final_preds]

                final_preds2 = np.array(final_preds)
                final_preds2 = final_preds2.reshape(final_preds2.shape[0],
                                                    final_preds2.shape[1] * final_preds2.shape[2])  # (8620, 10)
                np.savetxt("{}/testresult_y2_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit), final_preds2,
                           fmt='%.2f', delimiter=',')
                final_preds = np.concatenate(final_preds, axis=1)  # (8620, 10, 1)
                final_preds3 = np.array(final_preds)  # (8620, 10, 1)
                final_preds3 = final_preds3.reshape(final_preds3.shape[0],
                                                    final_preds3.shape[1] * final_preds3.shape[2])  # (8620, 10)
                np.savetxt("{}/testresult_y3_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit), final_preds3,
                           fmt='%.2f', delimiter=',')

            # remove duplicate hours and concatenate into one long array
            test_y_expand = np.concatenate(
                [self.test_y[_].reshape(-1) for _ in range(0, self.test_y.shape[0], self.OUTPUT_SEQ_LEN)], axis=0)
            final_preds_expand = np.concatenate(
                [final_preds[_].reshape(-1) for _ in range(0, final_preds.shape[0], self.OUTPUT_SEQ_LEN)], axis=0)
            # np.savetxt("{}/final_preds_expand_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit),
            #            final_preds_expand, fmt='%.2f', delimiter=',')

            print('y_std : {} - y_mean : {}'.format(round(self.y_std, 3), round(self.y_mean, 3)))
            final_preds_expand2 = final_preds_expand * self.y_std + self.y_mean  # 정규화된 것 복구
            test_y_expand2 = test_y_expand * self.y_std + self.y_mean
            Y = pd.DataFrame(test_y_expand2)
            Y_pred = pd.DataFrame(final_preds_expand2)
            Y_result = pd.concat([Y, Y_pred], ignore_index=True, axis=1)
            Y_result.columns = ['Test', 'Prediction']
            Y_result.to_csv("{}/RNN_Test_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            print("[Y_result] Shape : {} - Type {}".format(Y_result.shape, type(Y_result)))

            # 성능지표
            Y_result.loc[Y_result['Test'] < 0.1, 'Prediction'] = 0
            Y_result['Test'] = Y_result['Test'].apply(lambda x: np.nan if x < 0.1 else x)
            Y_result['Prediction'] = Y_result['Prediction'].apply(lambda x: 0 if x < 0.1 else x)
            Y_result = Y_result.dropna()
            Y_result = Y_result.reset_index(drop=True)

            for c in Y_result.index:
                if Y_result['Prediction'][c] < 0:
                    Y_result['Prediction'][c] = 0

            if Y_result['Prediction'].isnull().sum() == 0:
                Test_RMSE = np.sqrt(np.mean((Y_result['Prediction'] - Y_result['Test']) ** 2))
                Test_cvRMSE = Test_RMSE * 100 / np.mean(Y_result['Test'])
            else:
                Test_cvRMSE = 1000

            print("[Test_cvRMSE] Accuracy(cvRMSE): {} %".format(Test_cvRMSE))
            df_acc = pd.DataFrame({"Accuracy(CvRMSE)": [round(float(Test_cvRMSE), 3)]})
            df_acc.to_csv("{}/Acc_y3_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            print("Seq2seq Completed! (Test Process)")

        elif self.method in ["AutoEncoder"]:
            print("AutoEncoder Start! (Test Process)")
            # 오토인코더 test
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                saver = model['saver']().restore(sess,"{}/Model_AutoEncoder_Outdoor_{}_Indoor_{}".format(save, out_unit, ind_unit))
                feed_dict = {model['enc_inp'][t]: self.test_x[:, t, :] for t in
                             range(self.INPUT_SEQ_LEN)}
                final_preds = sess.run(model['reshaped_output'], feed_dict)
                final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
                final_preds = np.concatenate(final_preds, axis=1)

                enc_preds = sess.run(model['enc_output'], feed_dict)
                enc_preds = [np.expand_dims(pred, 1) for pred in enc_preds]
                enc_preds = np.concatenate(enc_preds, axis=1)

            real_expand = np.concatenate([self.test_x[_] for _ in range(0, self.test_x.shape[0], self.INPUT_SEQ_LEN)], axis=0)
            final_preds_expand = np.concatenate([final_preds[_] for _ in range(0, final_preds.shape[0], self.INPUT_SEQ_LEN)], axis=0)

            #X_test가 갱신된다. 향후에 test에 사용된다.
            self.X_test = np.concatenate([enc_preds[i] for i in range(0, enc_preds.shape[0], self.INPUT_SEQ_LEN)], axis=0)

            pd.DataFrame(real_expand).to_csv("{}/test_X_AutoEncoder_Real_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            pd.DataFrame(final_preds_expand).to_csv("{}/test_X_AutoEncoder_Prediction_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            pd.DataFrame(self.X_test).to_csv("{}/X_test_AutoEncoder_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))

            # 오토인코더 Train데이터를 사용하여 Train 테스트 이 출력값이 시퀀스-투-시퀀스 입력값으로 사용된다.
            AE_test_model = self.build_Autoencoder(INPUT_DIM=int(self.X_train.shape[1]))
            init = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                saver = model['saver']().restore(sess, "{}/Model_AutoEncoder_Outdoor_{}_Indoor_{}".format(save, out_unit, ind_unit))
                feed_dict = {AE_test_model['enc_inp'][t]: self.train_x[:, t, :] for t in
                             range(self.INPUT_SEQ_LEN)}  # batch prediction

                final_preds = sess.run(AE_test_model['reshaped_output'], feed_dict)
                final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
                final_preds = np.concatenate(final_preds, axis=1)

                enc_preds = sess.run(AE_test_model['enc_output'], feed_dict)
                enc_preds = [np.expand_dims(pred, 1) for pred in enc_preds]
                enc_preds = np.concatenate(enc_preds, axis=1)
            real_expand = np.concatenate([self.train_x[i] for i in range(0, self.train_x.shape[0], self.INPUT_SEQ_LEN)], axis=0)
            final_preds_expand = np.concatenate([final_preds[i] for i in range(0, final_preds.shape[0], self.INPUT_SEQ_LEN)],
                                                axis=0)
            # X_train이 갱신된다. 향후에 Train에 사용된다.
            self.X_train = np.concatenate([enc_preds[i] for i in range(0, enc_preds.shape[0], self.INPUT_SEQ_LEN)], axis=0)

            pd.DataFrame(real_expand).to_csv("{}/train_X_AutoEncoder_Real_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            pd.DataFrame(final_preds_expand).to_csv("{}/train_X_AutoEncoder_Prediction_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            pd.DataFrame(self.X_train).to_csv("{}/X_train_AutoEncoder_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))

            self.y_train = self.y_train[:self.X_train.shape[0]]
            self.y_test = self.y_test[:self.X_test.shape[0]]
            # 이 과정까지 마무리되면 X_train, X_test, y_train, y_test 까지 만들어졌다.
            # 이제 다시 시퀀스-투-시퀀스를 돌려주면된다.
            print("AutoEncoder Completed! (Test Process)")

        elif self.method in ["Attention"]:
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                saver = model['saver']().restore(sess, "{}/Model_Outdoor_{}_Indoor_{}".format(save, out_unit, ind_unit))
                feed_dict = {model['enc_inp'][t]: self.test_x[:, t, :] for t in range(self.INPUT_SEQ_LEN)}  # batch prediction
                feed_dict.update(
                    {model['target_seq'][t]: np.zeros([self.test_x.shape[0], self.y_test.shape[1]], dtype=np.float32) for t in range(self.OUTPUT_SEQ_LEN)})
                final_preds = sess.run(model['reshaped_outputs'], feed_dict)
                final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
                final_preds2 = np.array(final_preds)
                final_preds2 = final_preds2.reshape(final_preds2.shape[0],
                                                    final_preds2.shape[1] * final_preds2.shape[2])
                np.savetxt("{}/testresult_y2_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit), final_preds2,
                           fmt='%.2f', delimiter=',')
                final_preds = np.concatenate(final_preds, axis=1)
                final_preds3 = np.array(final_preds)
                final_preds3 = final_preds3.reshape(final_preds3.shape[0],
                                                    final_preds3.shape[1] * final_preds3.shape[2])
                np.savetxt("{}/testresult_y3_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit), final_preds3,
                           fmt='%.2f', delimiter=',')

            ## remove duplicate hours and concatenate into one long array
            test_y_expand = np.concatenate([self.test_y[i].reshape(-1) for i in range(0, self.test_y.shape[0], self.OUTPUT_SEQ_LEN)],
                                           axis=0)
            final_preds_expand = np.concatenate(
                [final_preds[i].reshape(-1) for i in range(0, final_preds.shape[0], self.OUTPUT_SEQ_LEN)], axis=0)

            final_preds_expand2 = final_preds_expand * self.y_std + self.y_mean
            test_y_expand2 = test_y_expand * self.y_std + self.y_mean

            Y_pred = pd.DataFrame(final_preds_expand2)
            Y = pd.DataFrame(test_y_expand2)
            Y_result = pd.concat([Y, Y_pred], ignore_index=True, axis=1)
            Y_result.columns = ['Test', 'Prediction']
            Y_result.to_csv("{}/RNN_Test_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            print("[Y_result] Shape : {} - Type {}".format(Y_result.shape, type(Y_result)))

            # 성능지표
            Y_result.loc[Y_result['Test'] < 0.1, 'Prediction'] = 0
            Y_result['Test'] = Y_result['Test'].apply(lambda x: np.nan if x < 0.1 else x)
            Y_result['Prediction'] = Y_result['Prediction'].apply(lambda x: 0 if x < 0.1 else x)
            Y_result = Y_result.dropna()
            Y_result = Y_result.reset_index(drop=True)

            for c in Y_result.index:
                if Y_result['Prediction'][c] < 0:
                    Y_result['Prediction'][c] = 0

            if Y_result['Prediction'].isnull().sum() == 0:
                Test_RMSE = np.sqrt(np.mean((Y_result['Prediction'] - Y_result['Test']) ** 2))
                Test_cvRMSE = Test_RMSE * 100 / np.mean(Y_result['Test'])
            else:
                Test_cvRMSE = 1000

            print("[Test_cvRMSE] Accuracy(cvRMSE): {} %".format(Test_cvRMSE))
            df_acc = pd.DataFrame({"Accuracy(CvRMSE)": [round(float(Test_cvRMSE), 3)]})
            df_acc.to_csv("{}/Acc_y3_Outdoor_{}_Indoor_{}.csv".format(save, out_unit, ind_unit))
            print("Seq2seq With Attention Completed! (Test Process)")

#Seq2Seq Hyperparameter
INPUT_SEQ_LEN = 10
OUTPUT_SEQ_LEN = 10
BATCH_SIZE = 500
LEARNING_RATE = 0.001

HIDDEN_DIM = 128
NUM_STACK_LAYERS = 4
DROPOUT = 0.5

KEEP_RATE = 0.7
GRADIENT_CLIPPING = 2.5
FEED_PREVIOUS = False

#AutoEncoder HyperParameter
LAYERS = 3
PERCENTAGE = 0.7
LAMBDA_L2_REG = 0.003

TOTAL_ITERATION = 5000

TIME = 'updated_time' # 시계열 컬럼 이름
start ='2021-07-01' #데이터 시작시간
end = '2021-09-30' #데이터 끝시간
test_start_time = '2021-09-26' #테스트 데이터 시작 시간

#예측하고자하는 값
TARGET = "room_temp"

# 클래스 안에서 사용한 공통 언어
SIGNAL = 'indoor_power'
meterValue = 'value'
TspValue = 'set_temp'
# TzValue =  'room_temp'
ToaValue = 'outdoor_temp'

METHOD = "AutoEncoder" #"Seq2seq" ,"Attention", "AutoEncoder"
# 특징중요도 메소드
IMP_METHOD = "Randomforest" #"Randomforest","Adaboosting","Gradientboosting"

NumOfFeatures = 5

DML = DEEPMODEL(time=TIME, LEARNING_RATE=LEARNING_RATE, LAMBDA_L2_REG=LAMBDA_L2_REG, BATCH_SIZE=BATCH_SIZE,
                INPUT_SEQ_LEN=INPUT_SEQ_LEN, OUTPUT_SEQ_LEN=OUTPUT_SEQ_LEN, NUM_STACK_LAYERS=NUM_STACK_LAYERS,
                HIDDEN_DIM=HIDDEN_DIM, GRADIENT_CLIPPING=GRADIENT_CLIPPING, KEEP_RATE=KEEP_RATE,
                TOTAL_ITERATION=TOTAL_ITERATION, FEED_PREVIOUS=FEED_PREVIOUS, DROPOUT=DROPOUT, LAYERS=LAYERS, PERCENTAGE=PERCENTAGE,
                start=start, end=end, test_start_time=test_start_time)

for outdv in [909]:#, 910, 921, 920, 919, 917, 918, 911]:
    DML.DEEP_PROCESSING(out_unit=outdv, signal=SIGNAL, meterValue=meterValue,
                        TspValue=TspValue, ToaValue=ToaValue,
                        target=TARGET, method=METHOD, IMP_METHOD=IMP_METHOD,
                        NumOfFeatures=NumOfFeatures)