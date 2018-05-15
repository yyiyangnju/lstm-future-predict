# coding=gbk
'''
Created on 2017年2月20日

@author: Lu.yipiao
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf



def calc_rsq(y, yhat):
    ret = 1 - ((y - yhat) ** 2).mean() / y.var()
    return ret

# 获取训练集

def get_train_data(input_len=20, output_len=1, train_begin=0,  train_mode=True):
    train_end = (TOTAL_LEN // 4) * 3
    if train_mode:
        data_train = data[train_begin: train_end]
    else:
        data_train = data[train_end: ]
    mean, std = np.mean(data_train, axis=0), np.std(data_train, axis=0)
    data_train_norm = (data_train - mean) / std  # 标准化

    train_x, train_y = [], []  # 训练集
    iter_end = len(data_train_norm) - max(input_len, output_len)
    X, Y = data_train_norm[:, : y_col_num], data_train_norm[:, y_col_num]
    for i in range(iter_end):
        x = X[i: i + input_len]
        y = Y[i: i + output_len, np.newaxis]
        train_x.append(x.copy())
        train_y.append(y.copy())

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return mean, std, train_x, train_y


def get_data(x, y, train_ratio=0.7):
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    x_train, x_test = x[: train_len], x[train_len: ]
    y_train, y_test = y[: train_len], y[train_len: ]
    return x_train, x_test, y_train, y_test


def rolling_standardize(arr, window=5):
    """
    Assume first axis is rolling axis.

    """
    df = pd.DataFrame(arr)
    if window == 0:
        mean, std = arr.mean(axis=0), arr.std(axis=0)
    else:
        roll = df.rolling(window=window, axis=0)
        mean, std = roll.mean(), roll.std()
        mean, std = mean.values, std.values
    df_std = (df - mean) / std
    df_std = df_std.iloc[window: ]
    arr_std = df_std.values
    return arr_std, mean[window: ], std[window: ]


def ts2sample(ts, window=10, stride=1):
    """

    Parameters
    ------
    ts : np.ndarray
        of shape [ts_len, ...]

    Returns
    -------
    res : np.ndarray
        of shape [n_samples, window, ...]

    """
    samples = []
    ts_len = len(ts)
    for start in range(0, ts_len - window, stride):
        end = start + window
        sample = ts[start: end]
        samples.append(sample[np.newaxis])
    res = np.vstack(samples)
    return res

# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    """

    :param X: inpu
    :return:prediction final states of units
    only on layer
    """
    X_shape = X.get_shape()
    batch_size = BATCH_SIZE
    w_in = weights['in']
    b_in = biases['in']
    w_in = tf.tile(tf.expand_dims(w_in, 0), [batch_size, 1, 1])
    # input = tf.reshape(X, [batch_size, -1])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入 TODO
    input = X
    input_rnn = tf.matmul(input, w_in) + b_in
    # input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入 TODO
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [batch_size, -1])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm(x_train, y_train,
               batch_size=77,
               input_len=15, output_len=1,
               train_begin=2000, train_end=5800,
               n_epoch=1000):
    X = tf.placeholder(tf.float32, shape=[batch_size, input_len, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[batch_size, output_len, OUTPUT_SIZE])
    # _, _, train_x, train_y = get_train_data(input_len, output_len, train_begin, train_end)
    pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)

    # Create model saver and Set dir
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
    # get latest checkpoint file name
    module_file = tf.train.latest_checkpoint(CKPT_DIR)

    with tf.Session() as sess:
        if module_file is not None:
            # restore latestest checkpoint
            saver.restore(sess, module_file)
        else:
            # initialize global variables
            sess.run(tf.global_variables_initializer())

        # Continue to train from latest checkpoint
        for epoch in range(n_epoch):
            train_len = len(x_train)
            for start, end in zip(range(0,          train_len, batch_size),
                                  range(batch_size, train_len, batch_size)):
                _, loss_ = sess.run(fetches=[train_op, loss],
                                    feed_dict={X: x_train[start: end],
                                               Y: y_train[start: end]})

            print(epoch, loss_)

            if epoch % 200 == 0:
                saved_file = saver.save(sess, CKPT_DIR + '\stock2.model', global_step=epoch)
                print("保存模型：", saved_file)


# ————————————————预测模型————————————————————
def prediction(x_test, y_test, input_len=20):
    X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, input_len, INPUT_SIZE])
    # mean, std, test_x, test_y = get_train_data(input_len=input_len, output_len=OUTPUT_LEN, train_mode=False)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint(CKPT_DIR)
        if module_file is not None:
            saver.restore(sess, module_file)
            print("module_file resotred.")

        test_predict = []
        test_len = len(x_test)
        start, end = 0, 0
        for start, end in zip(range(0,          test_len, BATCH_SIZE),
                              range(BATCH_SIZE, test_len, BATCH_SIZE)):
        # for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: x_test[start: end]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)

    return test_predict



if __name__ == "__main__":
    # checkpoint dir
    # CKPT_DIR = 'checkpoints_no_rolling'
    CKPT_DIR = 'checkpoints_return'

    # 定义常量
    rnn_unit = 20  # hidden layer units
    OUTPUT_SIZE = 1
    INPUT_LEN = 15
    OUTPUT_LEN = 1
    BATCH_SIZE = 77
    lr = 0.0006  # 学习率

    # ——————————————————导入数据——————————————————————
    df = pd.read_csv('Archive/futures-historical-data/rb1505_300.csv',
                     usecols=[3, 4, 5, 6, 7, 6])
    df = df.reindex(columns=['open', 'high', 'low', 'close', 'volume'])

    shift_len = INPUT_LEN + 1
    df.loc[:, 'label'] = df['close'].shift(shift_len)  # -1
    df.loc[:, 'rtn'] = df['close'].pct_change(shift_len)
    df.loc[:, 'label'] = df['close'].pct_change(1).shift(-shift_len)  # -1


    data = df.dropna()
    data = data.values

    INPUT_SIZE = data.shape[1] - 1  # 与之后读入的df有关

    # split x y and train, test
    data_x = data[:, :-1]
    data_y = data[:, [-1]]
    x_train, x_test, y_train, y_test = get_data(data_x, data_y, train_ratio=0.7)

    # rolling standardization
    ROLL_WINDOW = 50
    print(x_train.shape, y_train.shape)
    x_train, _, _ = rolling_standardize(x_train, window=ROLL_WINDOW)
    x_test, _, _ = rolling_standardize(x_test, window=ROLL_WINDOW)
    # y_train, _, _ = rolling_standardize(y_train, window=ROLL_WINDOW)
    # y_test_std, mean_y_test, std_y_test = rolling_standardize(y_test, window=ROLL_WINDOW)
    y_test_std, mean_y_test, std_y_test = y_test, 0, 1
    print(x_train.shape, y_train.shape)

    # INPUT_LEN > OUTPUT_LEN
    x_train = ts2sample(x_train, INPUT_LEN, stride=1)
    x_test = ts2sample(x_test, INPUT_LEN, stride=1)
    y_train = ts2sample(y_train, INPUT_LEN, stride=1)
    y_test_std = ts2sample(y_test_std, INPUT_LEN, stride=1)
    y_train = y_train[:, [-OUTPUT_LEN]]
    y_test_std = y_test_std[:, [-OUTPUT_LEN]]

    TOTAL_LEN = len(data)

    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置

    weights = {
        'in': tf.Variable(tf.random_normal([INPUT_SIZE, rnn_unit])),
        'out': tf.Variable(tf.random_normal([rnn_unit * INPUT_LEN, 1]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

    train_mode = 1
    if train_mode:
        train_lstm(x_train, y_train,
                   batch_size=BATCH_SIZE,
                   input_len=INPUT_LEN, output_len=1,
                   n_epoch=10000
                   )
    else:
        # y_pred = prediction(x_test, y_test_std, INPUT_LEN)
        y_pred = prediction(x_train, y_train, INPUT_LEN)
        y_pred = np.squeeze(y_pred)
        # y_test = np.squeeze(y_test)
        y_test = np.squeeze(y_train)
        n = len(y_pred)
        y_test = y_test[: n]

        # y_pred = y_pred * np.squeeze(std_y_test[: n]) + np.squeeze(mean_y_test[: n])
        y_pred = y_pred * std_y_test + mean_y_test
        def calc_return(x, window):
            return x[window: ] / x[: -window] - 1
        FORWARD_LEN = abs(shift_len)
        rtn = calc_return(y_test, FORWARD_LEN)
        rtn_pred = calc_return(y_pred, FORWARD_LEN)

        def show_rsq(y, yhat, title):
            rsq = calc_rsq(y, yhat)
            #print("rsq = {:.5f}".format(rsq))
            plt.figure()
            plt.scatter(y, yhat)
            plt.title("{:s} (rsq = {:.3f}%)".format(title, rsq*100))
            plt.show()
            #plt.savefig("{:s}.png".format(title))
            #plt.close()

        show_rsq(rtn, rtn_pred, "return_true v.s. return_pred")
        show_rsq(y_test, y_pred, "close_true v.s. close_pred")

        acc = np.average(np.abs(y_pred - y_test) / y_test)  # 偏差
