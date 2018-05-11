# coding=gbk
'''
Created on 2017年2月20日

@author: Lu.yipiao
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# 定义常量
rnn_unit = 10  # hidden layer units
INPUT_SIZE = 7
OUTPUT_SIZE = 1
OUTPUT_LEN = 1
lr = 0.0006  # 学习率

# ——————————————————导入数据——————————————————————
df = pd.read_csv('dataset/dataset_2.csv')  # 读入股票数据
# store = pd.HDFStore('train_test_dataset', mode='r')
# data = store['dataset']
# store.close()
# store = pd.HDFStore('data/train_test_dataset', mode='r')
# data = store['dataset']
# store.close()
data = df.iloc[:, 2:10].values  # 取第3-10


# 获取训练集
def get_train_data(input_len=20, output_len=1, train_begin=0, train_end=5800):
    data_train = data[train_begin: train_end]
    data_train_norm = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化

    train_x, train_y = [], []  # 训练集
    y_col_num = 7
    iter_end = len(data_train_norm) - input_len - output_len
    X, Y = data_train_norm[:, : y_col_num], data_train_norm[:, y_col_num]
    for i in range(iter_end):
        x = X[i: i + input_len]
        y = Y[i + input_len + 1:
              i + input_len + 1 + output_len,
              np.newaxis]
        train_x.append(x.copy())
        train_y.append(y.copy())

    return train_x, train_y


# 获取测试集
def get_test_data(time_step=20, test_begin=5800):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :7]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, 7]).tolist())
    return mean, std, test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    """

    :param X: inpu
    :return:prediction final states of units
    only on layer
    """
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, INPUT_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入 TODO
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入 TODO
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)  # TODO forget bias
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm(batch_size=80, input_len=15, output_len=1, train_begin=2000, train_end=5800):
    X = tf.placeholder(tf.float32, shape=[None, input_len, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, output_len, OUTPUT_SIZE])
    train_x, train_y = get_train_data(input_len, output_len, train_begin, train_end)
    pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # Create model saver and Set dir
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    cp_dir = r'C:\Users\lenovo\Desktop\dissertation\datamining\tensorflow-program\rnn\stock_predict\checkpoint'
    # get latest checkpoint file name
    module_file = tf.train.latest_checkpoint(cp_dir)

    with tf.Session() as sess:
        # initialize global variables
        sess.run(tf.global_variables_initializer())

        # restore latestest checkpoint
        if module_file is not None:
            saver.restore(sess, module_file)

        # Continue to train from latest checkpoint
        for i in range(2000):
            train_len = len(train_x)
            for start, end in zip(range(0,          train_len, batch_size),
                                  range(batch_size, train_len, batch_size)):
                _, loss_ = sess.run(fetches=[train_op, loss],
                                    feed_dict={X: train_x[start: end],
                                               Y: train_y[start: end]})

            print(i, loss_)

            if i % 200 == 0:
                saved_file = saver.save(sess, cp_dir + '\stock2.model', global_step=i)
                print("保存模型：", saved_file)


train_lstm()


# ————————————————预测模型————————————————————
def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    # Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint()
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[7] + mean[7]
        test_predict = np.array(test_predict) * std[7] + mean[7]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


prediction()
