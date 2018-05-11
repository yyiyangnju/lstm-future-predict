from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Flatten
import numpy as np
import pandas as pd


df = pd.read_csv('dataset/dataset_2.csv')

time_step = 15
Epochs = 1000
BatchSize = 80
INPUT_DIM = 7  # determined by input data
OUTPUT_DIM = 1
time_step_out = 1

shift_len = -(time_step - 1 )
df.loc[:, 'label'] = df['label'].shift(shift_len)  # 由输入数据决定
data = df.dropna()
data = data.iloc[:, 2:10].values  # 取第3-10

TOTAL_LEN = len(data)

#建立网络

def get_train_data(input_len=20, output_len=1, train_begin=0,  train_mode=True):

    train_end = (TOTAL_LEN // 4) * 3
    if train_mode:
        data_train = data[train_begin: train_end]
    else:
        data_train = data[train_begin: ]
    mean, std = np.mean(data_train, axis=0), np.std(data_train, axis=0)
    data_train_norm = (data_train - mean) / std  # 标准化

    train_x, train_y = [], []  # 训练集
    y_col_num = data_train_norm.shape[1] - 1
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

model = Sequential()
model.add(SimpleRNN(units=10,
                    activation='sigmoid',
                    use_bias=True,
                    input_shape=(time_step, INPUT_DIM)
                    ))
#model.add(Flatten())
model.add(Dense(units=OUTPUT_DIM,
                activation='linear',
                # input_shape=(time_step,),
                ))  #TODO
model.compile(optimizer='rmsprop',
              loss='mse')

_, _, x_train, y_train = get_train_data(input_len=time_step, output_len=time_step_out)
y_train = y_train[:, :, 0]  #将y转换成2D
model.fit(x=x_train,
          y=y_train,
          batch_size=BatchSize,
          epochs=Epochs)
loss_train = model.evaluate(x=x_train, y=y_train, batch_size= BatchSize * 2)

_, _, x_test, y_test = get_train_data(input_len=time_step, output_len=time_step_out, train_mode=False)
y_test = y_test[:, :, 0]  #将y转换成2D
loss_test = model.evaluate(x=x_test, y=y_test, batch_size=BatchSize * 2)
y_pred = model.predict(x_test, batch_size=BatchSize)

from stock_predict_2 import calc_rsq
rsq = calc_rsq(y_test.squeeze(), y_pred.squeeze())
