from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, GRU
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd



def calc_rsq(y, yhat):
    ret = 1 - ((y - yhat) ** 2).mean() / y.var()
    return ret
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


if __name__ == "__main__":

    time_step = 80
    Epochs = 750
    BatchSize = 80
    OUTPUT_DIM = 1
    time_step_out = 1
    n_classes = 3
    """
        ##导入数据
        df = pd.read_csv('Archive/futures-historical-data/rb1505_300.csv',
                         usecols=[3, 4, 5, 6, 7, 6])
        df = df.reindex(columns=['open', 'high', 'low', 'close', 'volume'])
        # TODO 导入热卷轧
        shift_len = time_step
        df.loc[:, 'rtn'] = df['close']
        df.loc[:, 'rtn'] = df['rtn'].pct_change(1)
        df.loc[:, 'label'] = df['rtn']
        df.loc[:, 'label'] = df['label'].shift(-shift_len)

        data = df.dropna()
        data = data.values

        INPUT_DIM = data.shape[1] - 1

        # split x y and train, test
        data_x = data[:, :-1]
        data_y = data[:, [-1]]
        x_train, x_test, y_train, y_test = get_data(data_x, data_y, train_ratio=0.7)

        # rolling standardization
        ROLL_WINDOW = 50
        print(x_train.shape, y_train.shape)
        x_train, _, _ = rolling_standardize(x_train, window=ROLL_WINDOW)
        x_test, _, _ = rolling_standardize(x_test, window=ROLL_WINDOW)
        y_train, _, _ = rolling_standardize(y_train, window=ROLL_WINDOW)
        y_test, mean_y_test, std_y_test = rolling_standardize(y_test, window=ROLL_WINDOW)
        #y_test_std, mean_y_test, std_y_test = y_test, 0, 1
        print(x_train.shape, y_train.shape)

        # INPUT_LEN > OUTPUT_LEN
        x_train = ts2sample(x_train, time_step, stride=1)
        x_test = ts2sample(x_test, time_step, stride=1)
        y_train = ts2sample(y_train, time_step, stride=1)
        y_test = ts2sample(y_test, time_step, stride=1)
        #y_test_std = ts2sample(y_test_std, time_step, stride=1)

        y_train = y_train[:, [-time_step_out]]
        y_test = y_test[:, [-time_step_out]]

        TOTAL_LEN = len(data)

    """

    # 导入数据
    x_train = np.load('Archive/futures-historical-data/rb_X_train_1501-1701.npy')
    x_test = np.load('Archive/futures-historical-data/rb_X_test_1501-1701.npy')
    y_train = np.load('Archive/futures-historical-data/rb_Y_train_1501-1701.npy')
    y_test = np.load('Archive/futures-historical-data/rb_Y_test_1501-1701.npy')

    # test with minimal data size
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    y_train = y_train[:, 0]  # dicard time step dimension (which is 1)
    y_test = y_test[:, 0]  # dicard time step dimension (which is 1)
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)
    INPUT_DIM = x_train.shape[2]

    #建立网络
    model = Sequential()
    model.add(LSTM(units=INPUT_DIM*5,
                   return_sequences=True,
                   input_shape=(time_step, INPUT_DIM),
                   kernel_initializer='Orthogonal',
                   unit_forget_bias=True
                   )
              )
    model.add(LSTM(units=INPUT_DIM*3,
                   return_sequences=True,
                   input_shape=(time_step, INPUT_DIM),
                   kernel_initializer='Orthogonal',
                   unit_forget_bias=True)
              )
    model.add(LSTM(units=INPUT_DIM,
                   input_shape=(time_step,INPUT_DIM),
                   kernel_initializer='Orthogonal',
                   unit_forget_bias=True)
              )
    #model.add(Dense(units=INPUT_DIM,
     #               activation='tanh',
      #              kernel_initializer='RandomNormal')
       #       )
    model.add(Dense(units=n_classes * 3)
              )

    model.add(Dense(units=n_classes,
                    activation='softmax',
                    )
              )
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # test
    # y_pred = model.predict(x_test[:10], batch_size=1)

    #y_train = y_train[:, :, 0]  #将y转换成2D
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=BatchSize,
                        epochs=Epochs)
    loss_train = model.evaluate(x=x_train, y=y_train, batch_size= 1)
    print(loss_train)
    #y_test = y_test[:, :, 0]  #将y转换成2D
    y_pred = model.predict(x_test, batch_size=BatchSize)
    loss_pre = model.evaluate(x=x_test, y=y_pred, batch_size=1)
    print(loss_pre)

    from stock_predict_2 import calc_rsq
    rsq = calc_rsq(y_test.squeeze(), y_pred.squeeze())
    print(rsq)
