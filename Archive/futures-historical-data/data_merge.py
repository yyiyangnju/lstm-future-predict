"""
做数据
包括标准化
"""
import numpy as np
import pandas as pd

input_files = ['rb1501_300.csv',
               'rb1505_300.csv',
               'rb1510_300.csv',
               'rb1601_300.csv',
               'rb1605_300.csv',
               'rb1610_300.csv',
               'rb1701_300.csv', ]

output_files = ['rb_X_train_1501-1701.csv',
                'rb_X_test_1501-1701.csv',
                'rb_Y_train_1501-1701.csv',
                'rb_Y_test_1501-1701.csv']


def get_data(x, y, train_ratio=0.7):
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    x_train, x_test = x[: train_len], x[train_len:]
    y_train, y_test = y[: train_len], y[train_len:]
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
    df_std = df_std.iloc[window:]
    arr_std = df_std.values
    return arr_std, mean[window:], std[window:]


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


global time_step  # 要与RNN里的一致
time_step = 80
global time_step_out
time_step_out = 1

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
for index in range(len(input_files)):

    df = pd.read_csv(input_files[index],
                     usecols=[3, 4, 5, 6, 7, 6])
    df = df.reindex(columns=['open', 'high', 'low', 'close', 'volume'])

    shift_len = time_step
    df.loc[:, 'rtn'] = df['close']
    df.loc[:, 'rtn'] = df['rtn'].pct_change(1)


    def label_return(r):
        if r > 0:
            res = 1
        elif r < 0:
            res = -1
        else:
            res = 0
        return res


    df.loc[:, 'label'] = df['rtn'].apply(label_return)
    df.loc[:, 'label'] = df['label'].shift(-shift_len)

    data = df.dropna()
    data = data.values

    INPUT_DIM = data.shape[1] - 1

    # split x y and train, test
    data_x = data[:, :-1]
    data_y = data[:, [-1]]
    x_train, x_test, y_train, y_test = get_data(data_x, data_y, train_ratio=0.7)

    # rolling standardization
    ROLL_WINDOW = time_step
    print("roll window = {:d}".format(ROLL_WINDOW))
    print(x_train.shape, y_train.shape)
    x_train, _, _ = rolling_standardize(x_train, window=ROLL_WINDOW)
    x_test, _, _ = rolling_standardize(x_test, window=ROLL_WINDOW)
    y_train = y_train[ROLL_WINDOW:]
    y_test = y_test[ROLL_WINDOW:]
    # y_train, _, _ = rolling_standardize(y_train, window=ROLL_WINDOW)
    # y_test, mean_y_test, std_y_test = rolling_standardize(y_test, window=ROLL_WINDOW)
    # y_test_std, mean_y_test, std_y_test = y_test, 0, 1
    print("shape after std: ", x_train.shape, y_train.shape)

    # INPUT_LEN > OUTPUT_LEN
    x_train = ts2sample(x_train, time_step, stride=1)
    x_test = ts2sample(x_test, time_step, stride=1)
    y_train = ts2sample(y_train, time_step, stride=1)
    y_test = ts2sample(y_test, time_step, stride=1)
    # y_test_std = ts2sample(y_test_std, time_step, stride=1)

    y_train = y_train[:, [-time_step_out]]
    y_test = y_test[:, [-time_step_out]]

    TOTAL_LEN = len(data)
    if index == 0:
        X_train = x_train
        X_test = x_test
        Y_train = y_train
        Y_test = y_test
    else:
        X_train = np.concatenate((X_train, x_train), axis=0)
        X_test = np.concatenate((X_test, x_test), axis=0)
        Y_train = np.concatenate((Y_train, y_train), axis=0)
        Y_test = np.concatenate((Y_test, y_test), axis=0)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
np.save('rb_X_train_1501-1701.npy', X_train)
np.save('rb_X_test_1501-1701.npy', X_test)
np.save('rb_Y_train_1501-1701.npy', Y_train)
np.save('rb_Y_test_1501-1701.npy', Y_test)
