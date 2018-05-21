"""
做数据
包括标准化
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
input_files = ['rb1501_300.csv',
               'rb1505_300.csv',
               'rb1510_300.csv',
               'rb1601_300.csv',
               'rb1605_300.csv',
               'rb1610_300.csv',
               'rb1701_300.csv', ]

input_files1 = ['hc1501_300.csv',
                'hc1505_300.csv',
                'hc1510_300.csv',
                'hc1601_300.csv',
                'hc1605_300.csv',
                'hc1610_300.csv',
                'hc1701_300.csv', ]
output_files = ['rb_X_train_1501-1701.csv',
                'rb_X_test_1501-1701.csv',
                'rb_Y_train_1501-1701.csv',
                'rb_Y_test_1501-1701.csv']

"""

def get_data(x, y, train_ratio=0.7):
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    x_train, x_test = x[: train_len], x[train_len:]
    y_train, y_test = y[: train_len], y[train_len:]
    return x_train, x_test, y_train, y_test
    
    
    
def rolling_standardize(arr, window=5):

    ""
    Assume first axis is rolling axis.

    ""
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


"""


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
    for start in range(0, ts_len - window + 1, stride):
        end = start + window
        sample = ts[start: end]
        samples.append(sample[np.newaxis])
    res = np.vstack(samples)
    return res

'''

def label_return(r):
    """

    :param r: is an floay
    :return: labeled return of single point
    """
    if r > 0.3:
        res = 2
    elif r < -0.3:
        res = 0
    else:
        res = 1
    return res
def label_return_array(arrr):
    """

    :param arrr: an array of return
    :return: an array of labeled return
    """
    tmp = np.apply_along_axis(label_return, axis=1, arr=arrr)
    return np.reshape(tmp, (time_step, 1))

'''

def label_return_np(arr, k=0):
    mask1 = arr >= k
    mask2 = arr < -k
    #mask3 = np.logical_not(np.logical_or(mask1, mask2))
    arr[mask1] = 0
    arr[mask2] = 1
    #arr[mask3] = 1

    values, counts = np.unique(arr, return_counts=1)
    #counts /= len(arr)
    print()
    return arr

def convert_datetime_to_int(dt):
    import datetime
    f = lambda x: x.year * 10000 + x.month * 100 + x.day
    if isinstance(dt, (datetime.datetime, datetime.date)):
        dt = pd.Timestamp(dt)
        res = f(dt)
    elif isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt)
        res = f(dt)
    else:
        dt = pd.Series(dt)
        res = dt.apply(f)
    return res

def convert_datetime_to_int_time(dt):
    import datetime
    f = lambda x: x.hour * 10000 + x.minute * 100 + x.second
    if isinstance(dt, (datetime.datetime, datetime.date)):
        dt = pd.Timestamp(dt)
        res = f(dt)
    elif isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt)
        res = f(dt)
    else:
        dt = pd.Series(dt)
        res = dt.apply(f)
    return res

global time_step  # 要与RNN里的一致
time_step = 80
global time_step_out
time_step_out = 1

'''
X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
'''

x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []
y_test_index_list = []
i=0   ########bing

for i in range(len(input_files)):

    #0     1       2         3      4    5     6     7    8      9
    # , symbol, trade_date, date, time, open, high, low, close, volume
    x_cols = ['open_1', 'high_1', 'low_1', 'close_1', 'volume_1', 'rtn_1','open_2', 'high_2', 'low_2', 'close_2', 'volume_2', 'rtn_2',]
    y_col = 'label'
    idx_cols = ['date', 'time']

    df = pd.read_csv(input_files[i],# usecols=[2, 3, 4, 5, 6, 7]
                     )
    # df.columns = ['datetime_str', 'open', 'high', 'low', 'close', 'volume']
    df = df.reindex(columns=['symbol', 'trade_date', 'date', 'time',
                             'open', 'high', 'low', 'close', 'volume'])
    # df.loc[:, 'datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H:%M:%S')
    # df.loc[:, 'date'] = convert_datetime_to_int(df['datetime'])
    # df.loc[:, 'time'] = convert_datetime_to_int_time(df['datetime'])
    #引入热轧卷板
    df1 = pd.read_csv(input_files1[i],# usecols=[2, 3, 4, 5, 6, 7]
                     )
    # df.columns = ['datetime_str', 'open', 'high', 'low', 'close', 'volume']
    df1 = df1.reindex(columns=['symbol', 'trade_date', 'date', 'time',
                             'open', 'high', 'low', 'close', 'volume'])
    df0 = pd.merge(df, df1, left_on=['trade_date', 'date', 'time'], right_on=['trade_date', 'date', 'time'], suffixes=('_1', '_2'))
    data0 = df0.copy()
    mask = (data0[['open_1', 'high_1', 'low_1', 'close_1', 'open_2','high_2','low_2','close_2']] == 0).any(axis=1)
    data0.loc[mask, :] = np.nan

    shift_len = 1
    data0.loc[:, 'rtn_1'] = data0['close_1'].pct_change(shift_len)
    data0.loc[:, 'rtn_2'] = data0['close_2'].pct_change(shift_len)
    data0.loc[:, 'label'] = data0['rtn_1'].shift(-shift_len-1)

    data0 = data0.dropna()

    index = data0[idx_cols].values
    data = data0[x_cols + [y_col]].values

    INPUT_DIM = len(x_cols)

    # split x y and train, test
    data_x = data[:, :-1]
    data_y = data[:, [-1]]
    print("data_x shape {}, data_y shape {}".format(data_x.shape, data_y.shape))

    if i < (len(input_files)-1):
        x_train = data_x
        y_train = data_y

        y_train = ts2sample(y_train, time_step, stride=1)
        y_train = np.apply_along_axis(scale, axis=0, arr=y_train)
        y_train = label_return_np(y_train)
        y_train = y_train[:, [-time_step_out]]

        x_train = ts2sample(x_train, time_step, stride=1)
        x_train = np.apply_along_axis(scale, axis=0, arr=x_train)

        x_train_index = ts2sample(index, time_step, stride=1)
        y_train_index = x_train_index[:, [-time_step_out]]

        print("train_x shape {}, train_y shape {}".format(x_train.shape, y_train.shape))
        x_train_list.append(x_train)
        y_train_list.append(y_train)




    if i == (len(input_files) - 1):
        x_test = data_x
        y_test = data_y

        ####TODO 在这加入scaler transform
        x_test = ts2sample(x_test, time_step, stride=1)
        y_test = ts2sample(y_test, time_step, stride=1)
        y_test = y_test[:, [-time_step_out]]

        y_test = np.apply_along_axis(scale, axis=0, arr=y_test)
        y_test = label_return_np(y_test)
        y_test = y_test[:, [-time_step_out]]

        x_test_index = ts2sample(index, time_step, stride=1)
        y_test_index = x_test_index[:, [-time_step_out]]
        y_test_index_list.append(y_test_index)

        print("test_x shape {}, test_y shape {}".format(x_test.shape, y_test.shape))
        x_test_list.append(x_test)
        y_test_list.append(y_test)
    #x_train, x_test, y_train, y_test = get_data(data_x, data_y, train_ratio=0.7)

    # rolling standardization
    # y_train, _, _ = rolling_standardize(y_train, window=ROLL_WINDOW)
    # y_test, mean_y_test, std_y_test = rolling_standardize(y_test, window=ROLL_WINDOW)
    # y_test_std, mean_y_test, std_y_test = y_test, 0, 1

    # INPUT_LEN > OUTPUT_LEN
    # y_test_std = ts2sample(y_test_std, time_step, stride=1)



X_train = np.concatenate(x_train_list, axis=0)
Y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(x_test_list, axis=0)
Y_test = np.concatenate(y_test_list, axis=0)
Y_test_index = np.concatenate(y_test_index_list, axis=0)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
np.save('rb_X_train_1501-1605.npy', X_train)
np.save('rb_X_test_1501-1605.npy', X_test)
np.save('rb_Y_train_1501-1605.npy', Y_train)
np.save('rb_Y_test_1501-1605.npy', Y_test)
np.save('rb_Y_test_index_1501-1605.npy', Y_test_index)
