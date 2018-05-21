# encoding: utf-8

#import gpu_config
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import keras


def calc_rsq(y, yhat):
    ret = 1 - ((y - yhat) ** 2).mean() / y.var()
    return ret


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


def main():
    time_step =80
    Epochs = 6
    BatchSize =128
    OUTPUT_DIM = 1
    time_step_out = 1
    n_classes = 2


    # 导入数据
    x_train = np.load('Archive/futures-historical-data/rb_X_train_1501-1605.npy')
    x_test = np.load('Archive/futures-historical-data/rb_X_test_1501-1605.npy')
    y_train = np.load('Archive/futures-historical-data/rb_Y_train_1501-1605.npy')
    y_test = np.load('Archive/futures-historical-data/rb_Y_test_1501-1605.npy')
    y_test_index = np.load('Archive/futures-historical-data/rb_Y_test_index_1501-1605.npy')
    y_test_index = pd.DataFrame(y_test_index[:, 0, :])
    y_test_index.columns = ['date', 'time']

    assert len(y_test) == len(y_test_index)

    print(x_test.shape, y_test.shape)

    # test with minimal data size
    n_cut = 0
    if n_cut > 0:
        x_train = x_train[:n_cut]
        y_train = y_train[:n_cut]

    y_train = y_train[:, 0]  # dicard time step dimension (which is 1)
    y_test = y_test[:, 0]  # dicard time step dimension (which is 1)
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    INPUT_DIM = x_train.shape[2]

    # 建立网络
    penalty_c = 0.001
    model = Sequential()
    model.add(LSTM(units=12,
                   return_sequences=True,
                   kernel_initializer='Orthogonal',
                   recurrent_initializer='Orthogonal',
                   input_shape=(time_step, INPUT_DIM),
                   recurrent_dropout=0.5,
                   kernel_regularizer=keras.regularizers.l2(penalty_c)
                   )
              #merge_mode='sum'
              )

    model.add(LSTM(units=12,
                   return_sequences=True,
                   kernel_initializer='Orthogonal',
                   recurrent_initializer='Orthogonal',
                   input_shape=(time_step, INPUT_DIM),
                   recurrent_dropout=0.5,
                   kernel_regularizer=keras.regularizers.l2(penalty_c)
                   )
              #merge_mode='sum'
              )
    model.add(LSTM(units=12,
                   input_shape=(time_step, INPUT_DIM),
                   recurrent_dropout=0.5,
                   kernel_regularizer=keras.regularizers.l2(penalty_c)
                   )
              )
    model.add(Dense(units=6, activation='relu')
              )

    # model.add(keras.layers.core.Dropout(0.5))
    model.add(Dense(units=n_classes,
                    activation='softmax',
                    )
              )
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    batch_size_eval = 128

    mode = 'train'
    if mode == 'predict':
        pred = model.predict(x=x_test, batch_size=batch_size_eval, verbose=1)
        print(pred)
        y_pred = np.argmax(pred, axis=1)
        assert len(y_pred) == len(y_test_index)
        res = pd.concat([y_test_index,
                         pd.DataFrame(data={'y_pred': y_pred})],
                        axis=1)
        res.to_hdf('y_pred.hd5', key='y_pred')
        return
    elif mode == 'evaluate':
        # evaluate before fit
        metric_test = model.evaluate(x=x_test, y=y_test, batch_size=batch_size_eval, verbose=0)
        print(metric_test)
        return

    # evaluate before fit
    metric_test = model.evaluate(x=x_test, y=y_test, batch_size=batch_size_eval, verbose=0)
    print(metric_test)

    #filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callback_list = [checkpoint]
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_test, y_test),
                        batch_size=BatchSize,
                        epochs=Epochs)
    # callbacks=callback_list)

    to_dump = {'history': history.history,
               'params': history.params,
               'epoch': history.epoch}
    with open('train_history.pkl', mode='wb') as f:
        pickle.dump(to_dump, f, pickle.HIGHEST_PROTOCOL)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_weights.h5")
    print("Saved model to disk")

    metric_train = model.evaluate(x=x_train, y=y_train, batch_size=batch_size_eval, verbose=0)
    print("Train eval result: ", metric_train)

    metric_test = model.evaluate(x=x_test, y=y_test, batch_size=batch_size_eval, verbose=0)
    with open('test_metric.pkl', mode='wb') as f:
        pickle.dump(metric_test, f, pickle.HIGHEST_PROTOCOL)
    print("Test eval result: ", metric_test)

    pred = model.predict(x=x_test, batch_size=batch_size_eval, verbose=1)
    print(pred)
    y_pred = np.argmax(pred, axis=1)
    assert len(y_pred) == len(y_test_index)
    res = pd.concat([y_test_index,
                     pd.DataFrame(data={'y_pred': y_pred})],
                    axis=1)
    res.to_hdf('y_pred.hd5', key='y_pred')


if __name__ == "__main__":
    import pickle

    main()
