import os
from numpy.linalg import linalg
import numpy
import theano
import theano.tensor as T
import pandas as pd

valid_end = 17222

test_end = 16500

train_end = 5189

__author__ = 'taylor'

nploc = '/home/taylor/data/random/7.7.4.7.x.npy'
nplocy = '/home/taylor/data/random/7.7.4.7.y.npy'


# testing indicates 10 million points ~= 37 MB, so theano is using c sized float32 (4Byte/item)
# total size = model + data = 4 * (shape + n_in+ n_hid), but really dominated by shape(rows*cols)
# get data for flattened shape of n,7,4,7,7
# i.e. n, time, channel, width, height
def get_data_raw(shuffle=False):
    if os.path.isfile(nploc):
        data_x = numpy.load(nploc)
        data_y = numpy.load(nplocy)
    else:
        # data = numpy.loadtxt('/home/taylor/data/random/7.7.4.7.july.oct.csv', delimiter=',')
        data = numpy.loadtxt('/home/taylor/data/random/7.7.4.7.july.oct.csv', delimiter=',')
        n = len(data)
        rx = numpy.zeros([n, 7, 4, 7, 7])
        for i in range(n):
            rx[i, :, :, :, :] = reorder(reshape(data[i, :-1]))
        for c in range(4):
            p, mean = fit_zca(rx[:train_end, :, c, :, :].reshape((train_end, 7 * 7 * 7)))
            rx[:, :, c, :, :] = apply_zca(rx[:, :, c, :, :].reshape((n, 7 * 7 * 7)), p, mean).reshape((n, 7, 7, 7))
        data_x = numpy.zeros([n, 7 * 4 * 7 * 7])
        for i in range(n):
            data_x[i, :] = rx[i].flatten()
        data_y = data[:, -1]
        numpy.save(nploc, data_x)
        numpy.save(nplocy, data_y)
    if shuffle:
        rxy = numpy.random.permutation(numpy.hstack((data_x, data_y.reshape((data_y.shape[0], 1)))))
        return rxy[:, :-1], rxy[:, -1]
    else:
        return data_x, data_y


def get_data_raw_new(shuffle=False):
    if os.path.isfile(nploc):
        data_x = numpy.load(nploc)
        data_y = numpy.load(nplocy)
    else:
        data = pd.read_csv('/home/taylor/data/random/conv.drug.csv', delimiter=',',header=None).values
        data_x = data[:, :-1]
        data_y = data[:, -1]
        numpy.save(nploc, data_x)
        numpy.save(nplocy, data_y)
    if shuffle:
        rxy = numpy.random.permutation(numpy.hstack((data_x, data_y.reshape((data_y.shape[0], 1)))))
        return rxy[:, :-1], rxy[:, -1]
    else:
        return data_x, data_y


def reshape_data(x, conv2d_size, conv3d_size):
    prev_index = 0
    if conv2d_size:
        c, w, h = conv2d_size
        x_stat = x[:, prev_index: c * w * h].reshape((x.shape[0], c, w, h))
        prev_index = c * w * h
        for idx, chan in enumerate(range(c)):
            P, mean = fit_zca(x_stat[:, chan, :, :].reshape(x.shape[0], w * h))
            x_stat[:, chan, :, :] = apply_zca(x_stat[:, chan, :, :].reshape(x.shape[0], w * h), P, mean).reshape(
                x.shape[0], w, h)
        x_stat = x_stat.reshape(x.shape[0], c * w * h)
    if conv3d_size:
        t, c, w, h = conv3d_size
        x_dyn = x[:, prev_index:].reshape((x.shape[0], t, c, w, h))
        for idx, chan in enumerate(range(c)):
            P, mean = fit_zca(x_dyn[:, :, chan, :, :].reshape(x.shape[0], t * w * h))
            x_dyn[:, :, chan, :, :] = apply_zca(x_dyn[:, :, chan, :, :].reshape(x.shape[0], t * w * h), P,
                                                mean).reshape(x.shape[0], t, w, h)
        x_dyn = x_dyn.reshape(x.shape[0], t * c * w * h)
    return x_stat, x_dyn


def get_shared(input, dtype=theano.config.floatX):
    val = theano.shared(numpy.asarray(input,
                                      dtype=theano.config.floatX),
                        borrow=True)
    if dtype != theano.config.floatX:
        return T.cast(val, dtype)
    else:
        return val


def get_data(shuffle=False):
    data_x, data_y = get_data_raw(shuffle)
    train_set_x = get_shared(data_x[0:train_end, :])
    test_set_x = get_shared(data_x[train_end:test_end, :])
    valid_set_x = get_shared(data_x[test_end:valid_end, :])
    y_type = 'int32'
    train_set_y = get_shared(data_y[0:train_end], y_type)
    test_set_y = get_shared(data_y[train_end:test_end], y_type)
    valid_set_y = get_shared(data_y[test_end:valid_end], y_type)
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), data_y


def get_data_no_time(shuffle=False):
    data_x, data_y = get_data_raw(shuffle)
    x_stat_temp = numpy.zeros([data_x.shape[0], 4, 7, 7], dtype=theano.config.floatX)
    for row, line in enumerate(data_x):
        rx = line.reshape(7, 4, 7, 7)
        for k in range(4):
            x_stat_temp[row, k, :, :] = rx[0, k, :, :]
    x_stat = numpy.zeros([data_x.shape[0], 4 * 7 * 7], dtype=theano.config.floatX)
    for row in range(len(data_x)):
        x_stat[row, :] = x_stat_temp[row, :, :, :].flatten()
    train_set_x_stat = get_shared(x_stat[0:train_end, :])
    test_set_x_stat = get_shared(x_stat[train_end:test_end, :])
    valid_set_x_stat = get_shared(x_stat[test_end:valid_end, :])
    y_type = 'int32'
    train_set_y = get_shared(data_y[0:train_end], y_type)
    test_set_y = get_shared(data_y[train_end:test_end], y_type)
    valid_set_y = get_shared(data_y[test_end:valid_end], y_type)
    return (train_set_x_stat, train_set_y), \
           (valid_set_x_stat, valid_set_y), \
           (test_set_x_stat, test_set_y), data_y


def get_data_dyn_split_raw(shuffle):
    data_x, data_y = get_data_raw(shuffle)
    static = [0, 1]
    dyn = [2, 3]
    x_stat_temp = numpy.zeros([data_x.shape[0], len(static), 7, 7], dtype=theano.config.floatX)
    x_dyn_temp = numpy.zeros([data_x.shape[0], 7, len(dyn), 7, 7], dtype=theano.config.floatX)
    for row, line in enumerate(data_x):
        rx = line.reshape(7, 4, 7, 7)
        for k, s in enumerate(static):
            x_stat_temp[row, k, :, :] = rx[0, s, :, :]
        for k, d in enumerate(dyn):
            x_dyn_temp[row, :, k, :, :] = rx[:, d, :, :]
    x_stat = numpy.zeros([data_x.shape[0], len(static) * 7 * 7], dtype=theano.config.floatX)
    x_dyn = numpy.zeros([data_x.shape[0], 7 * len(dyn) * 7 * 7], dtype=theano.config.floatX)
    for row in range(len(data_x)):
        x_stat[row, :] = x_stat_temp[row, :, :, :].flatten()
        x_dyn[row, :] = x_dyn_temp[row, :, :, :, :].flatten()
    return data_y, x_dyn, x_stat


def get_data_dyn_split(shuffle=False):
    # data_y, x_dyn, x_stat = get_data_dyn_split_raw(shuffle)
    data_x, data_y = get_data_raw_new(shuffle)
    x_stat, x_dyn = reshape_data(data_x, (2,7,7), (7, 2,7,7))
    train_set_x_stat = get_shared(x_stat[0:train_end, :])
    train_set_x_dyn = get_shared(x_dyn[0:train_end, :])
    test_set_x_stat = get_shared(x_stat[train_end:test_end, :])
    test_set_x_dyn = get_shared(x_dyn[train_end:test_end, :])
    valid_set_x_stat = get_shared(x_stat[test_end:valid_end, :])
    valid_set_x_dyn = get_shared(x_dyn[test_end:valid_end, :])
    y_type = 'int32'
    train_set_y = get_shared(data_y[0:train_end], y_type)
    test_set_y = get_shared(data_y[train_end:test_end], y_type)
    valid_set_y = get_shared(data_y[test_end:valid_end], y_type)
    return (train_set_x_stat, train_set_x_dyn, train_set_y), \
           (valid_set_x_stat, valid_set_x_dyn, valid_set_y), \
           (test_set_x_stat, test_set_x_dyn, test_set_y), data_y


def reorder(row):
    t, w, h, c = row.shape
    rr = numpy.array(row).reshape(t, c, w, h)
    for time in range(t):
        for ch in range(c):
            rr[time, ch, :, :] = row[time, :, :, ch]
    return rr


def reshape(row):
    return row.reshape(7, 7, 7, 4)

#adapted from pylearn2
def fit_zca(X):
    filter_bias = 0.1
    n_samples = X.shape[0]
    X = X.copy()
    # Center data
    mean_ = numpy.mean(X, axis=0)
    X -= mean_
    eigs, eigv = linalg.eigh(numpy.dot(X.T, X) / X.shape[0] +
                             filter_bias * numpy.identity(X.shape[1]))
    assert not numpy.any(numpy.isnan(eigs))
    assert not numpy.any(numpy.isnan(eigv))
    assert eigs.min() > 0
    P = numpy.dot(eigv * numpy.sqrt(1.0 / eigs),
                  eigv.T)
    assert not numpy.any(numpy.isnan(P))
    return P, mean_


def apply_zca(X, P=None, mean_=None):
    if P == None:
        P, mean_ = fit_zca(X)
    return numpy.dot(X - mean_, P)


if __name__ == '__main__':
    data_y, x_dyn, x_stat = get_data_dyn_split_raw(False)
    d = numpy.hstack((x_stat, x_dyn, data_y.reshape(data_y.shape[0],1)))
    numpy.savetxt('/tmp/all.csv', d, delimiter=',')