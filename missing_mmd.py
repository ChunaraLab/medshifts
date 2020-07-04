'''
Calculating MMD with missing data
'''

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib import rc

seed = 1
np.random.seed(seed)

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('axes', labelsize=20)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
rc('legend', fontsize=12)

datset = 'gaussian'

def load_hosp_dataset(dataset):
    x_train = None
    external_dataset_path = './datasets/'

    if dataset == 'gaussian':
        pass
    elif dataset == 'boston':
        (x_train, _), (x_test, _) = boston_housing.load_data()
    elif dataset == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train.reshape(len(x_train), 28, 28, 1)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)
        x_train, x_test = x_train[:10000,], x_test[:1000,]

    return x_train, x_test

def add_missing(data, prop=0.0):
    N = int(prop*data.shape[0]*data.shape[1])
    row = np.random.randint(0, data.shape[0], N)
    col = np.random.randint(0, data.shape[0], N)

    data_m = data.copy()
    data_m.ix_[row, col] = np.nan

    return data_m