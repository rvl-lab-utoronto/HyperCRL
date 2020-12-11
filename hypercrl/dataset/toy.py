import numpy as np
import sklearn.datasets
import torch

from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler


def make_moons(n_samples, n_sets=1, shuffle=True, noise=0, random_state=None, scale=False):
    Xs, ys = [np.zeros((0, 2)), np.array([])]
    offsets = np.array([[0, 0], [4, 0], [-4, 0], [0, 2], [0, -2],
                        [4, 2], [-4, 2], [4, -2], [-4, -2]])
    
    n_sets = min(n_sets, len(offsets))
    n_samples //= n_sets

    for i in range(n_sets):
        X, y = sklearn.datasets.make_moons(n_samples=n_samples, shuffle=shuffle,
                                          noise=noise, random_state=random_state)
        X += offsets[i]
        y += (2 * i)
        Xs = np.concatenate((Xs, X))
        ys = np.concatenate((ys, y))
    
    if scale:
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        Xs = min_max_scaler.fit_transform(Xs)
    return Xs, ys

def make_spirals(n_samples=10000, n_sets=1, shuffle=True, noise=0, random_state=None):
    Xs, ys = [np.zeros((0, 2)), np.array([])]

    thetas = np.linspace(0, np.pi * 2, num=n_sets, endpoint=False)
    
    def make_arcs():
        X, y = sklearn.datasets.make_moons(n_samples=n_samples, shuffle=shuffle,
                                        noise=noise, random_state=random_state)
        # only keep the top half moon
        X = X[y==0]
        # shift to right
        X += [1.5, 0.2]

        return X
    
    for i in range(n_sets):
        X = make_arcs()
        
        rot_mat = np.array([[np.cos(thetas[i]), -np.sin(thetas[i])],
                            [np.sin(thetas[i]), np.cos(thetas[i])]])

        X = X @ rot_mat.T
        y = np.full(X.shape[0], i)
        Xs = np.concatenate((Xs, X))
        ys = np.concatenate((ys, y))

    return Xs, ys

def split_array_2_torchd(X, y, test_size=0.25):
    trainval = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    valsize = int(len(trainval) * test_size)
    trainsize = len(trainval) - valsize
        
    train_set, val_set = random_split(trainval, [trainsize, valsize])
    return train_set, val_set


def moon_dataset(n_samples=10000, n_sets=1, shuffle=True, test_size = 0.25,
        noise=0.05, random_state=None, scale=True):

    X, y = make_moons(n_samples=n_samples, n_sets=n_sets, shuffle=shuffle,
        noise=noise, random_state=random_state, scale=scale)
        
    return split_array_2_torchd(X, y)

def spiral_dataset(n_samples=10000, n_sets=1, shuffle=True, test_size = 0.25, 
        noise=0.05, random_state=None):

    X, y = make_spirals(n_samples=n_samples, n_sets=n_sets, shuffle=shuffle,
        noise=noise, random_state=random_state)
    return split_array_2_torchd(X, y)