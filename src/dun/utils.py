import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch


def set_all_seeds_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gen_simple_1d(hetero=False):
    np.random.seed(0)
    Npoints = 1002
    x0 = np.random.uniform(-1, 0, size=int(Npoints / 3))
    x1 = np.random.uniform(1.7, 2.5, size=int(Npoints / 3))
    x2 = np.random.uniform(4, 5, size=int(Npoints / 3))
    x = np.concatenate([x0, x1, x2])

    def function(x):
        return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

    y = function(x)

    homo_noise_std = 0.25
    homo_noise = np.random.randn(*x.shape) * homo_noise_std
    y_homo = y + homo_noise

    hetero_noise_std = np.abs(0.1 * np.abs(x) ** 1.5)
    hetero_noise = np.random.randn(*x.shape) * hetero_noise_std
    y_hetero = y + hetero_noise

    X = x[:, np.newaxis]
    y_joint = np.stack([y_homo, y_hetero], axis=1)

    X_train, X_test, y_joint_train, y_joint_test = train_test_split(X, y_joint, test_size=0.5, random_state=42)
    y_hetero_train, y_hetero_test = y_joint_train[:, 1, np.newaxis], y_joint_test[:, 1, np.newaxis]
    y_homo_train, y_homo_test = y_joint_train[:, 0, np.newaxis], y_joint_test[:, 0, np.newaxis]

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_hetero_means, y_hetero_stds = y_hetero_train.mean(axis=0), y_hetero_train.std(axis=0)
    y_homo_means, y_homo_stds = y_homo_test.mean(axis=0), y_homo_test.std(axis=0)

    X_train = ((X_train - x_means) / x_stds).astype(np.float32)
    X_test = ((X_test - x_means) / x_stds).astype(np.float32)

    y_hetero_train = ((y_hetero_train - y_hetero_means) / y_hetero_stds).astype(np.float32)
    y_hetero_test = ((y_hetero_test - y_hetero_means) / y_hetero_stds).astype(np.float32)

    y_homo_train = ((y_homo_train - y_homo_means) / y_homo_stds).astype(np.float32)
    y_homo_test = ((y_homo_test - y_homo_means) / y_homo_stds).astype(np.float32)

    if hetero:
        return X_train, y_hetero_train, X_test, y_hetero_test
    else:
        return X_train, y_homo_train, X_test, y_homo_test
