from flax import linen as nn


class ConvNet(nn.Module):
    """
    A simple CNN model:
        https://github.com/google/flax/blob/main/examples/mnist/train.py
    """

    @nn.compact
    def __call__(self, x, n_conv_layers: int = 2, train: bool = False):
        x = x.reshape(1, 28, 28, 1)
        for i in range(n_conv_layers):
            x = nn.Conv(features=16 * (i + 1), kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = x.reshape(-1)
        return x
