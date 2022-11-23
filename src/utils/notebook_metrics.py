import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from chex import assert_rank, assert_shape, assert_equal_shape

from math import comb


def categorical_probs(logits):
    assert_rank(logits, 2)
    assert_shape([logits], (None, 10))
    probs = nn.softmax(logits.mean(axis=0))
    assert_shape([probs], (10,))
    return probs


def categorical_probs_avg_probs(logits):
    assert_rank(logits, 2)
    assert_shape([logits], (None, 10))
    probs = nn.softmax(logits).mean(axis=0)
    assert_shape([probs], (10,))
    return probs


def categorical_probs_prod(logits, M=5, C=10):
    assert_rank(logits, 2)
    assert_shape([logits], (M, C))
    probs = nn.softmax(logits).prod(axis=0)
    probs = probs / (probs.sum() + 1e-36)
    assert_shape([probs], (C,))
    return probs


def categorical_entropy(logits):
    probs = categorical_probs(logits)
    cat = distrax.Categorical(probs=probs)
    return cat.entropy()


def categorical_entropy_avg_probs(logits):
    probs = categorical_probs_avg_probs(logits)
    cat = distrax.Categorical(probs=probs)
    return cat.entropy()


def average_entropy(logits):
    """
    1/M \sum_{m=1}^M H[p_m(y | x)]
    """
    probs = nn.softmax(logits)
    entropies = jax.vmap(lambda x: distrax.Categorical(probs=x).entropy(), in_axes=(0,))(probs)
    return entropies.mean()


def mutual_information(logits):
    """
    H[1/M \sum_{m=1}^M p_m(y | x)] - 1/M \sum_{m=1}^M H[p_m(y | x)]
    """
    return categorical_entropy_avg_probs(logits) - average_entropy(logits)


def energy_num_stable(logits):
    assert_rank(logits, 1)
    c = jnp.max(logits)
    return - (c + jnp.log(jnp.exp(logits - c).sum()))


def average_energy(logits, C=10):
    """
    1/M \sum_{m=1}^M - log(\sum_k exp v_k), where v_k are the logits
    """
    assert_rank(logits, 2)
    assert_shape([logits], (None, C))
    return jax.vmap(energy_num_stable, in_axes=(0,))(logits).mean()


def categorical_entropy_prod_probs(logits):
    probs = ovr_prod_probs(logits)
    cat = distrax.Categorical(probs=probs)
    return cat.entropy()


def msp_avg_probs(logits):
    probs = categorical_probs_avg_probs(logits)
    return 1. - jnp.max(probs)


def msp_prod_probs(logits):
    probs = ovr_prod_probs(logits)
    return 1. - jnp.max(probs)


def categorical_nll(logits, y):
    probs = categorical_probs(logits).clip(min=1e-36)
    cat = distrax.Categorical(probs=probs)
    return -cat.log_prob(y)


def mse(x, y):
    assert_equal_shape([x, y])
    assert_shape(x, (10,))
    return ((x - y)**2).mean()


def categorical_brier(logits, y):
    probs = categorical_probs(logits)
    return mse(probs, jax.nn.one_hot(y, 10))


def categorical_err(logits, y):
    probs = categorical_probs(logits)
    return y != jnp.argmax(probs, axis=0)


def categorical_err_prod(logits, y):
    probs = categorical_probs_prod(logits)
    return y != jnp.argmax(probs, axis=0)


def multiply_no_nan(x, y):
    """Equivalent of TF `multiply_no_nan`.
    Computes the element-wise product of `x` and `y` and return 0 if `y` is zero,
    even if `x` is NaN or infinite.
    Args:
        x: First input.
        y: Second input.
    Returns:
        The product of `x` and `y`.
    Raises:
        ValueError if the shapes of `x` and `y` do not match.
    """
    dtype = jnp.result_type(x, y)
    return jnp.where(y == 0, jnp.zeros((), dtype=dtype), x * y)


def ovr_prod_probs(logits):
    assert_rank(logits, 2)
    assert_shape([logits], (None, 10))
    σ = nn.sigmoid(logits).round().prod(axis=0)#.clip(min=1e-36)
    assert_shape([σ], (10,))
    probs = σ/(σ.sum() + 1e-36)
    return probs


def ovr_entropy(logits):
    probs = ovr_prod_probs(logits)
    return -jnp.sum(multiply_no_nan(jnp.log(probs), probs), axis=-1)


def ovr_nll(logits, y):
    probs = ovr_prod_probs(logits)
    return -jnp.log(probs[y])


def ovr_brier(logits, y):
    probs = ovr_prod_probs(logits)
    return mse(probs, jax.nn.one_hot(y, 10))


def ovr_err(logits, y):
    probs = ovr_prod_probs(logits)
    return y != probs.argmax(axis=0)


def max_voting(logits, y):
    return nn.sigmoid(logits).round().sum(axis=0).argmax() != y


def pairwise_ce(logits):
    M = logits.shape[0]
    assert_rank(logits, 2)
    assert_shape([logits], (None, 10))
    softmax = nn.softmax(logits)
    total_ce = 0.
    for i in range(M):
        for j in range(M):
            if i != j:
                total_ce += -jnp.dot(softmax[i, :], jnp.log(softmax[j, :]))
    return total_ce / comb(M, 2)


def pairwise_abs_diff(logits):
    M = logits.shape[0]
    assert_rank(logits, 2)
    assert_shape([logits], (None, 10))
    softmax = nn.softmax(logits)
    total_diff = 0.
    for i in range(M):
        for j in range(M):
            if i != j:
                total_diff += jnp.abs(softmax[i, :] - softmax[j, :]).sum()
    return total_diff / comb(M, 2)




