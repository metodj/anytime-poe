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


def categorical_entropy(logits):
    probs = categorical_probs(logits)
    cat = distrax.Categorical(probs=probs)
    return cat.entropy()


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