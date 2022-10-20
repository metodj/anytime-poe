import distrax
from jax import numpy as jnp


def uniform_entropy(loc, scale):
    upper = loc + scale
    lower = loc - scale
    # return jnp.log(upper - lower)
    uni = distrax.Uniform(lower, upper)
    return uni.entropy()


def uniform_nll(loc, scale, y):
    upper = loc + scale
    lower = loc - scale
    # return jnp.log(upper - lower)
    uni = distrax.Uniform(lower, upper)
    return -uni.log_prob(y)


def normal_entropy(loc, scale):
    norm = distrax.Normal(loc, scale)
    return norm.entropy()


def normal_nll(loc, scale, y):
    norm = distrax.Normal(loc, scale)
    return -norm.log_prob(y)


def mse(loc, y):
    return jnp.mean((loc - y)**2)
