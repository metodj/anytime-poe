from typing import Callable, List
from enum import Enum

import jax
import jax.numpy as jnp
import flax.linen as nn
from chex import Array, assert_rank, assert_equal_shape
import distrax

from itertools import chain, combinations

PRNGKey = jnp.ndarray

NOISE_TYPES = [
    'homo',
    'per-ens-homo',
    'hetero',
]


class MembersLL(Enum):
    soft_ovr = "soft_ovr"
    softmax = "softmax"
    GND = "GND"
    gaussian = "gaussian"


def raise_if_not_in_list(val, valid_options, varname):
    if val not in valid_options:
       msg = f'`{varname}` should be one of `{valid_options}` but was `{val}` instead.'
       raise RuntimeError(msg)


def get_locs_scales_probs(
    obj,
    x: Array,
    train: bool = False,
    ensemble_ids: List[int] = (0, 1, 2, 3, 4,)
):
    # ens_preds = jnp.stack([net(x, train=train) for i, net in enumerate(obj.nets) if i in ensemble_ids], axis=0)  # (M, O * 2) or (M, O)
    ens_preds = jnp.stack([net(x, train=train) for net in obj.nets], axis=0)[jnp.array(ensemble_ids)]  # (M, O * 2) or (M, O)
    M, _ = ens_preds.shape

    if obj.noise == 'hetero':
        ens_preds = ens_preds.reshape(M, -1, 2)  # (M, O, 2)
        locs = ens_preds[:, :, 0]  # (M, O)
        log_scales = ens_preds[:, :, 1]  # (M, O)
        scales = jnp.exp(log_scales)
    elif obj.noise == 'per-ens-homo':
        locs = ens_preds
        scales = jnp.exp(obj.logscale)  # (M, O)
    else:
        locs = ens_preds
        scales = jnp.repeat(jnp.exp(obj.logscale)[jnp.newaxis, :], M, axis=0)  # (M, O)

    probs = nn.softmax(obj.weights[jnp.array(ensemble_ids)])[:, jnp.newaxis]

    return locs, scales, probs


def get_agg_fn(agg: str) -> Callable:
    raise_if_not_in_list(agg, ['mean', 'sum'], 'aggregation')

    if agg == 'mean':
        return jnp.mean
    else:
        return jnp.sum


def powerset(ens_size: int) -> List[List[int]]:
    ens_ids = [i for i in range(ens_size)]
    return list(map(list, list(chain.from_iterable(combinations(ens_ids, r) for r in range(1, len(ens_ids)+1)))))


def hardened_ovr_ll(y_1hot: Array, logits: Array, T: float, positive_class_only: bool = False):
    assert_rank(T, 0)
    assert_rank(y_1hot, 1)
    assert_equal_shape([y_1hot, logits])

    σ = nn.sigmoid(T * logits).clip(1e-6, 1 - 1e-6)
    if positive_class_only:
        res = jnp.sum(y_1hot * jnp.log(σ), axis=0)
    else:
        res = jnp.sum(y_1hot * jnp.log(σ) + (1 - y_1hot) * jnp.log(1 - σ), axis=0)
    return res


def softmax_ll(y: int, logits: Array):
    return distrax.Categorical(logits).log_prob(y)


def product_logprob_ovr(y: int, ens_logits: Array, β: float, probs: Array, N: int, positive_class_only: bool = False):
    y_1hot = jax.nn.one_hot(y, N)  # TODO: this would not work for pixelwise classification
    lls = jax.vmap(hardened_ovr_ll, in_axes=(None, 0, None, None))(y_1hot, ens_logits, β, positive_class_only)
    res = jnp.sum(probs * lls, axis=0)
    return res


def product_logprob_softmax(y: int, ens_logits: Array, probs: Array):
    lls = jax.vmap(softmax_ll, in_axes=(None, 0))(y, ens_logits)
    res = jnp.sum(probs * lls, axis=0)
    return res

