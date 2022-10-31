from typing import Callable, List, Tuple, Optional
from enum import Enum

import distrax
import jax
import jax.numpy as jnp
import flax.linen as nn
from chex import Array

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

