from typing import Any, Callable, Mapping, List
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array

from src.models.common import get_agg_fn, product_logprob_ovr, product_logprob_softmax
from src.models.resnet import ResNet
from src.models.convnet import ConvNet


KwArgs = Mapping[str, Any]


class Cls_Ens(nn.Module):
    """A standard classification ensemble."""
    size: int
    net: KwArgs
    weights_init: Callable = initializers.ones
    logscale_init: Callable = initializers.zeros
    learn_weights: bool = False
    ll_type: str = "softmax"   # softmax or ovr
    net_type: str = "ResNetMLP"

    def setup(self):
        if self.net_type == "ResNetMLP":
            self.nets = [ResNet(**self.net) for _ in range(self.size)]
        elif self.net_type == "ConvNet":
            self.nets = [ConvNet() for _ in range(self.size)]
        else:
            raise ValueError()
        weights = self.param(
            'weights',
            self.weights_init,
            (self.size,)
        )
        self.weights = weights if self.learn_weights else jax.lax.stop_gradient(weights)

    def __call__(
        self,
        x: Array,
        y: int,
        train: bool = False,
        β: float = 1.
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (M, O)
        probs = nn.softmax(self.weights, axis=0)  # (M,)

        n_classes = self.net['out_size']

        if self.ll_type == "ovr":
            product_logprob = partial(product_logprob_ovr, ens_logits=ens_logits,
                                      probs=probs, N=n_classes,  β=β)
        elif self.ll_type == "softmax":
            product_logprob = partial(product_logprob_softmax, ens_logits=ens_logits, probs=probs)
        else:
            raise ValueError()

        prod_ll = product_logprob(y)
        loss = -1. * prod_ll

        pred = nn.softmax(ens_logits.mean(axis=0))
        err = y != jnp.argmax(pred, axis=0)

        return loss, err, 1., prod_ll

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (M, O)
        probs = nn.softmax(self.weights, axis=0)[:, jnp.newaxis]  # (M, 1)

        logits = (probs * ens_logits).sum(axis=0)  # (O,)
        preds = nn.softmax(logits)

        if return_ens_preds:
            return preds, nn.softmax(ens_logits, axis=-1)
        else:
            return preds

    def ens_logits(
        self,
        x: Array,
        train: bool = False,
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (M, O)
        return ens_logits


def make_Cls_Ens_loss(
    model: Cls_Ens,
    x_batch: Array,
    y_batch: Array,
    train: bool = True,
    aggregation: str = 'mean',
    ensemble_ids: List[int] = (0, 1, 2, 3, 4,),
    alpha: float = 0.,
    β: float = 1.,
) -> Callable:
    """Creates a loss function for training a std Ens."""
    def batch_loss(params, state, rng):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            (loss, err, prod_ll, members_ll), new_state = model.apply(
                {"params": params, **state}, x, y, train=train, β=β,
                mutable=list(state.keys()) if train else {},
                rngs={'dropout': rng},
            )

            return loss, new_state, err, prod_ll, members_ll

        # broadcast over batch and aggregate
        agg = get_agg_fn(aggregation)
        loss_for_batch, new_state, err_for_batch, prod_ll_for_batch, members_ll_for_batch = jax.vmap(
            loss_fn, out_axes=(0, None, 0, 0, 0), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return agg(loss_for_batch, axis=0), (new_state, agg(err_for_batch, axis=0), agg(prod_ll_for_batch, axis=0), agg(members_ll_for_batch, axis=0))

    return batch_loss
