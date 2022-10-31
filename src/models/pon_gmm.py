from typing import Any, Callable, Mapping, Optional
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from distrax import MixtureSameFamily
from flax.linen import initializers
from chex import Array
import distrax
import matplotlib.pyplot as plt

from src.models.common import raise_if_not_in_list, NOISE_TYPES, get_locs_scales_probs, get_agg_fn
from src.models.resnet import ResNet


KwArgs = Mapping[str, Any]


def pon_gmm_ll(y, loc, scale, K):
    # TODO: get MixtureSameFamily to work with batches
    per_dim_lls = MixtureSameFamily(mixture_distribution=distrax.Categorical(probs=jnp.ones(int(K)) / K),
                                    components_distribution=distrax.Normal(loc=loc, scale=scale)).log_prob(y)
    print("BANANA")
    print(per_dim_lls.shape)
    print(per_dim_lls)
    return jnp.sum(per_dim_lls, axis=0, keepdims=True)


class PoN_Ens_GMM(nn.Module):
    """Ens trained as a Product of Normals."""
    size: int
    net: KwArgs
    weights_init: Callable = initializers.ones
    logscale_init: Callable = initializers.zeros
    noise: str = 'homo'
    learn_weights: bool = False
    alpha: float = 1.0
    K: int = 2
    learn_weights_gmm: bool = False

    def setup(self):
        raise_if_not_in_list(self.noise, NOISE_TYPES, 'self.noise')

        self.nets = [ResNet(**self.net) for _ in range(self.size)]
        weights = self.param(
            'weights',
            self.weights_init,
            (self.size,)
        )
        self.weights = weights if self.learn_weights else jax.lax.stop_gradient(weights)
        if self.noise != 'hetero':
            self.logscale = self.param(
                'logscale',
                self.logscale_init,
                (int(self.net['out_size'] / self.K,)) if self.noise == 'homo' else (self.size, self.net['out_size'],)
            )

    def __call__(
        self,
        x: Array,
        y: int,
        train: bool = False,
    ) -> Array:
        locs, scales, probs = get_locs_scales_probs_pon_gmm(self, x, train)
        print(y)
        print(x.shape)
        print(locs.shape)
        print(scales.shape)
        print(probs.shape)

        def product_logprob(y):
            prod_lls = jax.vmap(pon_gmm_ll, in_axes=(None, 0, 0, None))(y, locs, scales, self.K)
            return jnp.sum(prod_lls)
        dy = 0.001
        ys = jnp.arange(-10, 10 + dy, dy)
        ps = jnp.exp(jax.vmap(product_logprob)(ys))
        print(ps)
        Z = jnp.trapz(ps, ys)
        log_prob = product_logprob(y)
        nll = -(log_prob - self.alpha * jnp.log(Z + 1e-36))

        y_pred = (ps * ys) / Z
        err = jnp.mean((y_pred - y)**2)

        print(nll)
        print(err)

        return nll, err, nll, nll

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ) -> Array:
        # TODO: adjust
        locs, scales, probs = get_locs_scales_probs(self, x, train)

        loc, scale = normal_prod(locs, scales, probs)

        if return_ens_preds:
            return (loc, scale), (locs, scales)
        else:
            return (loc, scale)


def get_locs_scales_probs_pon_gmm(
    obj,
    x: Array,
    train: bool = False,
):
    ens_preds = jnp.stack([net(x, train=train) for net in obj.nets], axis=0)  # (M, O * 2) or (M, O)
    M, _ = ens_preds.shape

    if obj.noise == 'homo':
        locs = ens_preds
        scales = jnp.repeat(jnp.exp(obj.logscale)[jnp.newaxis, :], M, axis=0)  # (M, O)
    else:
        raise NotImplementedError()

    probs = nn.softmax(obj.weights)[:, jnp.newaxis]

    return locs, scales, probs


def make_PoN_Ens_GMM_loss(
    model: PoN_Ens_GMM,
    x_batch: Array,
    y_batch: Array,
    train: bool = True,
    aggregation: str = 'mean',
) -> Callable:
    """Creates a loss function for training a PoE Ens."""
    def batch_loss(params, state, rng):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            (nll, err, prod_ll, members_ll), new_state = model.apply(
                {"params": params, **state}, x, y, train=train,
                mutable=list(state.keys()) if train else {},
            )

            return nll, new_state, err, prod_ll, members_ll

        # broadcast over batch and aggregate
        agg = get_agg_fn(aggregation)
        loss_for_batch, new_state, err_for_batch, prod_ll_for_batch, members_ll_for_batch = jax.vmap(
            loss_fn, out_axes=(0, None, 0, 0, 0), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return agg(loss_for_batch, axis=0), (new_state, agg(err_for_batch, axis=0), agg(prod_ll_for_batch, axis=0), agg(members_ll_for_batch, axis=0))

    return batch_loss


def normal_prod(locs, scales, probs):
    scales2 = scales ** 2
    θ_1 = ((locs / scales2) * probs).sum(axis=0)
    θ_2 = ((-1 / (2 * scales2)) * probs).sum(axis=0)

    σ2 = -1 / (2 * θ_2)
    scale = jnp.sqrt(σ2)
    loc = θ_1 * σ2

    return loc, scale


def make_PoN_Ens_GMM_plots(
    pon_model, pon_state, pon_tloss, pon_vloss, X_train, y_train, X_val, y_val,
    ):
    pon_params, pon_model_state = pon_state.params, pon_state.model_state

    n_plots = 2
    fig, axs = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 6))

    xs = jnp.linspace(-3, 3, num=501)

    # pon preds
    pred_fun = partial(
        pon_model.apply,
        {"params": pon_params, **pon_model_state},
        train=False, return_ens_preds=True,
        method=pon_model.pred
    )
    (loc, scale), (locs, scales) = jax.vmap(
        pred_fun, out_axes=(0, 1), in_axes=(0,), axis_name="batch"
    )(xs.reshape(-1, 1))

    size = locs.shape[0]

    loc = loc[:, 0]
    scale = scale[:, 0]
    locs = locs[:, :, 0]
    scales = scales[:, :, 0]

    axs[0].scatter(X_train, y_train, c='C0')
    for i in range(size):
        axs[0].plot(xs, locs[i], c='k', alpha=0.25)

    axs[0].plot(xs, loc, c='C1')
    axs[0].fill_between(xs, loc - scale, loc + scale, color='C1', alpha=0.4)

    axs[0].set_title(f"PoN Ens - train loss: {pon_tloss:.6f}, val loss: {pon_vloss:.6f}")
    axs[0].set_ylim(-2.5, 2.5)
    axs[0].set_xlim(-3, 3)


    # plot locs and scales for each member
    axs[1].scatter(X_train, y_train, c='C0')
    for i in range(size):
        axs[1].plot(xs, locs[i], alpha=0.5)
        axs[1].fill_between(xs, locs[i] - scales[i], locs[i] + scales[i], alpha=0.1)
    axs[1].set_title(f"PoN Members")
    axs[1].set_ylim(-2.5, 2.5)
    axs[1].set_xlim(-3, 3)

    plt.show()

    return fig
