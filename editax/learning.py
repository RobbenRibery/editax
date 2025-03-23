import jax
import jax.numpy as jnp
import chex
from enum import IntEnum

from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
from editax.upomdp import EnvState, UnderspecifiedEnv

from typing import Tuple, List, Callable
from functools import partial

class EditorPolicyTrainState(BaseTrainState):
    num_updates:int

def update_editr_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: EditorPolicyTrainState,
    init_hstate: chex.ArrayTree,
    batch: Tuple[chex.Array],
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    kl_coeff: float,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, EditorPolicyTrainState], chex.ArrayTree]:

    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            (
                init_hstate,
                obs,
                actions,
                last_dones,
                log_probs,
                values,
                targets,
                advantages,
            ) = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(
                    params, init_hstate, (obs, last_dones)
                )
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                l_clip = (
                    -jnp.minimum(
                        ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
                    )
                ).mean()

                values_pred_clipped = values + (values_pred - values).clip(
                    -clip_eps, clip_eps
                )
                l_vf = (
                    0.5
                    * jnp.maximum(
                        (values_pred - targets) ** 2,
                        (values_pred_clipped - targets) ** 2,
                    ).mean()
                )
                # -------------------------------------------------------------------
                # KL penalty to keep the policy from deviating from a random
                # (uniform) distribution. Suppose pi has shape (..., n_actions) for
                # discrete actions. We'll compute the KL from uniform:
                #    KL(pi || uniform) = sum_a pi(a) * log[ pi(a) * n_actions ]
                # This will be 0 if pi is uniform, and higher if pi is more peaked.
                # kl_coeff is a new hyperparameter.
                # -------------------------------------------------------------------
                n_actions = pi.probs.shape[-1]
                # Small epsilon to avoid numerical issues in log
                kl_random = jnp.sum(
                    pi.probs * jnp.log(
                        pi.probs / ((1 / n_actions) + 1e-8)
                    ), 
                    axis=-1
                ).mean()

                loss = l_clip + critic_coeff * l_vf + kl_coeff * kl_random

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            grad_norm = jnp.linalg.norm(
                jnp.concatenate(
                    jax.tree_util.tree_map(
                        lambda x: x.flatten(), jax.tree_util.tree_flatten(grads)[0]
                    )
                )
            )
            return train_state, (loss, grad_norm)

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(
                    n_minibatch, -1, *x.shape[1:]
                ),
                init_hstate,
            ),
            *jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, (losses, grads) = jax.lax.scan(
            update_minibatch, train_state, minibatches
        )
        return (rng, train_state), (losses, grads)

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)
