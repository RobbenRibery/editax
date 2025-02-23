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


@partial(jax.jit, static_argnums=(6, 7))
def sample_edit_trajectories_rnn(
    rng: chex.PRNGKey,
    train_state: EditorPolicyTrainState,
    init_hstate: chex.ArrayTree,
    init_env_state: EnvState,
    num_envs: int,
    editors: List[Callable],
    num_edits: int = 8,
    edit_eps_length: int = 256,
) -> Tuple[
    Tuple[
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    Tuple[
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
]:
    """
    This function samples a trajectory using the given policy and a list of editors. 
    The trajectory is sampled for a total of `edit_eps_length` steps. 
    The function returns a tuple of two elements. 
    The first element is a tuple of five elements, 
        which are the random number generator, the policy, the last hidden state, the last environment state, and the last value. 
    The second element is a tuple of four elements, 
        which are the editor indicies, the done flags, the log probabilities, and the values.

    Args:
        rng: The random number generator.
        train_state: The policy.
        init_hstate: The initial hidden state.
        init_env_state: The initial environment state.
        num_envs: The number of environments to sample from.
        editors: A list of editors to use.
        num_edits: The number of edits to perform before returning a new trajectory.
        edit_eps_length: The total number of steps to sample for.

    Returns:
        A tuple of two elements. 
        The first element is a tuple of five elements, 
            which are the random number generator, the policy, the last hidden state, the last environment state, and the last value. 
        The second element is a tuple of four elements, 
            which are the editor indicies, the done flags, the log probabilities, and the values.
    """

    def sample_step(carry: Tuple, _) -> Tuple:
        """
        This function is used to sample a single step in the trajectory.
        """
        rng, train_state, hstate, env_state, edited_steps, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # get the editor indicies from the policy
        x = jax.tree_util.tree_map(
            lambda x: x[jnp.newaxis, ...],  # create one addituibak axus ti natcg tune
            (env_state, last_done),
        )
        print(x[0].shape)
        print(x[1].shape)
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        editor_idx = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(editor_idx)
        value, editor_idx, log_prob = (
            value.squeeze(0),
            editor_idx.squeeze(0),
            log_prob.squeeze(0),
        )

        # apply the editor across all envs
        next_env_state = jax.vmap(
            lambda editor_idx, rng_env, env_state: jax.lax.switch(
                editor_idx,
                editors,
                *(
                    rng_env,
                    env_state,
                ),
            )
        )(editor_idx, jax.random.split(rng_step, num_envs), env_state)
        # update the edited steps
        edited_steps += 1
        #
        done = jnp.where(
            edited_steps % num_edits == 0,
            jnp.ones((num_envs,), dtype=jnp.bool),
            jnp.zeros((num_envs,), dtype=jnp.bool),
        )

        carry = (rng, train_state, hstate, next_env_state, done)
        step = (editor_idx, done, log_prob, value)
        return carry, step

    edited_steps = 0
    (rng, train_state, hstate, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_env_state,
            edited_steps,
        ),
        None,
        length=edit_eps_length,
    )

    x = jax.tree_util.tree_map(
        lambda x: x[jnp.newaxis, ...], (last_env_state, last_done)
    )
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (
        rng,
        train_state,
        hstate,
        last_env_state,
        last_value.squeeze(0),
    ), traj


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

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy + kl_coeff * kl_random

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
