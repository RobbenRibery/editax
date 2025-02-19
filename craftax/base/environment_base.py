import jax
import chex
from typing import Union
from functools import partial

import pickle
import bz2
from typing import Any


def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title, "w") as f:
        pickle.dump(data, f)


def load_compressed_pickle(file: str):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data


class EnvironmentNoAutoReset(object):
    """Similar to the base Gymnax environment but without auto-resets."""

    @property
    def default_params(self):
        return NotImplementedError

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state, reward, done, info = self.step_env(key, state, action, params)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params,
    ):
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey, params):
        """Environment-specific reset."""
        raise NotImplementedError

    def get_obs(self, state) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state, params) -> bool:
        """Check whether state transition is terminal."""
        raise NotImplementedError

    def discount(self, state, params) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self, params):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params):
        """State space of the environment."""
        raise NotImplementedError