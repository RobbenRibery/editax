"""
Code reference to https://github.com/DramaCow/jaxued/blob/main/src/jaxued/environments/underspecified_env.py
"""
from typing import Any, Tuple, Optional, Union
import jax
import chex
from functools import partial
from flax import struct

@struct.dataclass
class EnvState:
    pass

@struct.dataclass
class Observation:
    pass

@struct.dataclass
class Level:
    pass

@struct.dataclass
class EnvParams:
    pass

@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int



class UnderspecifiedEnv(object):
    """
    The UnderspecifiedEnv class defines a UPOMDP, and acts similarly to (but not identically to) a Gymnax environment.

    The UnderspecifiedEnv class has the following interface:
        * `params = env.default_params`
        * `action_space = env.action_space(params)`
        * `obs, state = env.reset_to_level(rng, level, params)`
        * `obs, state, reward, done, info = env.step(rng, state, action, params)`

    Every environment must implement only the following methods:
        * `step_env`: Perform a step of the environment
        * `reset_env_to_level`: Reset the environment to a particular level
        * `action_space`: Return the action space of the environment
            
    The environment also does not automatically reset to a new level once the environment has restarted. 
    Look at the `AutoReplay` wrapper if this is desired.
    """
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        if params is None:
            params = self.default_params
        return self.step_env(rng, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: Optional[EnvParams] = None
    ) -> Tuple[Observation, EnvState]:
        if params is None:
            params = self.default_params
        return self.reset_env_to_level(rng, level, params)

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        raise NotImplementedError

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        raise NotImplementedError

    def action_space(self, params: EnvParams) -> Any:
        raise NotImplementedError
    


class LogWrapper(UnderspecifiedEnv):
    """Log the episode returns, lengths and achievements."""

    def __init__(self, env):
        self._env = env

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    @partial(jax.jit, static_argnums=(0, ))
    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        state = LogEnvState(level, 0.0, 0, 0.0, 0, 0)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action, params
        )
        env_state:EnvState

        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["achievements"] = env_state.achievements
        info["achievement_count"] = env_state.achievements.sum()

        if hasattr(env_state, "player_level"):
            info["floor"] = env_state.player_level
        return obs, state, reward, done, info

    def get_obs(self, state: LogEnvState) -> chex.Array:
        return self._env.get_obs(state.env_state)

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)