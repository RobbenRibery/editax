from jax import lax
from typing import Tuple, Dict, Any
from collections import OrderedDict
import chex

from craftax.constants import *
from craftax.common import log_achievements_to_info
from craftax.game_logic import craftax_step, is_game_over
from craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.renderer import render_craftax_pixels
from craftax.world_gen.world_gen import generate_world

from gymnax.environments import environment, spaces


class CraftaxPixelsEnv(environment.Environment):

    def __init__(self, **env_kwargs,):
        super().__init__()

        # take the static env params
        static_env_params:StaticEnvParams = env_kwargs.pop(
            "static_env_params", 
            None
        ) 

        # assign static params 
        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

        # assign other params
        self.params = EnvParams(**env_kwargs) \
            if env_kwargs is not None \
            else self.default_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, 
        key: chex.PRNGKey, 
        state: EnvState, 
        action: int,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        state, reward = craftax_step(
            key, 
            state, 
            action, 
            self.params, 
            self.static_env_params
        )

        done = self.is_terminal(state) #dead | max stpes reached | beaten boss
        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, 
        rng: chex.PRNGKey,
    ) -> Tuple[chex.Array, EnvState]:
        
        state = generate_world(rng, self.params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> Dict[str,chex.Array]:
        pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT) / 255.0
        return OrderedDict(dict(image=pixels))

    def is_terminal(self, state: EnvState) -> bool:
        return is_game_over(state, self.params, self.static_env_params)

    @property
    def name(self) -> str:
        return "CraftaxPixels"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self,) -> spaces.Discrete:
        return spaces.Discrete(len(Action), dtype=jnp.int32)

    def observation_space(self,) -> spaces.Dict:
        spaces_dict = {
            "image" : spaces.Box(
                0.0,
                1.0,
                (
                    OBS_DIM[1] * BLOCK_PIXEL_SIZE_AGENT,
                    (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_AGENT,
                    3,
                ),
            )
        }
        return spaces.Dict(spaces_dict)
    
    def max_episode_steps(self) -> int:
        """Maximum number of time steps in environment."""
        return self.params.max_timesteps
    

    def get_env_metrics(self, state:EnvState) -> Dict[str, Any]:
        
        # track total number of mobs 
        total_mele_mobs = state.melee_mobs.position.shape[0]
        total_ranged_mobs = state.ranged_mobs.position.shape[0]
        total_passive_mobs = state.passive_mobs.position.shape[0]

        # track chracteristics of each mobs
        mean_mele_mobs_health = state.melee_mobs.health.mean()
        mean_ranged_mobs_health = state.melee_mobs.health.mean()
        mean_passive_mobs_health = state.melee_mobs.health.mean()

        mean_mele_mobs_cd = state.melee_mobs.attack_cooldown.mean()
        mean_ranged_mobs_cd = state.melee_mobs.attack_cooldown.mean()

        return dict(
            total_mele_mobs = total_mele_mobs,
            total_ranged_mobs = total_ranged_mobs,
            total_passive_mobs = total_passive_mobs,
            mean_mele_mobs_health = mean_mele_mobs_health,
            mean_ranged_mobs_health = mean_ranged_mobs_health,
            mean_passive_mobs_health = mean_passive_mobs_health, 
            mean_mele_mobs_cd = mean_mele_mobs_cd,
            mean_ranged_mobs_cd = mean_ranged_mobs_cd,
            light_level = state.light_level,
            passable = 1, #default to passable as there is no way to time
        )
