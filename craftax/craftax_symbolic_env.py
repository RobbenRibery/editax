from typing import  Dict
from collections import OrderedDict

import chex
from craftax.constants import *
from craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.common import log_achievements_to_info
from craftax.renderer import render_craftax_symbolic
from craftax.craftax_pixels_env import CraftaxPixelsEnv

from gymnax.environments import spaces

def get_map_obs_shape():
    num_mob_classes = 5
    num_mob_types = 8
    num_blocks = len(BlockType)
    num_items = len(ItemType)
    return (
        OBS_DIM[0],
        OBS_DIM[1],
        num_blocks + num_items + num_mob_classes * num_mob_types + 1,
    )


def get_flat_map_obs_shape():
    map_obs_shape = get_map_obs_shape()
    return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
    return 51


class CraftaxSymbolicEnv(CraftaxPixelsEnv):
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

    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        pixels = render_craftax_symbolic(state)
        return OrderedDict(dict(image=pixels))
    
    @property
    def name(self) -> str:
        return "CraftaxSymbolic"

    def observation_space(self,) -> spaces.Dict:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()
        obs_shape = flat_map_obs_shape + inventory_obs_shape
        space_dict = {
            "image": spaces.Box(
                0.0,
                1.0,
                (obs_shape,),
                dtype=jnp.float32,
            )
        }
        return spaces.Dict(space_dict)
    
    def get_episodic_metrics(self, state: EnvState) -> Dict[str, chex.Array]:
        return log_achievements_to_info(state, False)