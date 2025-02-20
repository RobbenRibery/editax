import os 
import json 

import jax 
import jax.numpy as jnp
import chex 

from editax.moed import EditorManager
from editax.utils import code_utils_test_editors

from kinetix.environment.env import make_kinetix_env_from_args
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.environment.ued.distributions import sample_kinetix_level

from dotenv import load_dotenv
load_dotenv()

from pprint import pprint

env_params = EnvParams()
static_env_params = StaticEnvParams()
ued_params = UEDParams()

with open("experiments/kinetix.json", "r") as f:
    config = json.load(f)

editor_manager = EditorManager(**config)


if __name__ == "__main__":

    # Create the environment
    env = make_kinetix_env_from_args(
        obs_type="pixels", 
        action_type="multidiscrete", 
        reset_type="replay", 
        static_env_params=static_env_params,
    )

    # Sample a level
    rng = jax.random.PRNGKey(0)
    rng, rng_editor = jax.random.split(rng)
    level = sample_kinetix_level(
        rng,
        env.physics_engine,
        ued_params=ued_params,
        static_env_params=static_env_params,
        env_params=env_params,
    )
    
    num_inner_loops = 8
    # generate editors
    func_map, editor_buffer = editor_manager.reset(
        rng_editor,
        num_inner_loops=num_inner_loops,
        dummy_env_state=level,
        correction_only=True,
    )

    pprint(func_map)