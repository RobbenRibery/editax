import os 
import json 

import jax 
import jax.numpy as jnp
import chex 

from editax.moed import EditorManager
from editax.utils import code_utils_test_editors

# you must have installed kinetix first
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
    config:dict = json.load(f)

num_inner_loops = config.pop("num_inner_loops", 10)
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
    
    # generate editors
    func_map = editor_manager.reset(
        level,
        num_inner_loops,
    )

    rng, sample_rng = jax.random.split(rng)
    editor_indicies = editor_manager.samle_random_edit_seqs(sample_rng)

    rng, edits_rng = jax.random.split(sample_rng)
    edited_level = editor_manager.perform_random_edit_seqs(edits_rng, level, editor_indicies)

