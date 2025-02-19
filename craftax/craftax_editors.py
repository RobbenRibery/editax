from typing import Callable, List, Optional

import chex 

import numpy as np
import jax 
import jax.numpy as jnp

from .craftax_state import EnvState, EnvParams

def seq_edits(
    rng: chex.PRNGKey,
    env_params:EnvParams,
    state: EnvState,
    mutators_collection: List[Callable],
    n: int = 1,
    mutators_prob: Optional[chex.Array] = None,
    ) -> EnvState:

    if n == 0:
        return jnp.full((1,), -1, dtype=jnp.int32), state

    # Step function for jax.lax.scan
    def step_fn(carry, pair_):

        idx = pair_
        rng, current_env = carry
        
        # Split the RNG for the next step
        rng, arng, _ = jax.random.split(rng, 3)

        # Use jax.lax.switch to select the mutator 
        # function based on the index
        new_env = jax.lax.switch(
            idx, 
            mutators_collection, 
            *(
                arng, 
                current_env, 
            ),
            )
        # Return the new carry (rng, new_env) and
        # the mutated environment for tracking
        return (rng, new_env,), new_env

    # Create the initial state
    rng, nrng = jax.random.split(rng)

    # Create the indices for mutators collection 
    # (scan needs a list to scan over)
    mutator_indices = jax.random.choice(
        nrng, 
        np.arange(len(mutators_collection)), 
        (n,), 
        p=mutators_prob,
    )
    # Use lax.scan to iteratively apply all mutators
    initial_carry = (rng, state)  # Initial rng and environment

    # lax.scan applies step_fn over 
    # mutators_collection and tracks all mutated environments
    final_carry, _ = jax.lax.scan(
        step_fn, 
        initial_carry, 
        (mutator_indices),
    )
    rng, final_env_state = final_carry
    return mutator_indices, final_env_state
