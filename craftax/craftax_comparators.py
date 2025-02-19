import jax.numpy as jnp 
import jax 

import operator

from craftax import EnvState

@jax.jit
def is_equl_word(a:EnvState, b:EnvState) -> int:
    """
    Checks if two EnvStates are equal by comparing 
    all the elements in the EnvStates as arrays.
    Returns a boolean scalar indicating whether the two EnvStates are equal.
    """
    eq_tree:EnvState = jax.tree_map(
        lambda x, y: jnp.equal(x,y).all(),
        a, 
        b,
    )
    return jax.tree_util.tree_reduce(
        operator.eq, 
        eq_tree,
    )