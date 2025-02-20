# Imports
import jax
import jax.numpy as jnp
import chex
from flax import struct

# Explicit imports for environment state and assumed substructures.
from kinetix.environment.env_state import EnvState

# ------------------------------------------------------------------------------
# Utility functions (if needed)
# ------------------------------------------------------------------------------
# Assume that env_state has sub-fields 'motor_auto', 'polygon_densities', 'circle_densities'.
# Also assume that env_state.polygon and env_state.circle are dataclasses with a 'position' field
# and a replace() method for immutable updates.
#
# For the purpose of these MMPs, we assume that update semantics with .replace() are valid.

# ------------------------------------------------------------------------------
# MMP Functions
# ------------------------------------------------------------------------------

@jax.jit
def mmp_enable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Enable automatic motor control by setting all motor_auto flags to True.
    This reduces the control challenge by relieving the agent of the need to manually actuate joints.
    """
    new_motor_auto = jnp.ones_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)


@jax.jit
def mmp_disable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Disable automatic motor control by setting all motor_auto flags to False.
    This increases the challenge by forcing the agent to control all joint motors manually.
    """
    new_motor_auto = jnp.zeros_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)


@jax.jit
def mmp_reduce_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Reduce the densities of polygon and circle shapes by a fixed factor.
    Lower densities imply lower inertia and softer collisions, easing the dynamics.
    """
    factor = 0.8  # Reduce densities by 20%
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities = env_state.circle_densities * factor
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)


@jax.jit
def mmp_increase_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Increase the densities of polygon and circle shapes by a fixed factor.
    Higher densities make objects "heavier," thus increasing inertia and the challenge in handling collisions.
    """
    factor = 1.2  # Increase densities by 20%
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities = env_state.circle_densities * factor
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)


@jax.jit
def mmp_increase_spacing(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Increase the spacing between shapes by adding a small positive offset to their positions.
    This offset reduces the chance of collisions, thereby making the environment easier.
    """
    # Generate an offset vector using the provided rng key.
    offset = jax.random.uniform(rng, shape=(2,), minval=0.05, maxval=0.1)
    # Update positions of polygons and circles (assuming these have a 'position' field).
    new_polygon = env_state.polygon.replace(position=env_state.polygon.position + offset)
    new_circle = env_state.circle.replace(position=env_state.circle.position + offset)
    return env_state.replace(polygon=new_polygon, circle=new_circle)


@jax.jit
def mmp_cluster_shapes(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Cluster shapes by subtracting a small offset from their positions.
    This reduction in spacing increases the likelihood of collisions, thus increasing the challenge.
    """
    offset = jax.random.uniform(rng, shape=(2,), minval=0.05, maxval=0.1)
    new_polygon = env_state.polygon.replace(position=env_state.polygon.position - offset)
    new_circle = env_state.circle.replace(position=env_state.circle.position - offset)
    return env_state.replace(polygon=new_polygon, circle=new_circle)