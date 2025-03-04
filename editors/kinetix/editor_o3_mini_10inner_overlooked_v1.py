from typing import List
# imports 
import jax
import jax.numpy as jnp
import chex
from flax import struct
from jax import lax

# Importing environment state and needed utilities from the provided source code.
from kinetix.environment.env_state import EnvState  # type: ignore

##############################
## Utility Helper Functions ##
##############################

def shift_positions(positions: jnp.ndarray, active_mask: jnp.ndarray, delta: float):
    """
    Shifts positions for active shapes outward (if delta > 0) or inward (if delta < 0)
    by moving them a fraction delta in the normalized direction from the origin.
    """
    # Compute norms for each position with a small epsilon to avoid division by zero.
    norms = jnp.linalg.norm(positions, axis=-1, keepdims=True) + 1e-6
    directions = positions / norms
    # Expand active_mask to shape (n,1)
    active_mask_expanded = active_mask[..., None].astype(jnp.float32)
    return positions + active_mask_expanded * delta * directions

def get_random_index(rng: chex.PRNGKey, mask: jnp.ndarray) -> tuple[chex.PRNGKey, int]:
    """
    Returns a random index from those positions where mask is True.
    If no candidate exists, returns -1.
    This corrected implementation avoids using jnp.nonzero by sampling directly
    using a probability distribution computed from the mask.
    """
    total = jnp.sum(mask).astype(jnp.float32)
    rng, subkey = jax.random.split(rng)
    selected = lax.cond(
        total > 0,
        lambda: jax.random.choice(
            subkey,
            a=mask.shape[0],
            p=mask.astype(jnp.float32) / total
        ),
        lambda: -1
    )
    return rng, selected

######################
## MMP Definitions  ##
######################

@jax.jit
def mmp_enable_auto_control(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Sets all motor_auto flags to True to ease the control challenge by enabling automatic motor control.
    """
    new_motor_auto = jnp.ones_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)


@jax.jit
def mmp_disable_auto_control(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Sets all motor_auto flags to False to increase control challenge by requiring manual motor control.
    """
    new_motor_auto = jnp.zeros_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)


@jax.jit
def mmp_reduce_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Reduces the densities of both polygons and circles by a factor of 0.5, making objects lighter and easier to move.
    """
    new_polygon_densities = env_state.polygon_densities * 0.5
    new_circle_densities = env_state.circle_densities * 0.5
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)


@jax.jit
def mmp_increase_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Increases the densities of both polygons and circles by a factor of 1.5, making objects heavier and increasing inertia.
    """
    new_polygon_densities = env_state.polygon_densities * 1.5
    new_circle_densities = env_state.circle_densities * 1.5
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)


@jax.jit
def mmp_deactivate_random_polygon(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Deactivates one randomly chosen polygon (excluding the floor at index 0)
    to reduce environmental clutter and lower collision dynamics.
    """
    # Create a mask: polygon is active and not the floor (assumed at index 0)
    num_polys = env_state.polygon_densities.shape[0]
    indices = jnp.arange(num_polys)
    # Assuming the floor is at index 0; do not select it.
    mask = env_state.polygon_densities > 0  # Using density as proxy for presence/activity.
    mask = mask & (indices != 0)
    rng, selected_idx = get_random_index(rng, mask)
    # If no candidate selected, do nothing.
    def deactivate(idx, state):
        # Assume there is an array 'polygon.active' indicating active status.
        new_active = state.polygon.active.at[idx].set(False)
        return state.replace(polygon=state.polygon.replace(active=new_active))
    new_state = lax.cond(selected_idx >= 0,
                         lambda: deactivate(selected_idx, env_state),
                         lambda: env_state)
    return new_state


@jax.jit
def mmp_activate_random_polygon(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Activates one randomly chosen polygon that is currently inactive,
    increasing the environmental challenge by introducing an extra obstacle.
    """
    num_polys = env_state.polygon_densities.shape[0]
    indices = jnp.arange(num_polys)
    # Create a mask: polygon inactive.
    mask = jnp.logical_not(env_state.polygon.active)
    rng, selected_idx = get_random_index(rng, mask)
    def activate(idx, state):
        new_active = state.polygon.active.at[idx].set(True)
        return state.replace(polygon=state.polygon.replace(active=new_active))
    new_state = lax.cond(selected_idx >= 0,
                         lambda: activate(selected_idx, env_state),
                         lambda: env_state)
    return new_state


@jax.jit
def mmp_increase_spacing(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Increases spacing by shifting positions of active polygons and circles 0.1 units away from the origin,
    thereby reducing collision likelihood.
    """
    # Update polygon positions.
    new_poly_positions = shift_positions(env_state.polygon.position, env_state.polygon.active, 0.1)
    # Update circle positions.
    new_circle_positions = shift_positions(env_state.circle.position, env_state.circle.active, 0.1)
    new_state = env_state.replace(
        polygon=env_state.polygon.replace(position=new_poly_positions),
        circle=env_state.circle.replace(position=new_circle_positions),
    )
    return new_state


@jax.jit
def mmp_decrease_spacing(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Decreases spacing by shifting positions of active polygons and circles 0.1 units toward the origin,
    thereby increasing the likelihood of collisions.
    """
    new_poly_positions = shift_positions(env_state.polygon.position, env_state.polygon.active, -0.1)
    new_circle_positions = shift_positions(env_state.circle.position, env_state.circle.active, -0.1)
    new_state = env_state.replace(
        polygon=env_state.polygon.replace(position=new_poly_positions),
        circle=env_state.circle.replace(position=new_circle_positions),
    )
    return new_state


@jax.jit
def mmp_reduce_negative_roles(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Reassigns one randomly selected shape that has a negative role (role 3) to a safer role (role 2),
    easing potential penalties.
    """
    # For polygons: search among active ones with role == 3.
    poly_mask = (env_state.polygon_shape_roles == 3) & env_state.polygon.active
    rng, poly_idx = get_random_index(rng, poly_mask)
    new_state = env_state
    new_state = lax.cond(poly_idx >= 0,
                         lambda: new_state.replace(polygon_shape_roles=new_state.polygon_shape_roles.at[poly_idx].set(2)),
                         lambda: new_state)
    # For circles if no polygon candidate (optional: try circles only if no change was made)
    poly_candidate_exists = jnp.any(poly_mask)
    new_state = lax.cond(~poly_candidate_exists,
                         lambda: new_state.replace(circle_shape_roles=jnp.where(new_state.circle_shape_roles==3, 2, new_state.circle_shape_roles)),
                         lambda: new_state)
    return new_state


@jax.jit
def mmp_increase_negative_roles(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Converts one randomly selected shape that is safe (role 2) into a negative role (role 3),
    increasing penalties or hazards.
    """
    poly_mask = (env_state.polygon_shape_roles == 2) & env_state.polygon.active
    rng, poly_idx = get_random_index(rng, poly_mask)
    new_state = env_state
    new_state = lax.cond(poly_idx >= 0,
                         lambda: new_state.replace(polygon_shape_roles=new_state.polygon_shape_roles.at[poly_idx].set(3)),
                         lambda: new_state)
    poly_candidate_exists = jnp.any(poly_mask)
    new_state = lax.cond(~poly_candidate_exists,
                         lambda: new_state.replace(circle_shape_roles=jnp.where(new_state.circle_shape_roles==2, 3, new_state.circle_shape_roles)),
                         lambda: new_state)
    return new_state


@jax.jit
def mmp_increase_friction(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Increases friction by multiplying the friction coefficients of active polygons and circles by 1.5,
    making motion "stickier" and more challenging.
    """
    new_poly_friction = env_state.polygon.friction * (1.5 * env_state.polygon.active.astype(jnp.float32) + 
                                                     (1 - env_state.polygon.active.astype(jnp.float32)))
    new_circle_friction = env_state.circle.friction * (1.5 * env_state.circle.active.astype(jnp.float32) + 
                                                       (1 - env_state.circle.active.astype(jnp.float32)))
    new_state = env_state.replace(
        polygon=env_state.polygon.replace(friction=new_poly_friction),
        circle=env_state.circle.replace(friction=new_circle_friction)
    )
    return new_state


@jax.jit
def mmp_decrease_friction(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Decreases friction by multiplying the friction coefficients of active polygons and circles by 0.5,
    making movement smoother and easier.
    """
    new_poly_friction = env_state.polygon.friction * (0.5 * env_state.polygon.active.astype(jnp.float32) + 
                                                     (1 - env_state.polygon.active.astype(jnp.float32)))
    new_circle_friction = env_state.circle.friction * (0.5 * env_state.circle.active.astype(jnp.float32) + 
                                                       (1 - env_state.circle.active.astype(jnp.float32)))
    new_state = env_state.replace(
        polygon=env_state.polygon.replace(friction=new_poly_friction),
        circle=env_state.circle.replace(friction=new_circle_friction)
    )
    return new_state