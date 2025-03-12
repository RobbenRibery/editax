# imports
import jax
import jax.numpy as jnp
import chex
from flax import struct
from jax import lax
# Import EnvState from the environment module (assuming it is on PYTHONPATH)
from kinetix.environment.env_state import EnvState

## Utility function to update a sub-structure (if needed)
def update_substate(state_sub, field, new_value):
    return state_sub.replace(**{field: new_value})

# mmps

@jax.jit
def mmp_enable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that eases control:
    Enable motor automatic control by setting all motor_auto flags to True.
    '''
    new_motor_auto = jnp.ones_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)

@jax.jit
def mmp_disable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that increases control difficulty:
    Disable motor automatic control by setting all motor_auto flags to False.
    '''
    new_motor_auto = jnp.zeros_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)

@jax.jit
def mmp_reduce_shape_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that eases the physical challenge:
    Reduce the densities of polygons and circles by a factor (e.g. 0.8).
    This reduces inertia and friction.
    '''
    factor = 0.8
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities = env_state.circle_densities * factor
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)

@jax.jit
def mmp_increase_shape_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that increases the physical challenge:
    Increase the densities of polygons and circles by a factor (e.g. 1.25).
    This increases inertia and friction.
    '''
    factor = 1.25
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities = env_state.circle_densities * factor
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)

@jax.jit
def mmp_lower_gravity(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that eases dynamics:
    Lower the gravity by scaling its vector (e.g. 0.5x).
    Assumes env_state has an attribute 'gravity'.
    '''
    new_gravity = env_state.gravity * 0.5
    return env_state.replace(gravity=new_gravity)

@jax.jit
def mmp_raise_gravity(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that increases the dynamic challenge:
    Increase the gravity by scaling its vector (e.g. 1.5x).
    Assumes env_state has an attribute 'gravity'.
    '''
    new_gravity = env_state.gravity * 1.5
    return env_state.replace(gravity=new_gravity)

@jax.jit
def mmp_offset_positions(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that slightly perturbs spatial configuration:
    Add a small random offset to the positions of all active polygons and circles.
    '''
    # Define magnitude of offset
    offset_scale = 0.05
    # Generate offsets for polygons (assume env_state.polygon.position is an array)
    poly_shape = env_state.polygon.position.shape
    poly_offset = jax.random.uniform(rng, shape=poly_shape, minval=-offset_scale, maxval=offset_scale)
    # Generate new polygon positions only for active shapes; use masking
    new_poly_position = jnp.where(
        env_state.polygon.active[..., None], 
        env_state.polygon.position + poly_offset, 
        env_state.polygon.position
    )
    
    # Split the rng for circle positions
    rng, subkey = jax.random.split(rng)
    circ_shape = env_state.circle.position.shape
    circ_offset = jax.random.uniform(subkey, shape=circ_shape, minval=-offset_scale, maxval=offset_scale)
    new_circ_position = jnp.where(
        env_state.circle.active[..., None], 
        env_state.circle.position + circ_offset, 
        env_state.circle.position
    )
    # Update state: assume polygon and circle are sub-states supporting .replace() method.
    new_polygon = env_state.polygon.replace(position=new_poly_position)
    new_circle = env_state.circle.replace(position=new_circ_position)
    return env_state.replace(polygon=new_polygon, circle=new_circle)

@jax.jit
def mmp_highlight_goal(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that eases perception:
    Highlight goal shapes by setting highlight flags (for shapes with role==2) to True.
    '''
    # For polygons: update polygon_highlighted array where shape role equals 2.
    new_poly_highlighted = jnp.where(env_state.polygon_shape_roles == 2, True, env_state.polygon_highlighted)
    new_circ_highlighted = jnp.where(env_state.circle_shape_roles == 2, True, env_state.circle_highlighted)
    return env_state.replace(polygon_highlighted=new_poly_highlighted,
                             circle_highlighted=new_circ_highlighted)

@jax.jit
def mmp_dim_goal(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that increases perceptual challenge:
    Remove highlights from goal shapes by setting highlight flags for shapes with role==2 to False.
    '''
    new_poly_highlighted = jnp.where(env_state.polygon_shape_roles == 2, False, env_state.polygon_highlighted)
    new_circ_highlighted = jnp.where(env_state.circle_shape_roles == 2, False, env_state.circle_highlighted)
    return env_state.replace(polygon_highlighted=new_poly_highlighted,
                             circle_highlighted=new_circ_highlighted)

@jax.jit
def mmp_randomize_roles(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Minimal Perturbation that increases environmental unpredictability:
    Randomizes the roles of all shapes (polygons and circles) by assigning a random role in {0,1,2,3}.
    '''
    num_roles = 4
    # Randomize roles for polygons and circles independently.
    new_polygon_roles = jax.random.randint(rng, env_state.polygon_shape_roles.shape, 0, num_roles)
    # Split rng for circles
    rng, subkey = jax.random.split(rng)
    new_circle_roles = jax.random.randint(subkey, env_state.circle_shape_roles.shape, 0, num_roles)
    return env_state.replace(polygon_shape_roles=new_polygon_roles,
                             circle_shape_roles=new_circle_roles)