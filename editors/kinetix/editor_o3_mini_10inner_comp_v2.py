# Imports
import jax
import jax.numpy as jnp
import chex
from flax import struct

# Import the environment state from kinetix package
from kinetix.environment.env_state import EnvState

# Utility function for immutable state update using replace method
def update_env_state(env_state: EnvState, **kwargs) -> EnvState:
    return env_state.replace(**kwargs)

# MMP functions

@jax.jit
def mmp_enable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Enable automatic control for all motors.
    Reduces difficulty by bypassing manual control requirements.
    '''
    # Set motor_auto flag to True for all joints.
    new_motor_auto = jnp.ones_like(env_state.motor_auto, dtype=bool)
    return update_env_state(env_state, motor_auto=new_motor_auto)

@jax.jit
def mmp_disable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Disable automatic control for all motors.
    Increases challenge by requiring explicit motor actuation.
    '''
    # Set motor_auto flag to False for all joints.
    new_motor_auto = jnp.zeros_like(env_state.motor_auto, dtype=bool)
    return update_env_state(env_state, motor_auto=new_motor_auto)

@jax.jit
def mmp_reduce_densities(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Slightly reduce the densities of all polygons and circles.
    Reduces the physical inertia, making objects easier to move.
    '''
    # Reduce densities by multiplying with a factor less than 1.
    factor = 0.8
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities  = env_state.circle_densities  * factor
    return update_env_state(env_state, polygon_densities=new_polygon_densities,
                            circle_densities=new_circle_densities)

@jax.jit
def mmp_increase_densities(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Slightly increase the densities of all polygons and circles.
    Increases the physical inertia, making moving objects more challenging.
    '''
    # Increase densities by multiplying with a factor greater than 1.
    factor = 1.2
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities  = env_state.circle_densities  * factor
    return update_env_state(env_state, polygon_densities=new_polygon_densities,
                            circle_densities=new_circle_densities)

@jax.jit
def mmp_reduce_gravity(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Reduce the gravity value in the simulation.
    Eases the physical dynamics making it simpler to control objects.
    '''
    # Assume that env_state.gravity exists and is a jnp.ndarray.
    # Reduce gravity by 20%.
    factor = 0.8
    new_gravity = env_state.gravity * factor
    return update_env_state(env_state, gravity=new_gravity)

@jax.jit
def mmp_increase_gravity(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Increase the gravity value in the simulation.
    Intensifies the physical dynamics, increasing task difficulty.
    '''
    # Increase gravity by 20%.
    factor = 1.2
    new_gravity = env_state.gravity * factor
    return update_env_state(env_state, gravity=new_gravity)

@jax.jit
def mmp_randomize_thruster_bindings(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Randomize the thruster bindings.
    Disrupts the mapping between actions and thruster controls, increasing the control challenge.
    '''
    # Obtain a new permutation for thruster_bindings.
    # thruster_bindings is assumed to be a 1D jnp.ndarray of integers.
    new_bindings = jax.random.permutation(rng, env_state.thruster_bindings)
    return update_env_state(env_state, thruster_bindings=new_bindings)