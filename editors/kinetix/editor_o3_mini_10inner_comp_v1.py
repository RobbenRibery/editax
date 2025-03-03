# imports
import jax
import jax.numpy as jnp
import chex
from flax import struct

# Import the environment state definition.
from kinetix.environment.env_state import EnvState

## Utility function to update nested attributes.
def update_rigidbody_velocity(rigidbody, new_velocity, new_ang_velocity):
    return rigidbody.replace(
        velocity=new_velocity,
        angular_velocity=new_ang_velocity,
    )

# MMP 1: Reduce densities to ease dynamics.
@jax.jit
def mmp_reduce_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Slightly reduce the densities of all active polygons and circles.
    Lower density leads to lower inertia and softer collisions.
    '''
    new_polygon_densities = env_state.polygon_densities * 0.8
    new_circle_densities = env_state.circle_densities * 0.8
    return env_state.replace(
        polygon_densities=new_polygon_densities,
        circle_densities=new_circle_densities,
    )

# MMP 2: Increase densities to make the objects heavier.
@jax.jit
def mmp_increase_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Slightly increase the densities of all active polygons and circles.
    Higher density makes objects heavier and increases control difficulty.
    '''
    new_polygon_densities = env_state.polygon_densities * 1.2
    new_circle_densities = env_state.circle_densities * 1.2
    return env_state.replace(
        polygon_densities=new_polygon_densities,
        circle_densities=new_circle_densities,
    )

# MMP 3: Enable motor auto control.
@jax.jit
def mmp_enable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Set all motor_auto flags to True, enabling automatic motor behavior.
    This reduces the manual control difficulty.
    '''
    new_motor_auto = jnp.ones_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(
        motor_auto=new_motor_auto,
    )

# MMP 4: Disable motor auto control.
@jax.jit
def mmp_disable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Set all motor_auto flags to False, forcing manual motor control.
    This increases the challenge for the agent.
    '''
    new_motor_auto = jnp.zeros_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(
        motor_auto=new_motor_auto,
    )

# MMP 5: Zero all velocities for a static rest state.
@jax.jit
def mmp_zero_velocity(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Set linear and angular velocities for both polygons and circles to zero.
    This helps to mitigate dynamic uncertainty.
    '''
    # Assume env_state.polygon and env_state.circle are dataclasses with velocity and angular_velocity fields.
    new_polygon = env_state.polygon.replace(
        velocity=env_state.polygon.velocity * 0.0,
        angular_velocity=env_state.polygon.angular_velocity * 0.0,
    )
    new_circle = env_state.circle.replace(
        velocity=env_state.circle.velocity * 0.0,
        angular_velocity=env_state.circle.angular_velocity * 0.0,
    )
    return env_state.replace(
        polygon=new_polygon,
        circle=new_circle,
    )

# MMP 6: Add small random velocity perturbations.
@jax.jit
def mmp_random_velocity(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Inject small random perturbations to the linear and angular velocities of both polygons and circles.
    This increases the dynamic uncertainty of the environment.
    '''
    # Generate random noise for velocities.
    rng, subkey1, subkey2, subkey3, subkey4 = jax.random.split(rng, 5)
    noise_lin_poly = jax.random.uniform(subkey1, shape=env_state.polygon.velocity.shape, minval=-0.1, maxval=0.1)
    noise_ang_poly = jax.random.uniform(subkey2, shape=env_state.polygon.angular_velocity.shape, minval=-0.1, maxval=0.1)
    noise_lin_circle = jax.random.uniform(subkey3, shape=env_state.circle.velocity.shape, minval=-0.1, maxval=0.1)
    noise_ang_circle = jax.random.uniform(subkey4, shape=env_state.circle.angular_velocity.shape, minval=-0.1, maxval=0.1)
    
    new_polygon = env_state.polygon.replace(
        velocity=env_state.polygon.velocity + noise_lin_poly,
        angular_velocity=env_state.polygon.angular_velocity + noise_ang_poly,
    )
    new_circle = env_state.circle.replace(
        velocity=env_state.circle.velocity + noise_lin_circle,
        angular_velocity=env_state.circle.angular_velocity + noise_ang_circle,
    )
    return env_state.replace(
        polygon=new_polygon,
        circle=new_circle,
    )

# MMP 7: Simplify shape roles to an easy-to-interpret configuration.
@jax.jit
def mmp_simplify_roles(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Simplify the roles by assigning a fixed configuration:
      For polygons, set the first index as the goal (role 2) and the rest neutral (role 0).
      For circles, set the first index as the ball (role 1) and the rest neutral (role 0).
    This configuration makes reward association simpler.
    '''
    poly_size = env_state.polygon_shape_roles.shape[0]
    circ_size = env_state.circle_shape_roles.shape[0]
    new_polygon_roles = jnp.where(jnp.arange(poly_size) == 0, 2, 0)  # goal at index 0
    new_circle_roles = jnp.where(jnp.arange(circ_size) == 0, 1, 0)      # ball at index 0
    return env_state.replace(
        polygon_shape_roles=new_polygon_roles,
        circle_shape_roles=new_circle_roles,
    )

# MMP 8: Randomize shape roles to increase perceptual uncertainty.
@jax.jit
def mmp_complex_roles(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    '''
    Randomize the roles of all shapes:
      Assign random roles uniformly from {0, 1, 2, 3} for polygons and circles.
    This adds uncertainty to the reward structure and increases the difficulty.
    '''
    # Assume there are 4 roles, as given by EnvParams.num_shape_roles
    num_roles = 4
    rng, subkey1, subkey2 = jax.random.split(rng, 3)
    poly_shape = env_state.polygon_shape_roles.shape
    circ_shape = env_state.circle_shape_roles.shape
    new_polygon_roles = jax.random.randint(subkey1, shape=poly_shape, minval=0, maxval=num_roles)
    new_circle_roles = jax.random.randint(subkey2, shape=circ_shape, minval=0, maxval=num_roles)
    return env_state.replace(
        polygon_shape_roles=new_polygon_roles,
        circle_shape_roles=new_circle_roles,
    )