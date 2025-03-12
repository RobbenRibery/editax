from typing import List
# Imports
import jax
import jax.numpy as jnp
import chex
from kinetix.environment.env_state import EnvState

# Utility: a helper function for immutable updates in arrays.
def update_array(arr: jnp.ndarray, idx: int, new_val):
    return arr.at[idx].set(new_val)

# MMP 1: Reduce densities of all polygons and circles (easier dynamics)
@jax.jit
def mmp_reduce_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Reduce densities by a small factor (0.9x) for all polygons and circles.
    Lower densities make objects lighter, easing the control challenge.
    """
    factor = 0.9
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities = env_state.circle_densities * factor
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)

# MMP 2: Increase densities of all polygons and circles (harder dynamics)
@jax.jit
def mmp_increase_density(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Increase densities by a small factor (1.1x) for all polygons and circles.
    Higher densities make objects heavier, increasing the control challenge.
    """
    factor = 1.1
    new_polygon_densities = env_state.polygon_densities * factor
    new_circle_densities = env_state.circle_densities * factor
    return env_state.replace(polygon_densities=new_polygon_densities,
                             circle_densities=new_circle_densities)

# MMP 3: Enable automatic motor control for all joints (reducing control difficulty)
@jax.jit
def mmp_enable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Set all motor_auto flags to True.
    This allows the environment to override manual motor commands.
    """
    new_motor_auto = jnp.ones_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)

# MMP 4: Disable automatic motor control for all joints (increasing control difficulty)
@jax.jit
def mmp_disable_motor_auto(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Set all motor_auto flags to False.
    This forces the agent to control every joint manually.
    """
    new_motor_auto = jnp.zeros_like(env_state.motor_auto, dtype=bool)
    return env_state.replace(motor_auto=new_motor_auto)

# MMP 5: Remove an obstacle by deactivating a non-floor polygon (reducing challenge)
@jax.jit
def mmp_remove_obstacle(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Randomly deactivates one polygon (excluding the floor at index 0) that is currently active.
    Removing obstacles can simplify navigation.
    """
    # Retrieve the 'active' boolean array for polygons.
    active = env_state.polygon.active
    indices = jnp.arange(active.shape[0])
    
    # Exclude the floor (index 0) and select only active polygons.
    valid_mask = (indices > 0) & active
    
    # Set a static maximum size equal to the total number of polygons.
    max_candidates = active.shape[0]
    # Compute candidates using a static-size nonzero; invalid positions are filled with -1.
    candidates = jnp.nonzero(valid_mask, size=max_candidates, fill_value=-1)[0]
    num_candidates = jnp.sum(valid_mask)

    def no_candidate():
        return env_state

    def deactivate_candidate():
        key, subkey = jax.random.split(rng)
        # Randomly select an index among the candidates.
        choice = jax.random.randint(subkey, shape=(), minval=0, maxval=num_candidates)
        chosen_idx = candidates[choice]
        new_active = active.at[chosen_idx].set(False)
        return env_state.replace(polygon=env_state.polygon.replace(active=new_active))
    
    return jax.lax.cond(num_candidates > 0, deactivate_candidate, no_candidate)

# MMP 6: Add an obstacle by reactivating a non-floor polygon (increasing challenge)
@jax.jit
def mmp_add_obstacle(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Randomly reactivates one polygon (excluding the floor at index 0) that is currently inactive.
    Adding extra obstacles increases the difficulty.
    """
    active = env_state.polygon.active
    indices = jnp.arange(active.shape[0])
    valid_mask = (indices > 0) & (~active)
    count = jnp.sum(valid_mask)
    
    def no_candidate():
        return env_state

    def activate_candidate():
        key, subkey = jax.random.split(rng)
        rand_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=count)
        # Get candidate indices as an array from the first element of the tuple output.
        candidate_indices = jnp.nonzero(valid_mask, size=active.shape[0], fill_value=-1)[0]
        # Use jnp.take to perform safe integer indexing with a traced integer.
        chosen_idx = jnp.take(candidate_indices, rand_idx)
        new_active = active.at[chosen_idx].set(True)
        return env_state.replace(polygon=env_state.polygon.replace(active=new_active))

    return jax.lax.cond(count > 0, activate_candidate, no_candidate)

# MMP 7: Swap shape roles among polygons and circles (introducing perceptual ambiguity)
@jax.jit
def mmp_swap_shape_roles(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Randomly permutes polygon_shape_roles and circle_shape_roles.
    This introduces ambiguity in identifying goals versus obstacles.
    """
    key1, key2 = jax.random.split(rng)
    poly_roles = env_state.polygon_shape_roles
    perm_poly = jax.random.permutation(key1, poly_roles)
    circle_roles = env_state.circle_shape_roles
    perm_circle = jax.random.permutation(key2, circle_roles)
    return env_state.replace(polygon_shape_roles=perm_poly, circle_shape_roles=perm_circle)

# MMP 8: Perturb positions of active polygons and circles (increasing unpredictability)
@jax.jit
def mmp_perturb_positions(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Adds small random displacements to the positions of active polygons and circles.
    This subtle perturbation can affect collision timing and control.
    """
    # Assume env_state.polygon.position and env_state.circle.position exist.
    eps = 0.05  # Maximum displacement amplitude
    key1, key2 = jax.random.split(rng)
    
    poly_pos = env_state.polygon.position
    poly_active = env_state.polygon.active.reshape(-1, 1)
    perturb_poly = jax.random.uniform(key1, shape=poly_pos.shape, minval=-eps, maxval=eps) * poly_active
    new_poly_pos = poly_pos + perturb_poly
    
    circle_pos = env_state.circle.position
    circle_active = env_state.circle.active.reshape(-1, 1)
    perturb_circle = jax.random.uniform(key2, shape=circle_pos.shape, minval=-eps, maxval=eps) * circle_active
    new_circle_pos = circle_pos + perturb_circle
    
    return env_state.replace(polygon=env_state.polygon.replace(position=new_poly_pos),
                             circle=env_state.circle.replace(position=new_circle_pos))

# MMP 9: Facilitate goal attainment by forcing one polygon to be a goal
@jax.jit
def mmp_facilitate_goal(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Ensures that at least one active polygon (excluding the floor at index 0) is assigned the goal role (role=2).
    This modification may help the policy in obtaining reward by highlighting a clear target.
    """
    poly_roles = env_state.polygon_shape_roles
    active = env_state.polygon.active
    N = poly_roles.shape[0]
    indices = jnp.arange(N)
    # Exclude the floor (index 0) and ensure polygon is active:
    valid_mask = (indices > 0) & active

    # Pre-allocate an array for valid indices with a known static size.
    init_array = -jnp.ones((N,), dtype=jnp.int32)
    init_count = 0

    # Loop over indices and accumulate valid ones.
    def body_fun(i, carry):
        candidate_array, count = carry
        candidate_array, count = jax.lax.cond(
            valid_mask[i],
            lambda _: (candidate_array.at[count].set(i), count + 1),
            lambda _: (candidate_array, count),
            operand=None
        )
        return (candidate_array, count)

    candidate_array, valid_count = jax.lax.fori_loop(0, N, body_fun, (init_array, init_count))

    # Define branch to keep the env_state unchanged.
    def no_candidate():
        return env_state

    # Define branch to assign the role with goal (role=2) randomly selected from valid candidates.
    def assign_goal():
        key, subkey = jax.random.split(rng)
        random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=valid_count)
        chosen_idx = candidate_array[random_index]
        new_roles = poly_roles.at[chosen_idx].set(2)
        return env_state.replace(polygon_shape_roles=new_roles)

    return jax.lax.cond(valid_count > 0, assign_goal, no_candidate)

# MMP 10: Complexify shape roles by turning a friendly shape into a hazard (role 3)
@jax.jit
def mmp_complexify_shape_roles(rng: chex.PRNGKey, env_state: EnvState) -> EnvState:
    """
    Randomly selects an active polygon (excluding the floor at index 0) that currently has a friendly role (1 or 2)
    and sets its role to 3 (hazard). This increases the challenge by confusing the perceptual cues.
    """
    poly_roles = env_state.polygon_shape_roles
    active = env_state.polygon.active
    indices = jnp.arange(poly_roles.shape[0])
    # Filter for active polygons (excluding floor) with friendly roles (assume roles 1 and 2 are friendly)
    valid_mask = (indices > 0) & active & ((poly_roles == 1) | (poly_roles == 2))
    valid_count = jnp.sum(valid_mask)

    def no_candidate():
        return env_state

    def assign_hazard():
        key, subkey = jax.random.split(rng)
        # Randomly choose one of the valid candidate indices based on the count
        choice = jax.random.randint(subkey, shape=(), minval=0, maxval=valid_count)
        # Compute the cumulative sum over the boolean mask to map the choice to an index
        cumsum = jnp.cumsum(valid_mask, dtype=jnp.int32)
        # 'choice+1' will match exactly once when a valid element is encountered
        chosen_idx = jnp.argmax(cumsum == (choice + 1))
        new_roles = poly_roles.at[chosen_idx].set(3)
        return env_state.replace(polygon_shape_roles=new_roles)

    return jax.lax.cond(valid_count > 0, assign_hazard, no_candidate)