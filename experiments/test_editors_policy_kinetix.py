import os 
import json 
os.environ['JAX_PLATFORMS'] = 'cpu' 

import jax 
import jax.numpy as jnp 
import chex 

from editax.moed import EditorManager, EditorPolicyTrainState

import optax  

from kinetix.render.renderer_symbolic_entity import make_render_entities
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.environment.env import make_kinetix_env_from_args
from kinetix.environment.ued.ued import UEDParams
from kinetix.environment.ued.ued import (
    make_reset_train_function_with_mutations,
    make_vmapped_filtered_level_sampler,
)
from kinetix.environment.ued.distributions import sample_kinetix_level

from kinetix.models.transformer_model import ActorCriticTransformer
from kinetix.models.actor_critic import ScannedRNN 

## configs. 
config = {
    "num_minibatches": 8,
    "update_epochs": 4,
    "num_updates": 1,
    "outer_rollout_steps": 4,
    "num_steps": 256,
    "num_train_envs": 32,
    "anneal_lr": True,
    "lr": 1e-4,
    "max_grad_norm": 1.0,
    "transformer_depth": 2,
    "transformer_size": 16,
    "transformer_encoder_size": 128,
    "num_heads": 8,
    "full_attention_mask": False,
    "aggregate_mode": "dummy_and_mean",
    "fc_layer_depth": 5,
    "fc_layer_width": 128,
    "activation": "tanh",
    "recurrent_model":True,
    "env_size_name": "m",
    "filter_levels": False,
    "level_filter_sample_ratio":0,
    "level_filter_n_steps":0,
    "env_name": "Kinetix-Entity-MultiDiscrete-v1",
}

# load ued env params 
env_params = EnvParams()
static_env_params = StaticEnvParams()
ued_params = UEDParams()
renderer = make_render_entities(params=env_params, static_params=static_env_params)

# Create the environment
print("Creating Kinetix environment...")
env = make_kinetix_env_from_args(
    obs_type="entity",
    action_type="multidiscrete",
    reset_type="replay",
    static_env_params=static_env_params,
)


sample_random_level = make_reset_train_function_with_mutations(
    env.physics_engine, env_params, static_env_params, config, make_pcg_state=False
)
sampler = make_vmapped_filtered_level_sampler(
    sample_random_level, env_params, static_env_params, config, make_pcg_state=False, env=env
)

# Sample a random level
print("Sampling random level...")
rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
level = sample_kinetix_level(
    _rng, 
    env.physics_engine, 
    env_params, 
    static_env_params, 
    ued_params, 
    env_size_name="m"
)
print(f"Level shape: {jax.tree_util.tree_map(lambda x: x.shape, level)}")

# Reset the environment state to this level
print("Resetting environment to sampled level...")
rng, _rng = jax.random.split(rng)
obs, env_state = env.reset_to_level(_rng, level, env_params)

print(f"env type: {type(env)}")
print(f"Level type: {type(level)}")
print(f"Env state type: {type(env_state)}")
print(f"Obs type: {type(obs)}")

print(f"Initial observation shape: {jax.tree_util.tree_map(lambda x: x.shape, obs)}")
#print(f"Environment state shape: {jax.tree_util.tree_map(lambda x: x.shape, env_state)}")

# load editor config
rng, rng_editor = jax.random.split(rng)
print("Loading editor config") 
with open("experiments/kinetix.json", "r") as f:
    editor_config = json.load(f)
editor_config["init_editors"] = False
num_inner_loops = editor_config.pop("num_inner_loops", 10)

editor_manager = EditorManager(**editor_config)
_ = editor_manager.reset(level, num_inner_loops)
print(f"Initliazed editor manager containing {len(editor_manager.editors)} editors")

batch_size = 32
init_seq_len = 1

print(f"\nInitializing batch with:")
print(f"Batch size: {batch_size}")
print(f"Initial sequence length: {init_seq_len}")

obs = jax.tree_util.tree_map(
    lambda x: jnp.repeat(
        jnp.repeat(
            x[jnp.newaxis, ...], 
            batch_size, 
            axis=0
        )[jnp.newaxis, ...], 
        init_seq_len, 
        axis=0
    ),
    obs
)

print(f"After batching:")
print(f"Batched observation shape: {jax.tree_util.tree_map(lambda x: x.shape, obs)}")

# load policy 
print("\nInitializing policy...")

n_editor = len(editor_manager.editors)
policy = ActorCriticTransformer(
    action_dim=(n_editor),
    fc_layer_width=config["fc_layer_width"],
    fc_layer_depth=config["fc_layer_depth"],
    action_mode="discrete",
    num_heads=config["num_heads"],
    transformer_depth=config["transformer_depth"],
    transformer_size=config["transformer_size"],
    transformer_encoder_size=config["transformer_encoder_size"],
    aggregate_mode=config["aggregate_mode"],
    full_attention_mask=config["full_attention_mask"],
    activation=config["activation"],
    **{
        "hybrid_action_continuous_dim": (n_editor),
        "multi_discrete_number_of_dims_per_distribution": [n_editor],
        "recurrent": True,
    }
)
print("Policy initialized successfully")

@jax.jit
def create_editor_policy_train_state(rng:chex.PRNGKey) -> EditorPolicyTrainState:
    print("\nCreating editor policy train state...")
    rng, _rng = jax.random.split(rng)
    init_x = (
        obs, 
        jnp.zeros(
            (init_seq_len, config["num_train_envs"]), dtype=jnp.bool_)
    )
    
    print("Initializing network parameters...")
    policy_init_carry = ScannedRNN.initialize_carry(config["num_train_envs"])
    network_params = policy.init(
        _rng, 
        policy_init_carry,
        init_x
    )
    print("Network parameters initialized")
    
    print("Creating optimizer...")
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["lr"], eps=1e-5),
    )

    print("Creating final train state...")
    editor_policy_train_state = EditorPolicyTrainState.create(
        apply_fn=policy.apply,
        params=network_params,
        tx=tx,
        num_updates=0,
    )
    print("Train state created successfully")
    return editor_policy_train_state


# create train state 
rng, _rng = jax.random.split(rng)
editor_train_state = create_editor_policy_train_state(_rng)

print("Sampling levels for editor policy rollouts...")
rng, _rng = jax.random.split(rng)
sampled_levels = sampler(_rng, config["num_train_envs"])

rng, _rng = jax.random.split(rng)
init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
    jax.random.split(_rng, config["num_train_envs"]), 
    sampled_levels, 
    env_params
)
print(f"Input env state type is {type(init_env_state)}")
# sampled editing trajectories 
(
    rng, 
    train_state, 
    hstate, 
    obs, 
    last_env_state, 
    last_value
), traj = editor_manager.sample_edit_trajectories(
        env = env,
        rng = _rng,
        train_state= editor_train_state,
        init_hstate= ScannedRNN.initialize_carry(config["num_train_envs"]),
        init_obs = init_obs,
        init_env_state = init_env_state.env_state.env_state.env_state,
        env_params = env_params,
        num_envs = config["num_train_envs"],
)
print(f"Rollout completed")
print(f"Last env state type is {type(last_env_state)} of shape {last_env_state.thruster_bindings.shape}")
# measure the reward as the lift in leanrability score  





# update the eidtors's policy 
