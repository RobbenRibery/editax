import os 
os.environ["JAX_PLATFORMS"] = "cpu"
import bz2
import pickle
import sys
import time
from pathlib import Path

import argparse
from typing import Any, Tuple

import pygame

import chex 
import jax
import jax.numpy as jnp
import numpy as np
from craftax.world_gen.world_gen_configs import (
    SmoothGenConfig, 
    OVERWORLD_CONFIG,
    GNOMISH_MINES_CONFIG,
    TROLL_MINES_CONFIG,
    FIRE_LEVEL_CONFIG,
    ICE_LEVEL_CONFIG,
    BOSS_LEVEL_CONFIG,
    DUNGEON_CONFIG,
    SEWER_CONFIG,
    VAULTS_CONFIG
)
from craftax.constants import (
    OBS_DIM,
    BlockType,
    ItemType,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from craftax.craftax_symbolic_env import CraftaxSymbolicEnv as CraftaxEnv
from craftax.craftax_env import make_craftax_env_from_name
from craftax.renderer import render_craftax_pixels
from craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.world_gen.world_gen import (
    generate_world,
    get_new_empty_inventory,
)
from craftax.game_logic import calculate_light_level

STATIC_PARAMS = CraftaxEnv.default_static_params()
KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_DIAMOND_PICKAXE,
    pygame.K_5: Action.MAKE_WOOD_SWORD,
    pygame.K_6: Action.MAKE_STONE_SWORD,
    pygame.K_7: Action.MAKE_IRON_SWORD,
    pygame.K_8: Action.MAKE_DIAMOND_SWORD,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_e: Action.REST,
    pygame.K_COMMA: Action.ASCEND,
    pygame.K_PERIOD: Action.DESCEND,
    pygame.K_y: Action.MAKE_IRON_ARMOUR,
    pygame.K_u: Action.MAKE_DIAMOND_ARMOUR,
    pygame.K_i: Action.SHOOT_ARROW,
    pygame.K_o: Action.MAKE_ARROW,
    pygame.K_g: Action.CAST_FIREBALL,
    pygame.K_h: Action.CAST_ICEBALL,
    pygame.K_j: Action.PLACE_TORCH,
    pygame.K_z: Action.DRINK_POTION_RED,
    pygame.K_x: Action.DRINK_POTION_GREEN,
    pygame.K_c: Action.DRINK_POTION_BLUE,
    pygame.K_v: Action.DRINK_POTION_PINK,
    pygame.K_b: Action.DRINK_POTION_CYAN,
    pygame.K_n: Action.DRINK_POTION_YELLOW,
    pygame.K_m: Action.READ_BOOK,
    pygame.K_k: Action.ENCHANT_SWORD,
    pygame.K_l: Action.ENCHANT_ARMOUR,
    pygame.K_LEFTBRACKET: Action.MAKE_TORCH,
    pygame.K_RIGHTBRACKET: Action.LEVEL_UP_DEXTERITY,
    pygame.K_MINUS: Action.LEVEL_UP_STRENGTH,
    pygame.K_EQUALS: Action.LEVEL_UP_INTELLIGENCE,
    pygame.K_SEMICOLON: Action.ENCHANT_BOW,
}


def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        pickle.dump(data, f)


# Define a function to modify the environment for each curriculum stage
def modify_environment_for_stage(
    rng, env_state: EnvState, env_params: EnvParams, static_params: StaticEnvParams, stage: int
):
    # Reset the environment state for modifications
    map = env_state.map.at[:, :, :].set(BlockType.GRASS.value)
    item_map = env_state.item_map.at[:, :, :].set(ItemType.NONE.value)
    mob_map = env_state.mob_map.at[:, :, :].set(False)
    light_map = env_state.light_map.at[:, :, :].set(1.0)
    achievements = env_state.achievements.at[:].set(False)
    player_position = jnp.array(
        [static_params.map_size[0] // 2, static_params.map_size[1] // 2], dtype=jnp.int32
    )
    env_state = env_state.replace(
        map=map,
        item_map=item_map,
        mob_map=mob_map,
        light_map=light_map,
        achievements=achievements,
        player_position=player_position,
        player_level=0,
        inventory=get_new_empty_inventory(),
    )

    # Stage-specific modifications
    if stage == 1:
        # Stage 1: Basic Navigation and Movement
        # Place a goal block (e.g., WOOD) at a fixed location
        goal_position = player_position + jnp.array([5, 0])
        env_state = env_state.replace(
            map=env_state.map.at[0, goal_position[0], goal_position[1]].set(BlockType.WOOD.value)
        )
    elif stage == 2:
        # Stage 2: Introduction to Resource Collection - Wood
        # Scatter trees across the map
        rng, sub_rng = jax.random.split(rng)
        tree_positions = jax.random.randint(
            sub_rng, (10, 2), 0, static_params.map_size[0]
        )
        for pos in tree_positions:
            env_state = env_state.replace(
                map=env_state.map.at[0, pos[0], pos[1]].set(BlockType.TREE.value)
            )
    elif stage == 3:
        # Stage 3: Crafting Basic Tools - Wooden Pickaxe
        # Place a crafting table near the agent's starting position
        crafting_table_position = player_position + jnp.array([2, 0])
        env_state = env_state.replace(
            map=env_state.map.at[0, crafting_table_position[0], crafting_table_position[1]].set(
                BlockType.CRAFTING_TABLE.value
            ),
            inventory=env_state.inventory.replace(wood=5),
        )

    elif stage == 4:
        # Stage 4: Mining Stone and Crafting Stone Tools
        # Add stone formations on the map
        rng, sub_rng = jax.random.split(rng)
        stone_positions = jax.random.randint(
            sub_rng, (10, 2), 0, static_params.map_size[0]
        )
        for pos in stone_positions:
            env_state = env_state.replace(
                map=env_state.map.at[0, pos[0], pos[1]].set(BlockType.STONE.value)
            )
        # Ensure the agent has a wooden pickaxe
        env_state = env_state.replace(
            inventory=env_state.inventory.replace(pickaxe=1, wood=5)
        )
        # Place a crafting table
        crafting_table_position = player_position + jnp.array([2, 0])
        env_state = env_state.replace(
            map=env_state.map.at[0, crafting_table_position[0], crafting_table_position[1]].set(
                BlockType.CRAFTING_TABLE.value
            )
        )
    elif stage == 5:
        # Stage 5: Managing Survival Needs - Hunger and Thirst
        # Scatter edible plants and water sources
        rng, sub_rng = jax.random.split(rng)
        plant_positions = jax.random.randint(
            sub_rng, (5, 2), 0, static_params.map_size[0]
        )
        for pos in plant_positions:
            env_state = env_state.replace(
                map=env_state.map.at[0, pos[0], pos[1]].set(BlockType.RIPE_PLANT.value)
            )
        water_positions = jax.random.randint(
            sub_rng, (5, 2), 0, static_params.map_size[0]
        )
        for pos in water_positions:
            env_state = env_state.replace(
                map=env_state.map.at[0, pos[0], pos[1]].set(BlockType.WATER.value)
            )
        # Enable hunger and thirst with slow decay rates
        env_state = env_state.replace(
            player_food=10, player_drink=10, player_hunger=0.0, player_thirst=0.0
        )

    elif stage == 6:
        # Stage 6: Combat Introduction - Passive Mobs
        # Introduce passive mobs that wander randomly
        passive_mob_positions = jnp.array(
            [
                player_position + jnp.array([3, 0]),
                player_position + jnp.array([-3, 0]),
            ],
            dtype=jnp.int32,
        )
        for idx, pos in enumerate(passive_mob_positions):
            env_state.passive_mobs = env_state.passive_mobs.replace(
                position=env_state.passive_mobs.position.at[0, idx].set(pos),
                mask=env_state.passive_mobs.mask.at[0, idx].set(True),
                type_id=env_state.passive_mobs.type_id.at[0, idx].set(0),
            )
    elif stage == 7:
        # Stage 7: Basic Combat - Hostile Mobs
        # Introduce hostile melee mobs (e.g., zombies)
        env_state = env_state.replace(
            inventory=env_state.inventory.replace(sword=2)
        )
        hostile_mob_positions = jnp.array(
            [
                player_position + jnp.array([5, 5]),
                player_position + jnp.array([-5, -5]),
            ],
            dtype=jnp.int32,
        )
        for idx, pos in enumerate(hostile_mob_positions):
            env_state.melee_mobs = env_state.melee_mobs.replace(
                position=env_state.melee_mobs.position.at[0, idx].set(pos),
                mask=env_state.melee_mobs.mask.at[0, idx].set(True),
                type_id=env_state.melee_mobs.type_id.at[0, idx].set(0),
            )

    elif stage == 8:
        # Stage 8: Advanced Survival - Energy and Rest
        # Increase hunger and thirst decay rates slightly
        env_state = env_state.replace(
            player_energy=10, player_fatigue=0.0,
            player_food=10, player_drink=10
        )
    elif stage == 9:
        # Stage 9: Exploring Multiple Levels
        # Enable movement between levels by ensuring ladders are appropriately placed
        # Place ladders down and up on the map
        ladder_down_position = player_position + jnp.array([5, 5])
        ladder_up_position = player_position + jnp.array([-5, -5])
        env_state = env_state.replace(
            item_map=env_state.item_map.at[0, ladder_down_position[0], ladder_down_position[1]].set(
                ItemType.LADDER_DOWN.value
            ),
            down_ladders=env_state.down_ladders.at[0].set(ladder_down_position),
        )
        env_state = env_state.replace(
            item_map=env_state.item_map.at[1, ladder_up_position[0], ladder_up_position[1]].set(
                ItemType.LADDER_UP.value
            ),
            up_ladders=env_state.up_ladders.at[1].set(ladder_up_position),
        )
        # Initialize the next level similarly
        env_state = env_state.replace(
            map=env_state.map.at[1, :, :].set(BlockType.GRASS.value),
            item_map=env_state.item_map.at[1, :, :].set(ItemType.NONE.value),
            mob_map=env_state.mob_map.at[1, :, :].set(False),
        )
    elif stage == 10:
        # Stage 10: Advanced Resource Collection - Ores and Precious Materials
        # Place ores in deeper levels
        rng, sub_rng = jax.random.split(rng)
        iron_positions = jax.random.randint(
            sub_rng, (5, 2), 0, static_params.map_size[0]
        )
        for pos in iron_positions:
            env_state = env_state.replace(
                map=env_state.map.at[1, pos[0], pos[1]].set(BlockType.IRON.value)
            )
        coal_positions = jax.random.randint(
            sub_rng, (5, 2), 0, static_params.map_size[0]
        )
        for pos in coal_positions:
            env_state = env_state.replace(
                map=env_state.map.at[1, pos[0], pos[1]].set(BlockType.COAL.value)
            )
        # Ensure the agent can access the deeper level
        env_state = env_state.replace(
            inventory=env_state.inventory.replace(pickaxe=2),
            down_ladders=env_state.down_ladders.at[1].set(player_position + jnp.array([5, 5])),
        )
    elif stage == 11:
        # Stage 11: Combat with Ranged Mobs and Boss Mechanics
        # Introduce ranged mobs
        env_state = env_state.replace(
            inventory=env_state.inventory.replace(sword=3, bow=1, arrows=10)
        )
        ranged_mob_positions = jnp.array(
            [
                player_position + jnp.array([7, 7]),
                player_position + jnp.array([-7, -7]),
            ],
            dtype=jnp.int32,
        )
        for idx, pos in enumerate(ranged_mob_positions):
            env_state.ranged_mobs = env_state.ranged_mobs.replace(
                position=env_state.ranged_mobs.position.at[0, idx].set(pos),
                mask=env_state.ranged_mobs.mask.at[0, idx].set(True),
                type_id=env_state.ranged_mobs.type_id.at[0, idx].set(0),
            )
        # Place the boss in the final level
        boss_position = jnp.array([10, 10], dtype=jnp.int32)
        env_state = env_state.replace(
            map=env_state.map.at[2, boss_position[0], boss_position[1]].set(
                BlockType.NECROMANCER.value
            ),
            player_level=2,
        )

    elif stage == 12:
        # Stage 12: Full Environment Integration
        # Reset to the original environment settings
        rng, env_state = generate_world(rng, env_params, static_params)

    return rng, env_state




class CraftaxRenderer:
    def __init__(self, env: CraftaxEnv, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()

    def render(self, env_state):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state):
        if state.is_sleeping or state.is_resting:
            return Action.NOOP.value

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(
                f"{Achievement(i).name} ({new_achievements.sum()}/{len(Achievement)})"
            )

def layer_add_ladder(
    rng:chex.PRNGKey,
    layer_map:chex.Array, 
    item_map:chex.Array,
    can_update_ladder:bool,
    n_ladders_to_add:int,
    ladder_value:int,
    valid_value:int,
): 
    
    valid_ladder_down = layer_map.flatten() == valid_value
    rng, _rng = jax.random.split(rng)
    ladder_down_indxs = jax.random.choice(
        _rng,
        jnp.arange(STATIC_PARAMS.map_size[0] * STATIC_PARAMS.map_size[1]),
        p=valid_ladder_down,
        shape=(n_ladders_to_add, )
    )

    ladder_down_locs = jax.vmap(
        lambda idx: jnp.array(
            [
                idx // STATIC_PARAMS.map_size[0],
                idx % STATIC_PARAMS.map_size[0],
            ]
        )
    )(ladder_down_indxs)
        
    def step_fn(carry, loc): 

        item_map = carry 

        item_map = item_map.at[loc[0], loc[1]].set(
            ladder_value * can_update_ladder + \
            item_map[loc[0], loc[1]] * (1 - can_update_ladder)
        )
        return item_map, None 
    

    final_item_map, _  = jax.lax.scan(
        step_fn, 
        (item_map),
        ladder_down_locs,   
    )

    return final_item_map 

def main(args):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.params

    print("Controls")
    for k, v in KEY_MAPPING.items():
        print(f"{pygame.key.name(k)}: {v.name.lower()}")

    if args.god_mode:
        env_params = env_params.replace(god_mode=True)
        env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=True, **env_params)
        env_params = env.params

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng)
    stage = 1

    rng, env_state = modify_environment_for_stage(_rng, env_state, env_params, STATIC_PARAMS, stage)

    # n_ladders_down, n_ladders_up = 30, 10
    # def _ladder_down_step_fn(carry, xs): 
    #     rng = carry
    #     item_map, map, can_update_ladder, valid_value = xs

    #     rng, brng = jax.random.split(rng)

    #     updated_item_map = layer_add_ladder(
    #         brng,
    #         map,
    #         item_map,
    #         can_update_ladder,
    #         n_ladders_down,
    #         ItemType.LADDER_DOWN.value,
    #         valid_value,
    #     )
    #     return rng, updated_item_map
    
    # #print(env_state.map.shape)
    # rng, _rng = jax.random.split(rng)
    # _, updated_item_maps = jax.lax.scan(
    #     _ladder_down_step_fn, 
    #     (
    #         _rng
    #     ),
    #     (
    #         env_state.item_map,
    #         env_state.map,
    #         jax.tree_util.tree_map(
    #             lambda x1,x2,x3,x4,x5,x6,x7,x8,x9: jnp.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9), axis=0),
    #             OVERWORLD_CONFIG.ladder_down,
    #             True,
    #             GNOMISH_MINES_CONFIG.ladder_down,
    #             True,
    #             TROLL_MINES_CONFIG.ladder_down,
    #             True,
    #             FIRE_LEVEL_CONFIG.ladder_down,
    #             ICE_LEVEL_CONFIG.ladder_down,
    #             BOSS_LEVEL_CONFIG.ladder_down,
    #         ),
    #         jax.tree_util.tree_map(
    #             lambda x1,x2,x3,x4,x5,x6,x7,x8,x9: jnp.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9), axis=0),
    #             OVERWORLD_CONFIG.path_block,
    #             BlockType.PATH.value,
    #             GNOMISH_MINES_CONFIG.path_block,
    #             BlockType.PATH.value,
    #             TROLL_MINES_CONFIG.path_block,
    #             BlockType.PATH.value,
    #             FIRE_LEVEL_CONFIG.path_block,
    #             ICE_LEVEL_CONFIG.path_block,
    #             BOSS_LEVEL_CONFIG.path_block,
    #         )
    #     ),
    #     length=9,
    # )
    # env_state = env_state.replace(
    #     item_map = updated_item_maps
    # )

    # def _ladder_up_step_fn(carry, xs): 
    #     rng = carry
    #     item_map, map, can_update_ladder, valid_value = xs
    #     rng, brng = jax.random.split(rng)
    #     updated_item_map = layer_add_ladder(
    #         brng,
    #         map,
    #         item_map,
    #         can_update_ladder,
    #         n_ladders_up,
    #         ItemType.LADDER_UP.value, 
    #         valid_value,
    #     )
    #     return rng, updated_item_map
    
    # rng, _rng = jax.random.split(rng)
    # _, updated_item_maps = jax.lax.scan(
    #     _ladder_up_step_fn, 
    #     (_rng),
    #     (
    #         env_state.item_map,
    #         env_state.map,
    #         jax.tree_util.tree_map(
    #             lambda x1,x2,x3,x4,x5,x6,x7,x8,x9: jnp.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9), axis=0),
    #             OVERWORLD_CONFIG.ladder_up,
    #             True,
    #             GNOMISH_MINES_CONFIG.ladder_up,
    #             True,
    #             TROLL_MINES_CONFIG.ladder_up,
    #             True,
    #             FIRE_LEVEL_CONFIG.ladder_up,
    #             ICE_LEVEL_CONFIG.ladder_up,
    #             BOSS_LEVEL_CONFIG.ladder_up,
    #         ),
    #         jax.tree_util.tree_map(
    #             lambda x1,x2,x3,x4,x5,x6,x7,x8,x9: jnp.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9), axis=0),
    #             OVERWORLD_CONFIG.path_block,
    #             BlockType.PATH.value,
    #             GNOMISH_MINES_CONFIG.path_block,
    #             BlockType.PATH.value,
    #             TROLL_MINES_CONFIG.path_block,
    #             BlockType.PATH.value,
    #             FIRE_LEVEL_CONFIG.path_block,
    #             ICE_LEVEL_CONFIG.path_block,
    #             BOSS_LEVEL_CONFIG.path_block,
    #         )
    #     ),
    #     length=9,
    # )
    # env_state = env_state.replace(
    #     item_map = updated_item_maps,
    #     monsters_killed = jnp.ones(
    #         STATIC_PARAMS.num_levels, 
    #         dtype=jnp.int32
    #     )*10
    # )

    # def _layer_add_to_fountain_chest(
    #     carry:Tuple, 
    #     xs: Tuple,
    # ) -> Tuple[None, chex.Array]: 
                
    #     rng = carry
    #     rng, _rng = jax.random.split(rng)
    #     layer_map, add_items, valid_value = xs

    #     rng, _rng = jax.random.split(rng)
    #     valid_pos = layer_map.flatten() == valid_value  # valid indsx (1D) (48*48)
    #     valid_indxs = jax.random.choice( # sampled valid indxs (1D) (n_items_to_add,)
    #         _rng,
    #         jnp.arange(STATIC_PARAMS.map_size[0] * STATIC_PARAMS.map_size[1]),
    #         p=valid_pos,
    #         shape=(9, )
    #     )

    #     def _add_resource(map_:chex.Array, inputs_:Tuple[chex.Array, chex.Array]):
    #         """Add resource across different locations
    #         """
    #         idx, added_value = inputs_ 
    #         loc_ = jnp.array(
    #             [
    #                 idx // STATIC_PARAMS.map_size[0],
    #                 idx % STATIC_PARAMS.map_size[0],
    #             ]
    #         )
    #         map_ = map_.at[loc_[0], loc_[1]].set(added_value)
    #         return map_
            
    #     def _chain_add_resource(
    #         map__:chex.Array, 
    #         inputs__:Tuple[chex.Array, chex.Array]
    #         ) -> Tuple[chex.Array, chex.Array]:
            
    #         map__ =jax.lax.switch(
    #             0, 
    #             [_add_resource], 
    #             *(
    #                 map__,
    #                 inputs__,
    #             ),  
    #         ) 
    #         return map__, None 

    #     return rng, jax.lax.select(
    #         add_items,
    #         jax.lax.scan(
    #             _chain_add_resource,
    #             layer_map,
    #             (
    #                 valid_indxs, 
    #                 jnp.array(
    #                     [
    #                         BlockType.FOUNTAIN.value,
    #                         BlockType.CHEST.value,
    #                         BlockType.FOUNTAIN.value,
    #                         BlockType.CHEST.value,
    #                         BlockType.FOUNTAIN.value,
    #                         BlockType.CHEST.value,
    #                         BlockType.FOUNTAIN.value,
    #                         BlockType.CHEST.value,
    #                         BlockType.FOUNTAIN.value
    #                     ]
    #                 )    
    #             ),
    #         )[0],
    #         layer_map
    #     )
            
    # rng, _rng = jax.random.split(rng)
    # _, map_with_fountain_chest = jax.lax.scan(
    #     _layer_add_to_fountain_chest, 
    #     _rng,
    #     (
    #         env_state.map, # layer_map
    #         jnp.full(9, dtype=jnp.bool_, fill_value=True), # add_items
    #         jax.tree_util.tree_map( #valid values
    #             lambda x1,x2,x3,x4,x5,x6,x7,x8,x9: jnp.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9), axis=0),
    #             OVERWORLD_CONFIG.default_block,
    #             DUNGEON_CONFIG.special_block,
    #             GNOMISH_MINES_CONFIG.default_block,
    #             DUNGEON_CONFIG.special_block,
    #             TROLL_MINES_CONFIG.default_block,
    #             SEWER_CONFIG.special_block,
    #             FIRE_LEVEL_CONFIG.ladder_up,
    #             ICE_LEVEL_CONFIG.ladder_up,
    #             BOSS_LEVEL_CONFIG.ladder_up,
    #         ),
    #     ),
    # )
    # env_state = env_state.replace(map = map_with_fountain_chest)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN
    renderer = CraftaxRenderer(
        env, 
        env_params, 
        pixel_render_size=pixel_render_size
    )
    renderer.render(env_state)

    step_fn = env.step

    traj_history = {"state": [env_state], "action": [], "reward": [], "done": []}

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state)

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            _, env_state, reward, done, _ = step_fn(
                _rng, 
                env_state, 
                action,
                reset_on_done=True,
            )
            new_achievements = env_state.achievements
            print_new_achievements(old_achievements, new_achievements)

            if reward > 0.8:
                print(f"Reward: {reward}\n")

            traj_history["state"].append(env_state)
            traj_history["action"].append(action)
            traj_history["reward"].append(reward)
            traj_history["done"].append(done)

            renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)

    if args.save_trajectories:
        save_name = f"play_data/trajectories_{int(time.time())}"
        if args.god_mode:
            save_name += "_GM"
        save_name += ".pkl"
        Path("play_data").mkdir(parents=True, exist_ok=True)
        save_compressed_pickle(save_name, traj_history)


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--god_mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--fps", type=int, default=60)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)


if __name__ == "__main__":
    entry_point()