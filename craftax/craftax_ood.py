from craftax.craftax_symbolic_env import CraftaxSymbolicEnv

class Craftax101(CraftaxSymbolicEnv):

    def __init__(self, **env_kwargs):
        env_kwargs["singleton_seed"] = 101
        super().__init__(**env_kwargs)
