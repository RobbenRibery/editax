from craftax.craftax_pixels_env import (
    CraftaxPixelsEnv,
)
from craftax.craftax_symbolic_env import (
    CraftaxSymbolicEnv,
)

def make_craftax_env_from_name(name: str, auto_reset: bool=True, **env_kwargs):
    if auto_reset:
        if name == "Craftax-Symbolic-v1" or name == "Craftax-Symbolic-AutoReset-v1":
            return CraftaxSymbolicEnv(**env_kwargs)
        elif name == "Craftax-Pixels-v1" or name == "Craftax-Pixels-AutoReset-v1":
            return CraftaxPixelsEnv(**env_kwargs)

    raise ValueError(f"Unknown craftax environment: {name}")


def make_craftax_env_from_params(classic: bool, symbolic: bool, auto_reset: bool):
    if symbolic:
        if auto_reset:
            return CraftaxSymbolicEnv()
        else:
            raise ValueError
    else:
        if auto_reset:
            return CraftaxPixelsEnv()
        else:
            raise ValueError