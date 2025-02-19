from craftax.craftax_state import EnvState
from craftax.constants import *

from typing import Dict 
import chex 

def log_achievements_to_info(state: EnvState, done: bool) -> Dict[str, chex.Array]:
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        info[name] = achievements[achievement.value]
    return info