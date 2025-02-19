import json 
from editax.moed import EditorManager
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.environment.ued.ued_state import UEDParams

from dotenv import load_dotenv
load_dotenv()

from pprint import pprint

env_params = EnvParams()
static_env_params = StaticEnvParams()
ued_params = UEDParams()


with open("experiments/kinetix.json", "r") as f:
    config = json.load(f)

editor_manager = EditorManager(**config)


if __name__ == "__main__":

    pprint(config)
    print(editor_manager.input_string)
