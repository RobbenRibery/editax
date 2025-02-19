from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import chex
from flax import struct

import os
import re
from typing import List, Dict
from textwrap import dedent
import ast 
import inspect 

from editax.template import EditorMaker, EditorCorrector
from editax.state import EnvState
from editax.utils import (
    LoggingHandler,
    EditorScriptParser, 
    code_utils_get_module_from_path,
    code_utils_clear_cache,
)

from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import logging

from tqdm import tqdm 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# def get_functions_from_file(file_path) -> List[Dict[str, str]]:
    
#     with open(file_path, 'r') as file:
#         tree = ast.parse(file.read())

#     def visit_function(node, nesting_level=0, parent=None):
#         # Get function source lines
#         start_line = node.lineno - 1
#         if node.decorator_list:
#             start_line = min(dec.lineno for dec in node.decorator_list) - 1
        
#         function_info = {
#             'text': '\n'.join(content.splitlines()[start_line:node.end_lineno]),
#             'nesting_level': nesting_level,
#             'parent': parent,
#             'lineno': node.lineno,
#             'args': [arg.arg for arg in node.args.args]
#         }
        
#         functions[node.name] = function_info
        
#         # Recursively visit nested functions
#         for child in ast.iter_child_nodes(node):
#             if isinstance(child, ast.FunctionDef):
#                 visit_function(child, nesting_level + 1, node.name)
    
#     functions = []
#     for node in ast.walk(tree):
#         if isinstance(node, ast.FunctionDef):
#             functions.append({
#                 'name': node.name,
#                 'lineno': node.lineno,
#                 'args': [arg.arg for arg in node.args.args],
#                 "code": astor.to_source(node),
#             })
#     return functions


class EditorBuffer(struct.PyTreeNode): 
    """Edditor Buffer's attributes

    Args:
        struct (_type_): _description_
    """
    successful_combos: chex.Array #(buffer_size, n_edits)
    unique_editor_indicies: chex.Array
    
    buffer_size: int = struct.field(pytree_node=False, default=400)
    n_edits: int = struct.field(pytree_node=False, default=20)
    default_num_mutation: int = struct.field(pytree_node=False, default=1)
    default_random_prob: float = struct.field(pytree_node=False, default=1.0)


class EditorManager:

    def __init__(
        self,
        # env info
        env_name: str,
        env_entry_point: str,
        env_source_files: List[str],
        env_var: str,
        env_var_type: str,
        out_dir: str,
        out_filename: str,

        # llm info
        llm_name: str,
        max_tokens: int,
        temperature: float,
        max_correction_retry:int,
        
        # editor templates
        maker_template = EditorMaker,
        corrector_template = EditorCorrector,
        parser = EditorScriptParser,
        engine_statement: str = "",
        editor_buffer_size:int = 320,
        n_edits:int = 20,
        init_editors: bool = True,
        verbose: bool = True,
    ):
        # env info
        self.env_name = env_name
        self.env_entry_point = env_entry_point
        self.env_source_files = env_source_files
        self.env_var = env_var
        self.env_var_type = env_var_type
        self.engine_statement = engine_statement
        self.input_string = self.load_env_input_string()

        # editor buffer 
        self.editor_buffer_size = editor_buffer_size
        self.n_edits = n_edits

        self.model_name = llm_name
        self.max_tokens = max_tokens
        self.max_correction_retry = max_correction_retry
        
        # prompt templates 
        self.maker_template = maker_template
        self.corrector_template = corrector_template

        # define llms 
        if "claude" in llm_name:
            assert "ANTHROPIC_API_KEY" in os.environ
            self.model = ChatAnthropic(
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,    
            )

        elif "gpt" in llm_name or 'o1' in llm_name or 'o3' in llm_name:
            assert "OPENAI_API_KEY" in os.environ
            if 'gpt' in self.model_name:
                self.model = ChatOpenAI(
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                self.model = ChatOpenAI(
                    model=self.model_name,
                    temperature=1,
                )
        elif "deepseek" in llm_name:
            assert "DEEPSEEK_API_KEY" in os.environ
            deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]
            self.model = BaseChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=deepseek_api_key,
                openai_api_base='https://api.deepseek.com',
            )
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")
        
        self.model.callbacks = [LoggingHandler()]
        
        # parser
        self.parser = parser

        # mutators info
        self.init_editors = init_editors
        self.out_dir = out_dir
        self.out_filename = out_filename
        self.out_path = os.path.join(self.out_dir, self.out_filename)
        self.verbose = verbose

    def load_env_input_string(self,) -> str: 
        delimitier = "***************"
        input_string = ""

        for source_file in self.env_source_files: 
            file_name = delimitier + f" {source_file} " + delimitier
            input_string += file_name + "\n" + open(source_file, 'r').read() + "\n"

        input_string  += f"\n *Entry point of the environment is {self.env_entry_point}"
        input_string += f"\n *Variable representing an environment is {self.env_var}:{self.env_var_type}"
        
        if self.engine_statement:
            input_string += f"\n *{self.engine_statement}"

        return input_string

    def reset(
        self, 
        rng:chex.PRNGKey,
        use_existing:bool = True, 
        dumm_env_state:EnvState | None = None,
    )-> Dict[str, Callable]:
        
        gen_func = self.llm_get_editors_sc
    
        init_editor_map = self.execute_and_correct(
            corrective_func= self.llm_fix_mutator,
            dummy_env_state= dumm_env_state,
            generation_func= gen_func,
            generative_args= (self.input_string, self.engine_statement, self.env_var, self.env_var_type),
            correction_only= use_existing,
        )
        
        self.org_editors_map = init_editor_map
        wrapped_editor_map = {
            k: wrapping_mutator(v, self.env_name) \
            for k,v in init_editor_map.items()
        }

        # init editors        
        self.editors:List[Callable] = [wrapped_editor_map[k] for k in init_editor_map]
        self.n_eidtors = len(self.editors)
        self.default_num_mutation = 1 

        # init the editors buffer 
        self.init_successful_combos = jax.random.choice(
            rng,
            self.n_eidtors,
            shape= (self.editor_buffer_size, self.n_edits),
        )
        return init_editor_map, EditorBuffer(
            successful_combos = self.init_successful_combos,
            unique_editor_indicies = jnp.arange(self.n_eidtors, dtype=jnp.int32),
            buffer_size = self.editor_buffer_size,
            n_edits = self.n_edits,
            default_num_mutation = self.default_num_mutation,
            default_random_prob= self.default_random_prob,
        )
    
    def llm_sample_editors_design(
        self, 
        input_string:str, 
        engine_statement:str, 
        env_var:str, 
        env_var_type:str
    ) -> str:
        self.model.stop = ["[PLAN ENDS HERE]"]
        out = self.model.invoke(
            [
                (
                    "human",
                    self.maker_template.get_system_template() + "\n\n" +
                    self.maker_template.get_human_template().format(
                        **{
                            "input_string": input_string,
                            "engine_statement": engine_statement,
                            "env_var":env_var,
                            "env_var_type":env_var_type,
                        }
                    )
                ),
            ]
        ).content
        self.model.stop = None
        return out
    
    def llm_sample_editors(
        self,
        input_string:str, 
        engine_statement:str, 
        env_var:str, 
        env_var_type:str,
        num_inner_loops:int = 8,
    ) -> Tuple[str, str]: 
        """
        Generates editors using a self-consistency approach based on input parameters.

        This function creates multiple editor designs based on the provided input parameters
        and selects the most consistent design through a self-consistency mechanism. The final
        design is then used to generate the editors according to the plan.

        Args:
            input_string (str): The input string to be formatted into the human template.
            engine_statement (str): The statement regarding the engine configuration.
            env_var (str): The environment variable to be considered.
            env_var_type (str): The type of the environment variable.
            num_inner_loops (int, optional): The number of design iterations to perform. Defaults to 5.

        Returns:
            str: The generated editors according to the most consistent design plan.
        """
        # base messages 
        messages = [
            (
                "human",
                self.maker_template.get_human_template().format(
                    **{
                        "input_string": input_string,
                        "engine_statement": engine_statement,
                        "env_var":env_var,
                        "env_var_type":env_var_type,
                    }
                )
            ),
        ]
        deisgn_text_path = os.path.join(self.mutators_dir, "designs.txt")
        if not os.path.exists(deisgn_text_path):
            # designs
            repr_designs = []
            logger.info(f"Starting creating {num_inner_loops} designs...")
            for _ in tqdm(range(num_inner_loops)):
                editors_design = self.llm_sample_editors_design(
                    input_string= input_string,
                    engine_statement= engine_statement,
                    env_var= env_var,
                    env_var_type= env_var_type,
                )
                repr_designs.append(editors_design)

            # select the most consistent design
            joined_designs = form_designs(repr_designs)
            deisgn_text_path = os.path.join(self.mutators_dir, "designs.txt")
            with open(deisgn_text_path, "w") as f:
                f.write(joined_designs)
                logger.info(f"saved to {deisgn_text_path}")
            f.close()
        else:
            logger.info(f"Loading existing designs from {deisgn_text_path}...")
            joined_designs = open(deisgn_text_path, 'r').read()
        
        logger.info("Selecting the most consistent editors design...")
        sc_messages = [
            messages[0],
            (
                "assistant",
                joined_designs
            ),
            (
                "human",
                self.maker_template.synthesize,
            )
        ]
        self.model.stop = ["[PLAN ENDS HERE]"]
        sc_plan = self.model.invoke(sc_messages).content
        self.model.stop = None
        
        # generate editors according to the plan
        logger.info("Starting generating editors according to the design...")
        logger.info(f"Self-consistency plan:\n{sc_plan}\n")
        self.model.stop = None
        code_gen_messages = [
            (
                "human", 
                self.maker_template.get_full().format(
                    **{
                        "input_string": input_string,
                        "engine_statement": engine_statement,
                        "env_var":env_var,
                        "env_var_type":env_var_type,
                    }   
                ),
            ),
            (
                "assistant",
                sc_plan,
            ),
            (
                "human",
                "Complete the function generation according to the plan that you've produced:"
            ),
        ]
        model_out = self.model.invoke(code_gen_messages).content
        return model_out, self.parser.parse(model_out)
        
    def llm_correct_editors(
        self, 
        editors_script:str, 
        error_logs:str, 
        regen:bool=False,
        source_code_augmentation:bool=False
        ) -> Tuple[str, str]:

        if regen: 
            if source_code_augmentation:
                out = self.model.invoke(
                    self.corrector_template.get_fix_full(
                        source_augmented=True
                    ).format(
                        **{
                            "editors_script": editors_script,
                            "error_logs": error_logs,
                            "input_string": self.input_string,
                            "engine_statement": self.engine_statement,
                        }
                    )
                )
            else:
                out = self.model.invoke(
                    self.corrector_template.get_regen_full().format(
                        **{
                            "editors_script": editors_script,
                            "error_logs": error_logs,
                        }
                    )
                )
        else:
            if source_code_augmentation:
                out = self.model.invoke(
                    self.corrector_template.get_fix_full(
                        source_augmented=True
                    ).format(
                        **{
                            "editors_script": editors_script,
                            "error_logs": error_logs,
                            "input_string": self.input_string,
                            "engine_statement": self.engine_statement,
                        }
                    )
                )
            else:
                out = self.model.invoke(
                    self.corrector_template.get_fix_full().format(
                        **{
                            "editors_script": editors_script,
                            "error_logs": error_logs,
                        }
                    )   
                )

        return out, self.parser.parse(out.content)
   
    def execute_and_correct(
        self, 
        corrective_func:Callable,
        dummy_env_state:EnvState,
        generation_func:Callable | None = None,
        generative_args:Tuple | None = None,
        correction_only:bool = False,
    ) -> Dict[str, Callable] | None:
        
        i = 0 
        file_name_without_ext = self.mutators_filename.rsplit('.py',1)[0]
        logger.info(f"File name: {file_name_without_ext}")
        caches:List[str] = []

        if not correction_only:
            logger.info(f"Generating Editors ...")
            editors:str = generation_func(*generative_args)
        else:
            editors:str = open(self.mutators_path, 'r').read()

        # save tmp editors 
        tmp_file_name = file_name_without_ext + f"_tmp_{i}.py"
        tmp_path = os.path.join(self.mutators_dir, tmp_file_name)
        with open(tmp_path, "w") as f:
            f.write(editors)
        f.close()
        caches.append(tmp_path)
        logger.info(f"Init editors -> {tmp_path}")

        # init test 
        errors_dict, _, func_map = get_testing_func(self.env_name)(dummy_env_state, tmp_path)
        updated_path = tmp_path

        if not errors_dict: 
            logger.info("Succeeded!")
            clear_cache(caches)

            if os.path.isfile(self.mutators_path):
                return func_map

            # write to the destination 
            logger.info(f"Writing editors -> {self.mutators_path}")
            with open(self.mutators_path, "w") as f:
                f.write(editors)
            f.close()
            return func_map

        logger.info("Starting correction...")
        while i <= self.max_correction_retry:
            logger.info(f"Iteration {i}/{self.max_correction_retry}")
            
            # fix the first function
            func_name = list(errors_dict.keys())[0]
            error = errors_dict[func_name]
                    
            # perform correction
            logger.info(f"LLM Attempting to fix: {func_name} ...")
            code_prior, _ = split_code(editors)
            editors_lit = code_prior + '\n' + inspect.getsource(func_map[func_name])
            corrections = corrective_func(editors_lit, error)
            
            # save the corrected script
            correction_file = file_name_without_ext + f"_tmp{i}_correction.py"
            correction_path = os.path.join(self.mutators_dir, correction_file)
            with open(correction_path, "w") as f:
                f.write(corrections)
            f.close()
            logger.info(f"Correction saved to {correction_path}")
            caches.append(correction_path)

            # load the corrected script as a module 
            updated_editors = inject_corrections(updated_path, correction_path)

            updated_file = file_name_without_ext + f"_tmp{i}_correction_merged.py"
            updated_path = os.path.join(self.mutators_dir, updated_file)
            with open(updated_path, "w") as f:
                f.write(updated_editors)
            f.close()
            logger.info(f"Merged script saved to {updated_path}")
            caches.append(updated_path)

            # re-test
            errors_dict, _, func_map = get_testing_func(self.env_name)(dummy_env_state, updated_path)
            
            if not errors_dict: 
                editors = updated_editors
                logger.info("Succeeded!")
                
                # save successful
                with open(self.mutators_path, "w") as f:
                    f.write(editors)
                f.close()
                clear_cache(caches)
                return func_map
                    
            # bump up the counter
            i += 1 

        clear_cache(caches)
        raise ValueError(f"Generation Failed after {self.max_correction_retry} attempts")


# class PoPEditorManager(EditorManager):
#     """Vectorised the Editor Manager 
#     across multiple students 
#     Args:
#         EditorManager (_type_): _description_
#     """
#     def __init__(self, *, n_agents, **kwargs):
#         super().__init__(**kwargs)
#         self.n_agents = n_agents 

#     @partial(jax.jit, static_argnums=(0, 1))
#     def reset(self, rng: chex.PRNGKey, n:int, *args):
#         sup = super()
#         return jax.vmap(
#             lambda *_: sup.reset()
#         )(np.arange(n))