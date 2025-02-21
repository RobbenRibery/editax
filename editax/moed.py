from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import chex
from flax import struct

import os
from typing import List, Dict
import inspect 

from editax.template import EditorMaker, EditorCorrector
from editax.state import EnvState
from editax.utils import (
    LoggingHandler,
    EditorScriptParser, 
    code_utils_clear_cache,
    code_utils_split_code,
    code_utils_inject_corrections,
    code_utils_test_editors,
    prompt_utils_form_designs,
)

from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from functools import partial

import logging

from tqdm import tqdm 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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

        # edits config
        n_edits:int = 20,
        init_editors: bool = True,

        # logging
        verbose: bool = True,
    ):
        """
        Initialize the EditorManager.

        Args:
            env_name: The name of the environment.
            env_entry_point: The entry point of the environment.
            env_source_files: The source files of the environment.
            env_var: The environment variable to be considered.
            env_var_type: The type of the environment variable.
            out_dir: The output directory.
            out_filename: The output filename.
            llm_name: The name of the large language model.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature of the language model.
            max_correction_retry: The maximum number of corrections to retry.
            maker_template: The template for generating editors.
            corrector_template: The template for correcting editors.
            parser: The parser for extracting information from the source code.
            engine_statement: The statement regarding the engine configuration.
            editor_buffer_size: The size of the editor buffer.
            n_edits: The number of edits to generate.
            init_editors: Whether to initialize the editors.
            verbose: Whether to print verbose information.
        """
        # env info
        self.env_name = env_name
        self.env_entry_point = env_entry_point
        self.env_source_files = env_source_files
        self.env_var = env_var
        self.env_var_type = env_var_type
        self.engine_statement = engine_statement
        self.input_string = self.load_env_input_string()
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
        
        # parser
        self.parser = parser()

        # mutators info
        self.init_editors = init_editors
        self.out_dir = out_dir
        self.out_filename = out_filename
        self.out_path = os.path.join(self.out_dir, self.out_filename)
        self.verbose = verbose

    def load_env_input_string(self,) -> str: 
        """
        Returns a string containing all the source code of the environment source files,
        the entry point of the environment, the variable representing the environment,
        and the engine statement (if any).

        The string is formatted as follows:
        * The file name of each source file is surrounded by delimiters.
        * The content of each source file is included after its name.
        * The entry point of the environment is specified after all the source files.
        * The variable representing the environment is specified after the entry point.
        * The engine statement is specified after the variable representing the environment (if any).

        This string is used as the input to the large language model when generating editors.
        """
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

    def reset(self, dummy_env_state:EnvState, num_inner_loops:int = 8)-> Dict[str, Callable]:
        """
        Resets the editor manager, generating new editors and an editor buffer.

        Args:
            dummy_env_state: The dummy environment state used for testing the editors.
                Defaults to None.
            num_inner_loops: The number of design iterations to perform when generating
                editors. Defaults to 8.

        Returns:
            A dictionary mapping editor names to the corresponding editor functions
        """
        init_editor_map = self.generate_and_correct(
            corrective_func= self.llm_correct_editors,
            dummy_env_state= dummy_env_state,
            generative_args= (num_inner_loops, ),
            correction_only= False if self.init_editors else True,
        )
        self.editors_map = init_editor_map

        # register editors        
        self.editors:List[Callable] = [init_editor_map[k] for k in init_editor_map]
        self.n_eidtors = len(self.editors)

        return init_editor_map

    def llm_sample_editors_design(self,) -> str:
        """
        Sample a design for editor generation using the large language model.

        Returns:
            str: The generated design plan.
        """
        self.model.stop = ["[PLAN ENDS HERE]"]
        out = self.model.invoke(
            [
                (
                    "human",
                    self.maker_template.get_system_template() + "\n\n" +
                    self.maker_template.get_human_template().format(
                        **{
                            "input_string": self.input_string,
                            "engine_statement": self.engine_statement,
                            "env_var":self.env_var,
                            "env_var_type":self.env_var_type,
                        }
                    )
                ),
            ],
            config={"callbacks": [LoggingHandler()]}
        ).content
        self.model.stop = None
        return out
    
    def llm_sample_editors(self, num_inner_loops:int = 5,) -> Tuple[str, str]: 
        """
        Samples and generates editors using a self-consistency mechanism.

        This function generates multiple editor designs based on the input parameters
        and attempts to find the most consistent design through iterations. The selected
        design is then used to generate editors according to a plan. It uses a large 
        language model to assist in the design and generation of editors.

        Args:
            num_inner_loops (int, optional): The number of design iterations to perform. 
                Defaults to 5.

        Returns:
            Tuple[str, str]: A tuple containing the generated editors and the parsed 
            representation of the editors.
        """
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        # base messages 
        messages = [
            (
                "human",
                self.maker_template.get_human_template().format(
                    **{
                        "input_string": self.input_string,
                        "engine_statement": self.engine_statement,
                        "env_var":self.env_var,
                        "env_var_type":self.env_var_type,
                    }
                )
            ),
        ]
        deisgn_text_path = os.path.join(self.out_dir, "designs.txt")
        if not os.path.exists(deisgn_text_path):
            # designs
            repr_designs = []
            logger.info(f"Starting creating {num_inner_loops} designs...")
            for _ in tqdm(range(num_inner_loops)):
                editors_design = self.llm_sample_editors_design()
                repr_designs.append(editors_design)

            # select the most consistent design
            joined_designs = prompt_utils_form_designs(repr_designs)
            deisgn_text_path = os.path.join(self.out_dir, "designs.txt")
            with open(deisgn_text_path, "w") as f:
                f.write(joined_designs)
            logger.info(f"saved to {deisgn_text_path}")
            
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
        sc_plan = self.model.invoke(sc_messages, config={"callbacks": [LoggingHandler()]}).content
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
                        "input_string": self.input_string,
                        "engine_statement": self.engine_statement,
                        "env_var":self.env_var,
                        "env_var_type":self.env_var_type,
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
        model_out = self.model.invoke(code_gen_messages, config={"callbacks": [LoggingHandler()]}).content
        return model_out, self.parser.parse(model_out)
        
    def llm_correct_editors(
        self, 
        editors_script:str, 
        error_logs:str, 
        regen:bool=False,
        source_code_augmentation:bool=False
        ) -> Tuple[str, str]:
        """
        Correct/Regenerate an editor script based on test errors.

        This function takes in the generated editor script, the error logs, and a boolean flag regen to
        decide whether to generate a new editor from scratch or to correct the existing one. It also takes
        in a boolean flag source_code_augmentation to decide whether to use the source code augmentation
        or not.

        If regen is True, then the function generates a new editor from scratch. If source_code_augmentation
        is True, then the function uses the source code augmentation template. Otherwise, it uses the regular
        template. If regen is False, then the function takes the existing editor script and correct it based
        on the error logs. If source_code_augmentation is True, then the function uses the source code
        augmentation correction template. Otherwise, it uses the regular correction template.

        Args:
            editors_script (str): The generated editor script.
            error_logs (str): The error logs during the execution of the editor.
            regen (bool, optional): Whether to generate a new editor from scratch. Defaults to False.
            source_code_augmentation (bool, optional): Whether to use the source code augmentation. Defaults to False.

        Returns:
            Tuple[str, str]: The generated editor script and its parsed object.
        """
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
   
    def generate_and_correct(
        self, 
        corrective_func:Callable,
        dummy_env_state:EnvState,
        generation_func:Callable | None = None,
        generative_args:Tuple | None = None,
        correction_only:bool = False,
    ) -> Dict[str, Callable] | None:
        """
        Generate and correct editors using the given LLM.

        Args:
            corrective_func: A callable that takes in a string of code and a string of error message
                and returns the corrected code.
            dummy_env_state: An instance of EnvState. This is used to test the generated editors.
            generation_func: A callable that takes in the same arguments as `llm_sample_editors` and returns
                the generated editors.
            generative_args: A tuple of arguments to pass to `generation_func`.
            correction_only: If True, `generation_func` is not called and the method will only correct the
                existing editors at `self.out_path`.

        Returns:
            A dictionary mapping function names to their corresponding Callables if successful, None otherwise.
        """
        i = 0 
        file_name_without_ext = self.out_filename.rsplit('.py',1)[0]
        logger.info(f"File name: {file_name_without_ext}")
        caches:List[str] = []

        if not correction_only:
            logger.info(f"Generating Editors ...")
            if not generation_func:
                generation_func = self.llm_sample_editors
            llm_out, editors = generation_func(*generative_args)
            with open(os.path.join(self.out_dir, "moed_out.txt"), "w") as f:
                f.write(llm_out)
        else:
            editors:str = open(self.out_path, 'r').read()

        # save tmp editors 
        tmp_file_name = file_name_without_ext + f"_tmp_{i}.py"
        tmp_path = os.path.join(self.out_dir, tmp_file_name)
        with open(tmp_path, "w") as f:
            f.write(editors)
        f.close()
        caches.append(tmp_path)
        logger.info(f"Init editors -> {tmp_path}")

        # init test 
        errors_dict, _, func_map = code_utils_test_editors(dummy_env_state, tmp_path)
        updated_path = tmp_path

        if not errors_dict: 
            logger.info("Succeeded!")
            code_utils_clear_cache(caches)

            if os.path.isfile(self.out_path):
                return func_map

            # write to the destination 
            logger.info(f"Writing editors -> {self.out_path}")
            with open(self.out_path, "w") as f:
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
            code_prior, _ = code_utils_split_code(editors)
            editors_lit = code_prior + '\n' + inspect.getsource(func_map[func_name])
            corrections = corrective_func(editors_lit, error)
            
            # save the corrected script
            correction_file = file_name_without_ext + f"_tmp{i}_correction.py"
            correction_path = os.path.join(self.out_dir, correction_file)
            with open(correction_path, "w") as f:
                f.write(corrections)
            f.close()
            logger.info(f"Correction saved to {correction_path}")
            caches.append(correction_path)

            # load the corrected script as a module 
            updated_editors = code_utils_inject_corrections(updated_path, correction_path)

            updated_file = file_name_without_ext + f"_tmp{i}_correction_merged.py"
            updated_path = os.path.join(self.out_dir, updated_file)
            with open(updated_path, "w") as f:
                f.write(updated_editors)
            f.close()
            logger.info(f"Merged script saved to {updated_path}")
            caches.append(updated_path)

            # re-test
            errors_dict, _, func_map = code_utils_test_editors(dummy_env_state, updated_path)
            
            if not errors_dict: 
                editors = updated_editors
                logger.info("Succeeded!")
                
                # save successful
                with open(self.out_path, "w") as f:
                    f.write(editors)
                f.close()
                code_utils_clear_cache(caches)
                return func_map
                    
            # bump up the counter
            i += 1 

        code_utils_clear_cache(caches)
        logger.error(f"Generation Failed after {self.max_correction_retry} attempts")
        return None
    
    @partial(jax.jit, static_argnums=(0, ))
    def samle_random_edit_seqs(self, rng:chex.PRNGKey) -> chex.Array: 
        return jax.random.choice(
            rng,
            a = len(self.editors),
            shape=(self.n_edits,)
        )
    
    @partial(jax.jit, static_argnums=(0, ))
    def perform_edits(
        self, 
        rng:chex.PRNGKey, 
        env_state:EnvState,
        editors_indices:chex.Array,
    ) -> EnvState:
        
        # Step function for jax.lax.scan
        def _step_fn(carry, pair_):
            idx = pair_
            rng, current_env = carry
            
            # Split the RNG for the next step
            rng, arng, _ = jax.random.split(rng, 3)

            # Use jax.lax.switch to select the mutator 
            # function based on the index
            new_env = jax.lax.switch(
                idx, 
                self.editors,
                *(
                    arng, 
                    current_env, 
                ),
                )
            # Return the new carry (rng, new_env) and
            # the mutated environment for tracking
            return (rng, new_env,), new_env
        
        initial_carry = (rng, env_state)
        final_carry, _ = jax.lax.scan(
            _step_fn, 
            initial_carry, 
            (editors_indices),
        )
        _, new_env_state = final_carry
        return new_env_state