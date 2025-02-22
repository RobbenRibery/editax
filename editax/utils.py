from typing import Any, Dict, List, Tuple, Callable
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.output_parsers import BaseOutputParser

from editax.upomdp import EnvState

import jax
import chex

import traceback
import importlib
import inspect
import logging
import ast 
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def code_utils_get_module_from_path(path:str) -> str: 
    path_without_suffix = path.strip(".py") 
    return  path_without_suffix.replace("/",".")

def code_utils_split_code(code:str) -> str: 
    """
    Extracts the imports, utility functions, and constants from the given code
    string up to the first @jax.jit or @jit decorator. Returns the extracted code
    as a string.

    Args:
        code (str): Source code string containing the imports and utility functions.

    Returns:
        str: Extracted import and utility code as a string.
    """
    code_prior_pattern = r"(.*?)\n(?:@jax\.jit|@jit)\s*\ndef\s+(?:mmp|mp)"
    code_prior_section = re.findall(code_prior_pattern, code, re.DOTALL)[0]
    code_main_body = code.replace(code_prior_section, "")
    code_prior_section_lines = code_prior_section.split('\n')

    code_prior = ""
    for line in code_prior_section_lines:
        if line:
            if line[0] != "#":
                code_prior += line + '\n'
    
    return code_prior, code_main_body

def code_utils_clear_cache(cache_files:List[str]) -> None:
    [os.remove(cache_file) for cache_file in cache_files]

def code_utils_get_truncated_error(log:str, keyword:str) -> str: 
    """
    Given a log and a keyword, return the portion of the log
    that starts at the line containing the keyword. 

    Args:
        log (str): the entire log
        keyword (str): the keyword to search for

    Returns:
        str: the portion of the log after the line containing the keyword
    """
    logs = log.split('\n')
    for i, line in enumerate(logs):
        if keyword in line:
            break 
    return '\n'.join(logs[i:])

def code_utils_inject_corrections(
    curr_file:str, 
    corr_file:str, 
    key_variable:str = "updates"
    ) -> str:
    """
    Inject corrections from a correction file to a current file.

    Parameters
    ----------
    curr_file : str
        The path to the current file.
    corr_file : str
        The path to the correction file.

    Returns
    -------
    str
        The modified script as a string.
    """
    curr_script = open(curr_file, "r").read()
    corr_script = open(corr_file, "r").read()

    curr_imports = parse_imports(curr_script)
    corr_imports = parse_imports(corr_script)

    # added the additional imports 
    for corr_import in corr_imports:
        if corr_import not in curr_imports:
            curr_script = corr_import + "\n" + curr_script

    curr_module = importlib.import_module(code_utils_get_module_from_path(curr_file)) 
    corr_module = importlib.import_module(code_utils_get_module_from_path(corr_file))

    for func_name in getattr(corr_module, key_variable): 
        print(f"Proposed correction for {func_name}")
        new_func = getattr(corr_module, func_name)

        # if the curr script has such function 
        if hasattr(curr_module, func_name):
            old_func = getattr(curr_module, func_name)

            new_func_str = inspect.getsource(new_func).strip()
            old_func_str = inspect.getsource(old_func).strip()
            print(new_func_str)
            print(old_func_str)
            assert old_func_str in curr_script

            if new_func_str != old_func_str:
        
                print(f"Updating {func_name}")
                print(len(curr_script))

                start_old_func = curr_script.index(old_func_str)
                end_old_func = start_old_func + len(old_func_str)

                curr_script = curr_script[:start_old_func] + new_func_str + curr_script[end_old_func:]
                print(len(curr_script))
        # if it's a new function that's not within the curr script 
        else:
            print(f"Inserting {func_name}")
            curr_import_utils, curr_body = code_utils_split_code(curr_script)
            print(curr_import_utils)
            print()
            print(inspect.getsource(new_func))
            curr_script = curr_import_utils + "\n" + inspect.getsource(new_func) + "\n" + curr_body

    return curr_script

def code_utils_try_editor(
    func:Callable, 
    rng:chex.PRNGKey, 
    state:EnvState
    ) -> str | None:
    """
    Try running an editor function on a given state and rng.

    Args:
        func: callable function of the editor
        rng: a jax PRNGKey
        state: the maze state to be edited

    Returns:
        str | None: None if the editor function runs without errors, otherwise the error str
    """
    try:
        #logger.info(f'Start Testing Editor {func.__name__}')
        outstate = func(rng, state)
        chex.assert_trees_all_equal_dtypes(*(outstate, state))
        #logger.info(f'Editor {func.__name__} pased the test')
        return None
    except Exception as e:
        error = traceback.format_exc() + '\n' + str(e) 
        logger.info(f'Editor {func.__name__} failed the test')
        return error

def prompt_utils_form_designs(inputs:List[str]) -> str: 
    delimiter = "*"*25 + " Assistant Response{index} " + "*"*25 + '\n'
    out = ""
    for i, text in enumerate(inputs):
        out += delimiter.format(index=i) + text + "\n"
    return out.strip()

def prompt_utils_form_error_log(editor_logs:Dict[str, str]) -> Tuple[Dict[str, str], int]:
    """
    Format the error logs for each editor.

    Args:
        editor_logs (Dict[str, str]): a dictionary of {editor_name: error_log}
            where error_log is None if the editor function runs without errors, 
            otherwise the error log as a string.

    Returns:
        Tuple[Dict[str, str], int]: 
            a tuple where the first element is a dictionary of {editor_name: formatted_error_log}
            and the second element is the number of failed editors. 
            formatted_error_log is None if the editor function runs without errors, 
            otherwise the formatted error log as a string.
    """
    formatted_logs:Dict[str, str] = {}
    num_failed = 0 
    for editor_name, error in editor_logs.items():

        if not error:
            continue
        else:
            num_failed += 1 
            title = f"\n********* FUNCTION NAME: {editor_name} **********\n"
            error = code_utils_get_truncated_error(error, editor_name)
            formatted_log = f"{title}\n{error}\n*********"
            formatted_logs[editor_name] = formatted_log

    if not formatted_logs: 
        return formatted_logs, num_failed

    return formatted_logs, num_failed

def code_utils_test_editors(env_state:EnvState, script_path:str,) -> Tuple[Dict[str,str], int, Dict[str, Callable]]:

    module_path = code_utils_get_module_from_path(script_path)
    logger.info(module_path)
    func_map = {
        k:v for k,v in \
        (
            importlib.import_module(
                module_path
            ).__dict__
        ).items() \
        if 'mp_' in k
    }
    logger.info(f"Loaded {len(func_map)} functions")
    assert len(func_map) != 0 
    
    editor_logs = jax.tree_util.tree_map(
        lambda x: code_utils_try_editor(
            x, 
            jax.random.PRNGKey(125), 
            env_state
        ), 
        func_map
    )
    formatted_logs, num_failed = prompt_utils_form_error_log(editor_logs)
    logger.warning(f"{num_failed} out of {len(func_map)} failed the test")
    return formatted_logs, num_failed, func_map

class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        """
        Initializes an ImportVisitor instance.

        Attributes:
            imports (list): A list to store import statements encountered during AST traversal.
        """
        self.imports = []
    
    def visit_Import(self, node):
        """
        Visits an Import node in the Abstract Syntax Tree.

        Args:
            node (ast.Import): The Import node to visit.

        Notes:
            The visit_Import method iterates over the names of the Import node and
            for each name, it appends a dictionary containing the type of import
            (i.e., 'import'), the module name, and the alias name to the imports list.
        """
        
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname
            })
    
    def visit_ImportFrom(self, node):
        """
        Visits an ImportFrom node in the Abstract Syntax Tree.

        Args:
            node (ast.ImportFrom): The ImportFrom node to visit.

        Notes:
            The visit_ImportFrom method iterates over the names of the ImportFrom node and
            for each name, it appends a dictionary containing the type of import (i.e.,
            'from_import'), the full module path, the name of the module, and the alias name
            to the imports list.
        """
        module = node.module or ''
        level = node.level
        # Construct the full module path with relative dots
        if level > 0:
            full_module = '.' * level + module
        else:
            full_module = module
        for alias in node.names:
            self.imports.append({
                'type': 'from_import',
                'from_module': full_module,
                'name': alias.name,
                'alias': alias.asname
            })

def parse_imports(source_code:str) -> List[str]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return []
    visitor = ImportVisitor()
    visitor.visit(tree)
    
    imports_strs:List[str] = []
    for import_item in visitor.imports:

        if import_item['type'] == 'import':
            imports_strs.append(f"import {import_item['module']}")
        elif import_item['type'] == 'from_import':
            imports_strs.append(f"from {import_item['from_module']} import {import_item['name']}")

        if import_item['alias']:
            imports_strs[-1] += f"as {import_item['alias']}"

    return imports_strs

class LoggingHandler(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        logger.info("=> LLM Output: \n %s", response.generations[0][0].text)

class EditorScriptParser(BaseOutputParser[str]):
    """Simple mutators parser returning the whole script from LLM

    Args:
        BaseOutputParser (_type_): _description_
    """
    def parse(self, text: str) -> str: 
        # main pattern 
        main_pattern = r"```Python(.*?)(?:```|$)"
        main_regex = re.compile(main_pattern, re.DOTALL)
        main_matches = main_regex.findall(text)

        # candidate pattern 
        candidate_pattern = r"```python(.*?)(?:```|$)"
        candidate_regex = re.compile(candidate_pattern, re.DOTALL)
        candidate_matches = candidate_regex.findall(text)

        if not candidate_matches and not main_matches:
            raise ValueError(f"No Python matches found in the script: \n {text}")

        if not candidate_matches:
            matches = main_matches
        else:
            if not main_matches:
                matches = candidate_matches
            else:
                main_matches_max_len = max([len(x) for x in main_matches])
                candidate_matches_max_len = max([len(x) for x in candidate_matches])
                if candidate_matches_max_len > main_matches_max_len:
                    matches = candidate_matches
                else:
                    matches = main_matches

        # sort by length
        if len(matches) > 1:
            matches = sorted(matches, key=len, reverse=True)

        return matches[0].strip()