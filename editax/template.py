"""
This module contains template classes and functions for generating environment editors.

The main class EditorMaker provides templates for prompts used to generate
environment modifications through large language models.
"""
from textwrap import dedent

EDITOR_DEFINITION = dedent(
"""
## Minimal Meaningful Perturbation (MMP)
We define a Minimal Meaningful Perturbation (MMP) as a function that have the following attributes:
```
* Must perform non-trivial operations at the most atomic level.
Either:
    * Make the simplest changes to slightly reduce the difficulty of the environment in particular aspects.
Or:
    * Make the simplest changes to slightly increase the difficulty of the environment in particular aspects
```
"""
)

class EditorMaker: 
    """
    This class provides templates for prompts used to generate environment modifications through large language models.
    """
    role = dedent(
    """ 
    ## Role 
    You are an exceptionally creative Curriculum Designer and Reinforcement Learning Environment Engineer.
    """
    )

    env = dedent(
    """
    ## Environment
    Here is the source code of a RL environment, representing an Under-specified Partially Observable Markov Decision Process (UPOMDP).
    ```python
    {input_string}   
    ```
    {engine_statement} 
    """
    ).strip()

    task = dedent(
    EDITOR_DEFINITION + \
    """
    ## Task
    1. Identify skills that a general policy must master to solve any particular POMDP sampled from such UPOMDP.
    2. According to the skills required, identify a minimum necessary set of states that are most relevant to harnessing the identified skills.
    3. Produce a minimal necessary set of Minimal Meaningful Perturbation (MMP) functions that covers such minimum necessary set of states.

    ## Execution
    * Your output should include a planning section as well as an end-to-end self-contained Python script without any illustrative examples.
    * Start with MMPs that reduce the environment's challenge and gradually move to MMPs that increase the challenge.
    * Different MMPs must:
        * Be unique
        * Not cancel out each other's operations in a deterministic way
    """
    ).strip()

    input_output = dedent(
    """
    ## Input and Output
    Each MMP is represented as a Python function written in JAX, which operates on an input {env_var} and returns a modified {env_var}.
    Each {env_var} is a Python variable that represents a POMDP.

    Each mmp function receives only two input arguments:  
        arg1: rng key
        arg2: {env_var}: {env_var_type} (representing the input POMDP)
    
    only returns the modified {env_var} (representing the edited POMDP)
    """    
    ).strip()

    coding_requirements = dedent(
    """    
    ## Coding Requirements
    1. Use @jax.jit as the decorator for MMP functions and all probabilistic operations must be seeded by an rng key.
    2. Import all variables explicitly, DO NOT comment out any imports.
    3. Take a pragmatic approach, allowing the edits to be done with the correct intent, instead of searching for the most elegant way of writing a MMP.
    4. Use of variables,functions and classes from the source code whenever possible.
    5. Utilize computational techniques (like masking and immutable updates). Make only 1 attempt to perform the edit per MMP, avoid any exhaustive loops.
    6. Must quote your code inside ```Python...``` blocks.
    """
    ).strip()

    format_ = dedent(
    """
    ## Output format 
    [PLAN STARTS HERE]
    ...
    [PLAN ENDS HERE]

    ```Python
    # imports 
    import jax
    import jax.numpy as jnp
    import chex 
    import flax
    ...

    ## utility functions 
    ...

    # mmps
    @jax.jit
    def mmp_<name>(rng:chex.PRNGKey, {env_var}:{env_var_type}) -> {env_var_type}:
        '''
        doc string
        '''
        # code ...

    @jax.jit
    def mmp_<name>(rng:chex.PRNGKey, {env_var}:{env_var_type}) -> {env_var_type}:
        '''
        doc string
        '''
        # code ...
    ...
    ```
    """
    ).strip()

    synthesize = dedent(
    """
    ## Synthesize
    Given all previous responses that has been produced, synthesize a single version based on majority consensus.
    This version should be the most consistent and comprehensive version of all previous responses.
    
    You must supply a short justification for why each function was picked in the final version.
    """
    ).strip()

    @classmethod 
    def get_system_template(cls, ) -> str:
        """
        Returns the system template which is the role of the AI.

        Returns:
            str: The system template
        """
        return cls.role 

    @classmethod 
    def get_human_template(cls, ) -> str:
        """
        Returns the human template which is the environment, task, input and output.

        Returns:
            str: The human template
        """
        return cls.env + '\n\n' +  cls.task  + '\n\n' + cls.input_output + '\n\n' + cls.coding_requirements + '\n\n' + cls.format_
    
    @classmethod
    def get_full(cls, )-> str:

        """
        Returns the full template which is the system and human templates concatenated.

        Returns:
            str: The full template
        """
        return cls.get_system_template() + '\n\n' + cls.get_human_template()
    
class EditorCorrector:

    role = dedent(
    """
    ## Role
    You are an exceptionally skilled Reinforcement Learning Environment Engineer. 
    You can debug and write high-quality JIT-compilable code in JAX.
    """
    ).strip()

    source_augmented_prev_code = EditorMaker.env + '\n\n' + EDITOR_DEFINITION + '\n\n' + dedent(
    """
    You've previously implemented the following MMP functions:
    ```python 
    {editors_script}
    ```

    ## Traceback Errors
    Here are some execution errors exposed
    ```txt
    {error_logs}
    ```
    """
    )

    prev_code = dedent(
    """
    # Previous implementation
    Your previous implementation:
    ```python 
    {editors_script}
    ```

    # Execution errors:
    ```txt
    {error_logs}
    ```
    """
    ).strip()

    fix_task = dedent(
    """
    # Task 
    You are tasked to generate the corrected version of your previous implemented function.
    1. Examine the execution errors to identify the source of the error from the code.
    2. Produce corrected function in a self-contained python script. 
        During modification:
        * Only correct the mistake. Don't introduce new errors or modify the intended functionality of the function.
        * Keep the @jax.jit decorator, the input and output arguments of the editor functions unchanged.
        * Keep the attributes and structure of the input and output variables unchanged.
        * Your code will be consumed directly, so write full & complete implementation without any illustrative examples. You must quote your code inside ```Python...``` blocks.
        * Import all required variables explicitly.
        * Add utility functions if needed.
    3. Insert the name of updated/added function into a list at the end of a script called "updates"
    """
    ).strip()

    regen_task = dedent(
    """
    # Task 
    You are tasked to re-generate the previous implemented function (with the same functionality, input and output).
    The implemented function has failed test multiple tests, please regenerate such function by starting from scratch via a different approach.
    During regeneration:
        * Only correct the mistake. Don't introduce new errors or modify the intended functionality of the function.
        * Keep the @jax.jit decorator, the input and output arguments of the editor functions unchanged.
        * Keep the attributes and structure of the input and output variables unchanged.
        * Your code will be consumed directly, so write full & complete implementation without any illustrative examples. You must quote your code inside ```Python...``` blocks.
        * Import all required variables explicitly.
        * Add utility functions if needed.
    3. Insert the name of updated/added function into a list at the end of a script called "updates"
    """
    ).strip()

    format_  = dedent(
    """
    ## Output format 
    [PLAN STARTS HERE]
    ```txt
    ...
    ```
    [PLAN ENDS HERE]
    
    ```Python
    from typing import List
    import jax 
    import jax.np as jnp 
    import chex 
    import flax 
    ...

    # utility functions
    ...

    # new editor functions 
    ...

    updates:List[str] = [
        <str>,
        <str>,
    ]

    ```
    """).strip()

    @classmethod
    def get_system_template(cls, ) -> str:
        return cls.role

    @classmethod 
    def get_fix_template(cls, ) -> str:
        return cls.prev_code + '\n\n' + cls.fix_task  + '\n\n' + cls.format_
    
    @classmethod 
    def get_regen_template(cls, ) -> str:
        return cls.prev_code + '\n\n' + cls.regen_task  + '\n\n' + cls.format_
    
    @classmethod
    def get_source_augmented_fix(cls, ) -> str:
        return cls.source_augmented_prev_code + '\n\n' + cls.get_fix_template()
    
    @classmethod
    def get_source_augmented_regen(cls, ) -> str:
        return cls.source_augmented_prev_code + '\n\n' + cls.get_regen_template()

    @classmethod
    def get_fix_full(cls, source_augmented:bool = False)-> str:
        return cls.get_system_template() + '\n\n' + \
            cls.get_fix_template() if not source_augmented else cls.get_source_augmented_fix()
    
    @classmethod
    def get_regen_full(cls, source_augmented:bool = False)-> str:
        return cls.get_system_template() + '\n\n' + \
            cls.get_regen_template() if not source_augmented else cls.get_source_augmented_regen()
    

class EditorMakerComprehensive(EditorMaker):

    synthesize = dedent(
    """
    ## Synthesize
    Given all previous responses that has been produced, synthesize a single version that has the following properties:
    (1) includes all mmps that appeared frequently across different responses
    (2) includes all mmps that are novel to each individual responses
    (3) Contain up to 10 mmps
    """
    ).strip()


class EditorMakerOverlooked(EditorMaker):

    synthesize = dedent(
    """
    ## Synthesize
    Given all previous responses that has been produced, synthesize a single version that has the following properties:
    (1) includes all mmps that appeared frequently across different responses
    (2) includes all mmps that are novel to each individual responses
    (3) includes a pair of mmps that represent areas overlooked by all existing responses.
    (4) Contain up to 10 mmps
    """
    ).strip()