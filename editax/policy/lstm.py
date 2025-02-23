"Code reference to https://github.com/DramaCow/jaxued/blob/main/examples/maze_plr.py"
import numpy as np
import jax
import jax.numpy as jnp
import chex
import flax.linen as nn
import distrax
from flax.linen.initializers import orthogonal, constant

from typing import Sequence, Tuple, Any


class ResetLSTM(nn.Module):
    """This is a wrapper around an RNN that automatically resets the hidden state upon observing a `done` flag. 
    In this way it is compatible with the jax-style RL loop where episodes automatically end/restart.
    """
    cell: nn.OptimizedLSTMCell

    @nn.compact
    def __call__(
        self,
        inputs: Tuple[chex.Array, chex.Array], # (x, resets)
        *,
        initial_carry: Any | None = None,
        reset_carry: Any | None = None,
    ) -> Tuple[Any, Any]:
        """
        Applies the ResetLSTM module to the input data.

        This method processes input sequences using an LSTM cell and automatically 
        resets the hidden state when a 'done' flag is encountered. If no initial 
        carry is provided, it initializes or resets the carry state using the 
        LSTM cell's `initialize_carry` method.

        Args:
            inputs: A tuple containing the input data `x` and a `resets` flag 
                    indicating when to reset the hidden state.
                    x: is an array of shape (num_steps, num_parallel_envs, in_dim)
                    resets: is an array of shape (num_steps, num_parallel_envs)

            initial_carry: The initial carry state for the LSTM. Defaults to 
                        `None`, in which case `reset_carry` is used.
            reset_carry: The carry state to reset to when a 'done' flag is 
                        encountered. Defaults to `None`, in which case a 
                        new carry state is initialized.

        Returns:
            A tuple containing the final carry state and the output of the LSTM 
            after processing the input data.
        """
        # On episode completion, model resets to this
        if reset_carry is None:
            reset_carry = self.cell.initialize_carry(jax.random.PRNGKey(0), inputs[0].shape[1:])
        carry = initial_carry if initial_carry is not None else reset_carry

        def scan_fn(cell, carry, inputs):
            x, resets = inputs
            carry = jax.tree_util.tree_map(
                lambda a, b: jnp.where(resets[:, jnp.newaxis], a, b), 
                reset_carry, 
                carry
            )
            return cell(carry, x)

        scan = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0, #scan starting from "inputs", iterating through the most outer dimension 
            out_axes=0,
        )

        return scan(self.cell, carry, inputs)

class EditorActorCritic(nn.Module):
    """
    A model that uses an LSTM to process sequential data and produce an
    output. The output is then passed through a 2-layer actor and a
    2-layer critic to produce a policy and a value function.
    """
    hidden_dim: int
    action_dim: Sequence[int]

    @nn.compact
    def __call__(
        self, 
        inputs:Tuple[chex.Array, chex.Array], 
        hidden_state:chex.Array
    ) -> Tuple[chex.Array, distrax.Distribution, chex.Array]:
        """
        The forward pass of the RNNActorCritic model.

        This model uses an RNN to process sequential data and produce an
        output. The output is then passed through a 2-layer actor and a
        2-layer critic to produce a policy and a value function.

        Args:
            inputs: A tuple containing the input data and a 'done' flag.
                    The input data is a sequence of shape
                    (num_steps, num_parallel_envs, in_dim) and the 'done' flag
                    is a boolean array of shape (num_steps, num_parallel_envs).
            hidden: The initial hidden state of the RNN.
                    (num_parallel_envs, hidden_dim)

        Returns:
            A tuple containing the final hidden state of the RNN, the policy
            and the value function.
        """
        x, dones = inputs 
        
        # base layer 
        base_activation = nn.relu
        embed = nn.Dense(
            self.hidden_dim, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        embed = base_activation(embed)

        embed = nn.Dense(
            self.hidden_dim, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        embed = base_activation(embed)

        # recurrent layer
        rnn_in = (embed, dones)
        hidden, x = ResetLSTM(
            nn.OptimizedLSTMCell(features=self.hidden_dim,)
        )(
            rnn_in, initial_carry=hidden_state,
        )

        ### 2 layers of actor #### 
        head_activation = nn.tanh

        actor_mean = nn.Dense(
            self.hidden_dim//2, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        actor_mean = head_activation(actor_mean)

        actor_mean = nn.Dense(
            self.hidden_dim//2, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = head_activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        #### 2 layers of critic ####
        critic = nn.Dense(
            self.hidden_dim//2, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        critic = head_activation(critic)

        critic = nn.Dense(
            self.hidden_dim//2, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(critic)
        critic = head_activation(critic)

        critic = nn.Dense(
            1, 
            kernel_init=orthogonal(1.0), 
            bias_init=constant(0.0))(
            critic
        )
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @classmethod
    def initialize_carry(
        cls, 
        batch_dims:Sequence[int], 
        hidden_dim:int
    ) -> Tuple[chex.Array, chex.Array]:
        """Initialize LSTM carry state.
        
        Args:
            batch_dims: Sequence of batch dimensions
            hidden_dim: Size of hidden dimension to use
            
        Returns:
            Initial carry state for LSTM
        """
        return nn.OptimizedLSTMCell(features=hidden_dim).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, hidden_dim)
        )