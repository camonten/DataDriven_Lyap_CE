"""
Utilities for experiments and plotting (neural network functions, result formatting).

Authors: Carlos A. Montenegro (Hybrid Systems Laboratory, UC Santa Cruz, GitHub: camonten)
         Santiago Jimenez Leudo (Hybrid Systems Laboratory, UC Santa Cruz, GitHub: sjleudo)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from functools import partial, reduce
from math import sqrt
import matplotlib
from matplotlib.ticker import ScalarFormatter


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self) -> None:
        """Set the scalar format to two decimal places."""
        self.format = "%1.2f"


def latexify(
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    columns: int = 1,
) -> None:
    """
    Configure matplotlib's RC parameters for LaTeX plotting.

    Parameters:
        fig_width (Optional[float]): Figure width in inches. Defaults based on number of columns.
        fig_height (Optional[float]): Figure height in inches. Defaults to the golden ratio times width.
        columns (int): Number of columns (1 or 2).

    Notes:
        Warns if the computed fig_height exceeds the maximum allowed.
    """
    assert columns in [1, 2]
    fig_width = fig_width or (3.39 if columns == 1 else 6.9)
    fig_height = fig_height or (fig_width * ((sqrt(5) - 1) / 2))
    if fig_height > 8.0:
        print(
            f"WARNING: fig_height too large: {fig_height} so will reduce to 8.0 inches."
        )
        fig_height = 8.0
    matplotlib.rcParams.update(
        {
            "text.latex.preamble": "\n".join(
                [r"\usepackage{gensymb}", r"\usepackage{amsfonts}"]
            ),
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "font.size": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "text.usetex": True,
            "figure.figsize": [fig_width, fig_height],
            "font.family": "serif",
        }
    )


def hausdorff_distance(S: np.ndarray, C: np.ndarray) -> float:
    """
    Compute the directed Hausdorff distance from set S to set C:
        d_H(S, C) = sup_{s in S} inf_{c in C} ||s - c||.

    Parameters:
        S (np.ndarray): Array of shape (N, d) for set S.
        C (np.ndarray): Array of shape (M, d) for the covering centers.

    Returns:
        float: The Hausdorff distance from S to C.
    """
    # Compute all pairwise distances (N x M)
    dists = np.linalg.norm(S[:, None, :] - C[None, :, :], axis=2)
    return np.max(np.min(dists, axis=1))


class NeuralNet:
    def __init__(
        self,
        net_dims: List[int],
        args: Any,
        dyn: Any,
        opt_name: str,
        act_fun: Callable[[jnp.ndarray], jnp.ndarray],
        ell_C: Optional[Callable[[jnp.ndarray], float]] = None,
        ell_D: Optional[Callable[[jnp.ndarray], float]] = None,
        opt_kwargs: Dict[str, Any] = {},
    ) -> None:
        """
        Initialize the NeuralNet with architecture, dynamics, optimizer, and activation function.

        Parameters:
            net_dims (List[int]): List of layer dimensions.
            args (Any): Hyperparameters and training settings.
            dyn (Any): Hybrid dynamics with flow and jump maps.
            opt_name (str): Optimizer name ('Adam', 'Adagrad', 'Nesterov', 'SGD', etc.).
            act_fun (Callable[[jnp.ndarray], jnp.ndarray]): Activation function (e.g., jax.nn.relu).
            ell_C (Optional[Callable[[jnp.ndarray], float]]): Cost function for flows (optional).
            ell_D (Optional[Callable[[jnp.ndarray], float]]): Cost function for jumps (optional).
            opt_kwargs (Dict[str, Any]): Additional keyword arguments for optimizer initialization.
        """
        self.args, self.opt_name, self.net_dims, self.act_fun = (
            args,
            opt_name,
            net_dims,
            act_fun,
        )
        self._init_optimizer(opt_kwargs)
        self.key = jrandom.PRNGKey(5433)
        self.forward = jax.vmap(self.forward_indiv, in_axes=(0, None))
        print(
            "Training for practical pAS!"
            if (ell_C is None and ell_D is None)
            else "Training for cost upper bound!"
        )
        self.ell_C, self.ell_D, self.dyn = ell_C, ell_D, dyn
        self.lam_C, self.lam_D, self.mu = args.lam_C, args.lam_D, args.mu

    def __call__(self, x: jnp.ndarray, params: Any) -> jnp.ndarray:
        """
        Perform a forward pass through the network.

        Parameters:
            x (jnp.ndarray): Input data.
            params (Any): Neural network parameters.

        Returns:
            jnp.ndarray: Network output for the given input.
        """
        return self.forward(x, params)

    @partial(jax.jit, static_argnums=0)
    def step(self, epoch: int, opt_state: Any, dataset: Dict[str, jnp.ndarray]) -> Any:
        """
        Perform one optimization step.

        Parameters:
            epoch (int): Current training epoch.
            opt_state (Any): Current optimizer state containing network parameters.
            dataset (Dict[str, jnp.ndarray]): Training dataset.

        Returns:
            Any: Updated optimizer state after one step.
        """
        p = self.get_params(opt_state)
        grads = jax.grad(self.loss, argnums=0)(p, dataset)
        return self.opt_update(epoch, grads, opt_state)

    @partial(jax.jit, static_argnums=0)
    def loss(self, params: Any, dataset: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute the loss for the given parameters and dataset.

        Parameters:
            params (Any): Neural network parameters.
            dataset (Dict[str, jnp.ndarray]): Training dataset.

        Returns:
            jnp.ndarray: Computed loss value.
        """
        return self.loss_and_constraints(params, dataset)[0]

    @partial(jax.jit, static_argnums=0)
    def constraints(
        self, params: Any, dataset: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute the constraint percentages for flows and jumps.

        Parameters:
            params (Any): Neural network parameters.
            dataset (Dict[str, jnp.ndarray]): Training dataset.

        Returns:
            Dict[str, jnp.ndarray]: Dictionary with percentages for 'flows' and 'jumps' constraints.
        """
        return self.loss_and_constraints(params, dataset)[1]

    @partial(jax.jit, static_argnums=0)
    def loss_and_constraints(
        self, params: Any, dataset: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute the augmented Lagrangian loss and constraint percentages.

        Parameters:
            params (Any): Neural network parameters.
            dataset (Dict[str, jnp.ndarray]): Training dataset.

        Returns:
            Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                A tuple (loss, constraints) where:
                  - loss is the computed loss value.
                  - constraints is a dict with 'flows' and 'jumps' constraint percentages.
        """
        fgrad = lambda x: jax.vmap(self.model_grad_indiv, in_axes=(0, None))(x, params)
        fcons = lambda x: jax.vmap(self.flows_constraint_indiv, in_axes=(0, None))(
            x, params
        )
        jcons = lambda x: jax.vmap(self.jumps_constraint_indiv, in_axes=(0, None))(
            x, params
        )
        penalty = lambda v: jnp.sum(jnn.relu(v) ** 2)
        constraint_pct = (
            lambda v: (1 - jnp.sum(jnp.heaviside(v, 0)) / v.shape[0]) * 100.0
        )

        param_loss = jnp.sum(jnp.linalg.norm(ravel_pytree(params)[0]))
        flows = fcons(dataset["x_flows"]) + self.args.gam_C
        jumps = jcons(dataset["x_jumps"]) + self.args.gam_D
        dV_loss = jnp.sum(
            jnp.linalg.norm(fgrad(jnp.vstack((dataset["x_flows"], dataset["x_jumps"]))))
        )
        s_flows = jnn.relu(-flows - self.mu * self.lam_C)
        s_jumps = jnn.relu(-jumps - self.mu * self.lam_D)
        loss = (
            self.args.lam_grad * dV_loss
            + self.args.lam_param * param_loss
            - jnp.sum(self.lam_C * (-flows - s_flows))
            - jnp.sum(self.lam_D * (-jumps - s_jumps))
            + 1
            / (2 * self.mu)
            * (jnp.sum((-flows - s_flows) ** 2) + jnp.sum((-jumps - s_jumps) ** 2))
        )
        return loss, {"flows": constraint_pct(flows), "jumps": constraint_pct(jumps)}

    def flows_constraint_indiv(self, x: jnp.ndarray, params: Any) -> jnp.ndarray:
        """
        Compute the Lyapunov flows constraint for a single input.

        Parameters:
            x (jnp.ndarray): Input data point.
            params (Any): Neural network parameters.

        Returns:
            jnp.ndarray: Computed flow constraint value.
        """
        return jnp.dot(self.dyn.f(x), self.model_grad_indiv(x, params)) + (
            self.ell_C(x) if self.ell_C is not None else 0
        )

    def jumps_constraint_indiv(self, x: jnp.ndarray, params: Any) -> jnp.ndarray:
        """
        Compute the Lyapunov jumps constraint for a single input.

        Parameters:
            x (jnp.ndarray): Input data point.
            params (Any): Neural network parameters.

        Returns:
            jnp.ndarray: Computed jump constraint value.
        """
        return (
            self.forward_indiv(self.dyn.g(x), params, self.act_fun)
            - self.forward_indiv(x, params, self.act_fun)
            + (self.ell_D(x) if self.ell_D is not None else 0)
        )

    def model_grad_indiv(self, x: jnp.ndarray, params: Any) -> jnp.ndarray:
        """
        Compute the gradient of the forward pass with respect to the input for a single instance.

        Parameters:
            x (jnp.ndarray): Input data point.
            params (Any): Neural network parameters.

        Returns:
            jnp.ndarray: Gradient of the network output with respect to the input.
        """
        return jax.grad(self.forward_indiv)(x, params, self.act_fun)

    @staticmethod
    def forward_indiv(
        x: jnp.ndarray,
        params: List[Any],
        act_fun: Callable[[jnp.ndarray], jnp.ndarray],
        eps: float = 0.1,
    ) -> jnp.ndarray:
        """
        Compute the forward pass through the neural network for a single instance.

        Parameters:
            x (jnp.ndarray): Input vector.
            params (List[Any]): List of layer parameters.
            act_fun (Callable[[jnp.ndarray], jnp.ndarray]): Activation function.
            eps (float): Small constant for numerical stability.

        Returns:
            jnp.ndarray: Scalar output of the network.
        """
        activations = reduce(
            lambda a, p: act_fun(
                a.dot(jnp.hstack((p[0].T @ p[0] + eps * jnp.eye(p[0].shape[1]), p[1])))
            ),
            params,
            x,
        )
        return jnp.squeeze(jnp.linalg.norm(activations))

    @staticmethod
    def random_layer_params(
        m: int, n: int, key: jax.random.KeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Randomly initialize parameters for a neural network layer.

        Parameters:
            m (int): Input dimension.
            n (int): Output dimension.
            key (jax.random.KeyArray): JAX random number generator key.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing a randomly initialized weight matrix and bias vector.
        """
        q = int(np.ceil(0.5 * (m + 1)))
        w_key, b_key = jrandom.split(key)
        return jrandom.normal(w_key, (q, m)), jrandom.normal(b_key, (m, n - m))

    def init_params(
        self, verbose: bool = False
    ) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Initialize the neural network parameters randomly.

        Parameters:
            verbose (bool): If True, prints parameter shapes.

        Returns:
            List[Tuple[jnp.ndarray, jnp.ndarray]]: List of randomly initialized parameters for each layer.
        """
        if not np.all(np.diff(self.net_dims) >= 0):
            raise ValueError(
                "Each layer must maintain or increase the dimension of its input!"
            )
        keys = jrandom.split(self.key, len(self.net_dims))
        params = [
            self.random_layer_params(m, n, k)
            for m, n, k in zip(self.net_dims[:-1], self.net_dims[1:], keys)
        ]
        if verbose:
            print(f"Param shapes are: {[(p1.shape, p2.shape) for p1, p2 in params]}")
        return params

    def _init_optimizer(self, kwargs: Dict[str, Any]) -> None:
        """
        Initialize the optimizer based on the chosen optimization algorithm.

        Parameters:
            kwargs (Dict[str, Any]): Keyword arguments for the optimizer.

        Raises:
            NotImplementedError: If the provided optimizer name is not supported.
        """
        opts: Dict[str, Callable[..., Any]] = {
            "SGD": optimizers.sgd,
            "Adam": optimizers.adam,
            "Adagrad": optimizers.adagrad,
            "Nesterov": optimizers.nesterov,
        }
        if self.opt_name in opts:
            self.opt_init, self.opt_update, self.get_params = opts[self.opt_name](
                **kwargs
            )
        else:
            raise NotImplementedError(
                f'Supported optimizers: {" | ".join(opts.keys())}'
            )


class AverageMeter:
    """
    Class for tracking and updating average values of a metric.
    """

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """
        Initialize an AverageMeter.

        Parameters:
            name (str): Name of the metric.
            fmt (str): Format string for printing the metric.
        """
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self) -> None:
        """Reset the meter's current value, sum, count, and average."""
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.

        Parameters:
            val (float): New value to update.
            n (int): Weight of the new value (default is 1).
        """
        self.val, self.sum, self.count = val, self.sum + val * n, self.count + n
        self.avg = self.sum / self.count
