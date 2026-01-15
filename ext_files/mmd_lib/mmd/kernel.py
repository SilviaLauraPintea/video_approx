################################################################################
# The code builds on the code of:
# Nguyen, Timothy, Zhourong Chen, and Jaehoon Lee. "Dataset meta-learning from kernel ridge-regression." ICLR, 2021.
# https://colab.research.google.com/github/google-research/google-research/blob/master/kip/KIP.ipynb
# All credit for third party software/code is with the authors.
# The authors' license is included here <LICENSE-2.0.txt>
################################################################################
### The code has been adapted and modified by Silvia L. Pintea.

import math
import neural_tangents as nt
from neural_tangents import stax
import functools
import jax


def init_kernel(args, kern):
    """
    Defines and initializes the kernel function.
    """
    # Stax returns: init_fn, apply_fn, kernel_fn
    init_fn, apply_fn, kernel_fn = get_kernel_fn(
        depth=args.depth,
        in_channels=args.input_channels,
        out_channels=1,
        W_std=2,
        b_std=0.1,
        parameterization="ntk",
        activation=args.activation,
    )
    if kern.startswith("nngp"):
        kernel_fx = jax.jit(functools.partial(kernel_fn, get="nngp"))
    else:
        kernel_fx = jax.jit(functools.partial(kernel_fn, get="ntk"))
    return kernel_fx


def get_kernel_fn(
    depth,
    in_channels,
    out_channels,
    W_std,
    b_std,
    parameterization="ntk",
    activation="relu",
):
    """
    Defines the network architecture.
    """
    if activation.startswith("relu"):
        activation_fn = stax.Relu()
    elif activation.startswith("tahn"):
        activation_fn = stax.ostax.Tanh()
    elif activation.startswith("cos"):
        activation_fn = stax.Cos()

    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization
    )

    layers = [stax.Flatten()]
    for _ in range(depth):
        layers += [dense(1024), activation_fn]

    layers += [
        stax.Dense(
            out_channels, W_std=W_std, b_std=b_std, parameterization=parameterization
        )
    ]
    return stax.serial(*layers)
