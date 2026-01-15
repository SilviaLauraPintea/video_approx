################################################################################
# The code builds on the code of:
# Nguyen, Timothy, Zhourong Chen, and Jaehoon Lee. "Dataset meta-learning from kernel ridge-regression." ICLR, 2021.
# https://colab.research.google.com/github/google-research/google-research/blob/master/kip/KIP.ipynb
# All credit for third party software/code is with the authors.
# The authors' license is included here <LICENSE-2.0.txt>
################################################################################
### The code has been adapted and modified by Silvia L. Pintea to optimize MMD and use different kernel options.

import functools
import numpy as np
import math
import jax
from jax import numpy as jnp
import optax


def make_update_fn(args, init_params, kernel_fn_list):
    """
    Defines the model update function, optimizer, parameters.
    """
    solver = optax.adam(args.learning_rate)
    opt_state = solver.init(init_params)

    loss_fn = make_loss_fn(args, kernel_fn_list)
    value_and_grad = jax.value_and_grad(
        lambda params, vi, ls, assign: loss_fn(
            vm=params["vm"], vi=vi, ls=ls, assign=assign
        ),
        has_aux=True,
    )

    @jax.jit
    def update_fn(step, opt_state, params, vi, ls, assign):
        (loss, extra), dparams = value_and_grad(params, vi, ls, assign)

        updates, opt_state = solver.update(dparams, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, (loss, extra)

    return opt_state, update_fn


def make_loss_fn(args, kernel_fn_list):
    """
    Defines the loss function, and that is MMD^2 over the kernels of real (vi) and approx-frames (vm).
    """

    @jax.jit
    def loss_fn(vm, vi, ls, assign):
        vi = vi.reshape(vi.shape[0], -1)
        vm = vm.reshape(vm.shape[0], -1)

        def spl(v1, v2):
            """Gaussian kernel with median length-scale."""
            cdist = jnp.sum((v1[:, None] - v2[None, :]) ** 2, -1)
            nonlocal ls
            ls = jax.lax.stop_gradient(ls)
            kernel_matrix = jnp.exp(-cdist / (ls**2))
            return kernel_matrix

        all_k_nm = None
        all_k_im = None
        all_k_ij = None
        for kfn in list(kernel_fn_list.keys()):
            kernel_fn = kernel_fn_list[kfn]
            kernel_name = kfn

            # NTK / NNGP / Sphere NTK
            if (
                kernel_name.endswith("ntk")
                or kernel_name.startswith("nngp")
                or kernel_name.startswith("sphere")
            ):
                if kernel_name.startswith("sntk"):
                    w = jnp.array(0.1 * np.random.randn(vm.shape[1], vm.shape[1]))
                    b = jnp.array(
                        np.random.uniform(low=0, high=2.0 * np.pi, size=vm.shape[1])
                    )
                    w = jax.lax.stop_gradient(w)
                    b = jax.lax.stop_gradient(b)

                    vm = jnp.sqrt(0.1 / vm.shape[1]) * jnp.cos(vm @ w + b)
                    vi = jnp.sqrt(0.1 / vi.shape[1]) * jnp.cos(vi @ w + b)
                elif kernel_name.startswith("sphere"):
                    vi = vi / (jnp.linalg.norm(vi, axis=0) + 1e-7)
                k_nm = kernel_fn(vm, vm)
                k_im = kernel_fn(vi, vm)
                k_ij = kernel_fn(vi, vi)

            # Gaussian kernel is characteristic
            elif kernel_name.startswith("spl"):
                k_ij = spl(vi, vi)
                k_im = spl(vi, vm)
                k_nm = spl(vm, vm)

            if all_k_ij is not None:
                alpha = jnp.abs(
                    jnp.median(all_k_ij) / jnp.median(k_ij)
                )  # Scaling so the ranges are comparable
                all_k_nm *= alpha * k_nm
                all_k_im *= alpha * k_im
                all_k_ij *= alpha * k_ij
            else:
                all_k_nm = k_nm
                all_k_im = k_im
                all_k_ij = k_ij

        # The MMD^2 equation (not the unbiased one):
        mmd = jnp.sqrt(
            jnp.mean(all_k_nm) + jnp.mean(all_k_ij) - 2.0 * jnp.mean(all_k_im)
        )
        reg = jnp.linalg.norm(vm)
        return mmd + args.wd * reg, all_k_im

    return loss_fn
