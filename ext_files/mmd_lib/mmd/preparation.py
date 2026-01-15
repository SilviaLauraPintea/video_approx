################################################################################
# The code builds on the code of:
# Nguyen, Timothy, Zhourong Chen, and Jaehoon Lee. "Dataset meta-learning from kernel ridge-regression." ICLR, 2021.
# https://colab.research.google.com/github/google-research/google-research/blob/master/kip/KIP.ipynb
# All credit for third party software/code is with the authors.
# The authors' license is included here <LICENSE-2.0.txt>
################################################################################
### The code has been adapted and extended by Silvia L. Pintea.

import numpy as np
from sklearn.cluster import KMeans

from utils.auxiliary import quantify_frames
from mmd.mmd import make_update_fn, make_loss_fn


def get_normalization(v_train: np.array, axis: int = 0):
    """
    Gets the mean and std of one video over time.
    """
    v_means = v_train.mean(axis=axis, keepdims=True)
    v_stds = v_train.std(axis=axis, keepdims=True) + 1e-7  # To avoid division by 0
    return v_means, v_stds


def normalization(
    v_input: np.array, v_means: np.array = None, v_stds: np.array = None, axis: int = 0
):
    """
    normalize the current batch / approx frames.
    """
    if v_means is None or v_stds is None:
        v_means, v_stds = get_normalization(v_train=v_input, axis=axis)
    norm_v = (
        np.copy(v_input) - v_means.repeat(v_input.shape[axis], axis=axis)
    ) / v_stds.repeat(v_input.shape[axis], axis=axis)
    return norm_v


def init_model(args, v_train, v_means, v_stds, kernel_fn, kmeans_init=False):
    """
    Initializes the model.
    Returns the update function, optimizer, and parameters.
    """
    if kmeans_init:
        means = KMeans(n_clusters=args.approx_size).fit(v_train)
        v_init = means.cluster_centers_
    else:
        v_init, _ = batch_sample(
            batch_size=args.approx_size,
            approx_size=args.approx_size,
            v_train=v_train,
            approx=True,
        )
    v_init = normalization(v_input=v_init, v_means=v_means, v_stds=v_stds)

    params = {"vm": v_init}

    # Model update function/params/optimizer
    opt_state, update_fn = make_update_fn(
        args=args, init_params=params, kernel_fn_list=kernel_fn
    )

    # Loss function
    loss_fn = make_loss_fn(args=args, kernel_fn_list=kernel_fn)
    return opt_state, update_fn, loss_fn, params


def batch_sample(
    batch_size: int,
    approx_size: int,
    v_train: np.ndarray,
    approx: bool,
):
    """
    Gets the approx-frames initialized with the uniform segment means, or a batch of random samples.
    """
    segments_id = np.arange(0, approx_size)
    sample = {"idx": np.arange(0, v_train.shape[0]) / v_train.shape[0]}
    segments = (quantify_frames(sample=sample, n_segments=approx_size)).astype(np.int32)

    if approx:
        # Take the mean per segment
        v_batch = np.concatenate(
            [
                np.mean(v_train[segments == s], axis=0, keepdims=True)
                if sum(segments == s) > 1
                else v_train[np.random.randint(0, v_train.shape[0]), :].reshape(1, -1)
                for s in segments_id
            ]
        )
        inds = np.arange(0, approx_size)
    else:
        # Select equal number of samples per segment
        n_per_segment = batch_size // approx_size
        inds = np.concatenate(
            [
                np.random.choice(
                    np.where(segments == s)[0], n_per_segment, replace=True
                )
                for s in segments_id
            ]
        )

        # Add missing indices
        rest = batch_size - (approx_size * n_per_segment)
        inds = np.array(
            inds.tolist()
            + np.random.choice(v_train.shape[0] - 1, rest, replace=True).tolist()
        )

        # Shuffle the batch
        v_batch = v_train[inds].copy()
    return v_batch, inds
