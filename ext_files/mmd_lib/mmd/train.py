### Author: Silvia L. Pintea
# It performs the main training loop and the inference loop.

import os
import time
import numpy as np
import jax
from jax import numpy as jnp
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from mmd.preparation import (
    normalization,
    batch_sample,
)
from utils.auxiliary import quantify_frames
from utils.eval_script import evaluate


def start_run(
    args,
    gts,
    v_train,
    v_means,
    v_stds,
    opt_state,
    params,
    update_fn,
    loss_fn,
    video_idx,
):
    """
    Runs the video labeling algorithms a-to-z.
    """
    # Do the labeling --------
    if args.what.startswith("mmd"):
        v_approx, k_im = train(
            args=args,
            gts=gts,
            num_steps=(
                args.iter
                if args.batch_size <= 0
                else args.iter * (v_train.shape[0] // args.batch_size)
            ),
            v_train=v_train,
            v_means=v_means,
            v_stds=v_stds,
            opt_state=opt_state,
            params=params,
            update_fn=update_fn,
            loss_fn=loss_fn,
            log_freq=(
                args.iter + 1
                if args.batch_size <= 0
                else (args.iter + 1) * (v_train.shape[0] // args.batch_size)
            ),
            video_idx=video_idx,
        )
        solved = np.argmax(np.array(k_im), axis=1)

    elif args.what.startswith("kmeans"):
        # Run k-means over all data
        means = KMeans(n_clusters=args.approx_size).fit(v_train)
        v_approx = means.cluster_centers_
        v_approx = normalization(v_input=v_approx, v_means=v_means, v_stds=v_stds)
        solved = (means.labels_).reshape(-1, 1)

    elif args.what.startswith("uniform"):
        # Get uniform data centers
        v_approx, _ = batch_sample(
            batch_size=args.approx_size,
            approx_size=args.approx_size,
            v_train=v_train,
            approx=True,
        )
        v_approx = normalization(v_input=v_approx, v_means=v_means, v_stds=v_stds)
        sample = {"idx": np.arange(0, v_train.shape[0]) / v_train.shape[0]}
        solved = quantify_frames(sample, args.approx_size)
        solved = solved.reshape(-1, 1)
    return solved, v_approx


def train(
    args,
    gts,  # For logging
    num_steps,
    v_train,
    v_means,
    v_stds,
    opt_state,
    params,
    update_fn,
    loss_fn,
    log_freq,
    video_idx,
    log_loss_mof=True,
):
    """
    The actual model training loop.
    """
    v_train = normalization(v_input=v_train, v_means=v_means, v_stds=v_stds)
    lossfile = open(os.path.join(args.figpath, video_idx + "loss.txt"), "w")

    # Just the initial estimate on some data
    if args.ls < 0:
        step = max(1, v_train.shape[0] // 5000)  # Hack to deal with memory limitations
        v_set = v_train[::step, :]
        args.ls = np.median(cdist(v_set, v_set, metric="euclidean")) ** 2
    print("------------------- Video ls:", args.ls)

    # Training loop
    assign = None
    start_time = time.time()
    for i in range(1, num_steps + 1):
        if args.batch_size <= 0:  # just shuffle all data
            bidx = np.arange(0, v_train.shape[0])
            np.random.shuffle(bidx)
            v_batch = v_train[bidx, :]

        else:
            v_batch, bidx = batch_sample(
                batch_size=args.batch_size,
                approx_size=args.approx_size,
                v_train=v_train,
                approx=False,
            )

        opt_state, params, aux = update_fn(
            step=i,
            opt_state=opt_state,
            params=params,
            vi=v_batch,
            ls=args.ls,
            assign=assign,
        )
        loss, k_im = aux

        if log_loss_mof and i % log_freq == 0:
            loss, k_im = loss_fn(vm=params["vm"], vi=v_train, ls=args.ls)
            solved = np.argmax(np.array(k_im), axis=1)
            bgts = np.copy(gts)
            pmetrics = evaluate(
                n_clusters=args.approx_size,
                in_gt=bgts.reshape(
                    -1,
                ).astype(int),
                in_pred=solved.reshape(
                    -1,
                ).astype(int),
                video_idx=str(video_idx),
                plot_path=args.figpath,
                label_names=args.label_names,
                verbose=True,
                plot=False,
                ignore=args.ignore,
                acc_boundary=(args.dataset.startswith("mnist")),
            )
            lossfile.write(str(i) + " " + str(loss) + " " + str(pmetrics["mof"]))
            lossfile.write(os.linesep)

        assign = jax.nn.one_hot(jnp.argmax(k_im, axis=1), num_classes=args.approx_size)

        if (i - 1) % log_freq == 0:
            print(
                f"----step {i} Loss:",
                loss,
                "k_im: (",
                k_im.mean(),
                "+/-",
                k_im.std(),
                ")",
            )
    print("Training loop time ", time.time() - start_time)
    lossfile.close()

    # Do the inference loop
    k_im_final = infer(args=args, v_train=v_train, params=params, loss_fn=loss_fn)
    return params["vm"], k_im_final


def infer(
    args,
    v_train,
    params,
    loss_fn,
):
    """
    Simply does the inference through the trained model.
    """

    start_time = time.time()
    if args.batch_size <= 0:  # Use the whole video
        _, k_im = loss_fn(vm=params["vm"], vi=v_train, ls=args.ls, assign=None)
    else:
        # Loop over batches and predict: shuffling does not matter here cause we only compute k_im
        k_im_list = []
        for i in range(0, v_train.shape[0] // args.batch_size + 1):
            s = i * args.batch_size
            e = min((i + 1) * args.batch_size, v_train.shape[0])
            inds = np.arange(s, e)

            use_shift = False
            if inds.shape[0] < args.batch_size:
                shift = args.batch_size - inds.shape[0]
                inds = np.arange(s - shift, e)  # move s backwards by shift
                use_shift = True
            if v_train[inds, ...].shape[0] > 0:
                _, k_im_batch = loss_fn(
                    vm=params["vm"], vi=v_train[inds, ...], ls=args.ls, assign=None
                )
                if use_shift:
                    k_im_list.append(np.array(k_im_batch)[shift : len(k_im_batch), :])
                else:
                    k_im_list.append(np.array(k_im_batch))
        k_im = np.concatenate(k_im_list, axis=0)
    print("Prediction loop ", time.time() - start_time)
    return k_im
