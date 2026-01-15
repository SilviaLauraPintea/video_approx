### Author Silvia. L. Pintea
# Helper functions for data reading, smoothing, visualizations, and score aggregation.

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import sys
import argparse
import os
import math
import numpy as np
import random
import jax
import scipy
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.switch_backend("agg")


def set_seed(seed):
    """
    Set random seed for np and jax.
    """
    random.seed(seed)
    np.random.seed(seed)


def check_make_dir(directory: str):
    """
    Check if a dir exists, otherwise it makes it.
    """
    dirpath = os.path.normpath(directory)
    pathsplits = dirpath.split(os.sep)

    recpath = os.path.normpath("")
    for split in pathsplits:
        recpath = os.path.join(recpath, split)
        if not os.path.exists(recpath):
            os.makedirs(recpath)


def gauss_smooth_data(data, M, alpha, gt):
    """
    Smooth data over the time dimension.
    """
    N = data.shape[0]
    L = math.ceil(alpha * N / M)
    sigma = (L - 1.0) / 6.0

    # Kernel size is: (2 radius + 1) = (2(truncate * sigma) + 1) = L
    smooth = scipy.ndimage.gaussian_filter1d(data, sigma=sigma, truncate=3, axis=0)
    return smooth


def get_dataset(args, vidx, smooth):
    """
    Loads a specific video feature and its ground truth.
    All paths and sizes are hard-coded here [Sorry].
    """
    try:
        if args.dataset.startswith("mnist"):
            from datasets.mnist_video import MnistVideo

            args.datapath = "./data/Moving5/mnist100.pkl"
            args.input_channels = 28 * 28 * 3
            args.imsize = 28
            data = MnistVideo(args, set_name=args.set)
            video = data[vidx]["samples"]
            gt = data[vidx]["labels"]
            args.video_name = data[vidx]["name"][:-4]
            args.label_names = data[vidx]["label_names"]
            args.avg_cls = data[vidx]["avg_cls"]

        elif args.dataset.startswith("bf"):
            from datasets.bf_video import BFVideo

            args.datapath = "./data/Breakfast/"
            args.split = "all"
            args.input_channels = 64
            data = BFVideo(args, set_name=args.set)
            video = data[vidx]["samples"]
            gt = data[vidx]["labels"]
            args.video_name = data[vidx]["name"][:-4]
            args.label_names = data[vidx]["label_names"]
            args.avg_cls = data[vidx]["avg_cls"]  # per activity

        elif args.dataset.startswith("fs"):
            from datasets.fs_video import FSVideo

            args.datapath = "./data/50Salads/"
            args.mapping = "mid"
            args.input_channels = 64
            data = FSVideo(args, set_name=args.set)
            video = data[vidx]["samples"]
            gt = data[vidx]["labels"]
            args.video_name = data[vidx]["name"][:-4]
            args.label_names = data[vidx]["label_names"]
            args.avg_cls = data[vidx]["avg_cls"]

        elif args.dataset.startswith("yti"):
            from datasets.yti_video import YTIVideo

            args.ignore = [0]  # equivalent of "-1"
            args.datapath = "./data/YouTubeInstructions/"
            args.input_channels = 3000
            data = YTIVideo(args, set_name=args.set)
            video = data[vidx]["samples"]
            gt = data[vidx]["labels"]
            args.video_name = data[vidx]["name"][:-4]
            args.label_names = data[vidx]["label_names"]
            args.avg_cls = data[vidx]["avg_cls"]  # per activity

        elif args.dataset.startswith("da"):
            from datasets.da_video import DAVideo

            args.datapath = "./data/DesktopAssembly/"
            args.input_channels = 512
            data = DAVideo(args, set_name=args.set)
            video = data[vidx]["samples"]
            gt = data[vidx]["labels"]
            args.video_name = data[vidx]["name"][:-4]
            args.label_names = data[vidx]["label_names"]
            args.avg_cls = data[vidx]["avg_cls"]

        elif args.dataset.startswith("he"):
            from datasets.he_video import HEVideo

            args.datapath = "./data/HollywoodExtended/"
            args.input_channels = 64
            data = HEVideo(args, set_name=args.set)
            video = data[vidx]["samples"]
            gt = data[vidx]["labels"]
            args.video_name = data[vidx]["name"][:-4]
            args.label_names = data[vidx]["label_names"]
            args.avg_cls = data[vidx]["avg_cls"]

        elif args.dataset.startswith("mp2"):
            from datasets.mp2_video import MP2Video

            args.datapath = "./data/MPIICooking/"
            args.input_channels = 64
            data = MP2Video(args, set_name=args.set)
            video = data[vidx]["samples"]
            gt = data[vidx]["labels"]
            args.video_name = data[vidx]["name"][:-4]
            args.label_names = data[vidx]["label_names"]
            args.avg_cls = data[vidx]["avg_cls"]
    except:
        return args, False, None, None

    # Define number of approx-frames per vid or per activity/dataset
    if args.per_vid:
        segments = np.unique(gt)
        args.approx_size = len(segments)
    elif args.rand_seg:
        args.approx_size = max(
            2, args.avg_cls + random.randint(-args.avg_cls, args.avg_cls)
        )
    else:
        args.approx_size = args.avg_cls

    print(
        "Starting on ",
        args.dataset,
        " of ..............",
        len(data),
        " with M=",
        args.approx_size,
        " and data shape ",
        video.shape,
    )
    if smooth is True:
        video = gauss_smooth_data(
            data=video, M=args.approx_size, alpha=args.smooth, gt=gt
        )
    return args, True, video, gt


def quantify_frames(sample, n_segments):
    """
    Quantifies frames into a number of segments.
    """
    inds = (sample["idx"]).reshape(-1, 1)
    ranges = np.linspace(0.0, 1.0, n_segments + 1)

    # Define segment id per sample
    segment_ids = np.zeros(inds.shape)
    for k in range(n_segments):
        wheres = np.where((inds >= ranges[k]) & (inds < ranges[k + 1]))
        segment_ids[wheres] = k
    wheres = np.where((inds >= ranges[n_segments]))
    segment_ids[wheres] = n_segments - 1
    segment_ids = segment_ids.astype(np.int32)
    return segment_ids.reshape(-1)


def visualize_approx(
    args: Any,
    v_approx: np.array,
    v_train: np.array,
    v_stds: np.array,
    v_means: np.array,
    gts: np.array,
    vidx: int,
):
    """
    For MNIST script to visualize the approx-frames.
    """
    # Un-normalize the approx-frames
    approx = np.copy(v_approx)
    approx = approx * v_stds.repeat(approx.shape[0], axis=0) + v_means.repeat(
        approx.shape[0], axis=0
    )

    # Compute the avg per segment
    n_segments = np.unique(gts).size
    avg_imgs = []
    for i in range(n_segments):
        one_segments = np.where(gts == i)[0]
        avg_img = v_train[one_segments[::2], ...]
        avg_img = avg_img.mean(axis=0).reshape(args.imsize, args.imsize, 3)
        avg_imgs.append(avg_img)
        avg_img = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())
    avgimg = np.concatenate(np.expand_dims(avg_imgs, axis=0), axis=0)

    # Compute the assignment between the true segment average and learned approx-frames
    avgimg = avgimg.reshape(avgimg.shape[0], -1)
    approx = approx.reshape(approx.shape[0], -1)
    dist_mat = np.sum((avgimg[:, None] - approx[None, :]) ** 2, -1)
    approx_id, _ = scipy.optimize.linear_sum_assignment(dist_mat)

    # Invert the assignment
    inv_approx_id = np.empty_like(approx_id)
    inv_approx_id[approx_id] = np.arange(0, approx_id.shape[0])

    # Plot the true frame average and the approx frames
    fig, axs = plt.subplots(2, approx.shape[0])
    axs[0, 2].set_title("Ground truth average frames")
    axs[1, 2].set_title("approx-frames")
    for i in range(avgimg.shape[0]):
        one_avg = avgimg[inv_approx_id[i], :]
        one_approx = approx[inv_approx_id[i], :]
        one_avg = one_avg.reshape(args.imsize, args.imsize, -1)
        one_approx = one_approx.reshape(args.imsize, args.imsize, -1)
        axs[0, i].imshow(one_avg + 0.2)
        axs[0, i].set_axis_off()
        axs[1, i].imshow(one_approx + 0.2)
        axs[1, i].set_axis_off()
    fig.tight_layout()
    plt.savefig(os.path.join(args.figpath, "gt_approx" + str(vidx) + ".pdf"))
    plt.close()


def plot_curve(figpath: str, loss: bool):
    """
    Makes a simple loss plot.
    """
    files = os.listdir(figpath)
    dim = 1 if loss else 2
    # Collect all pickles
    avgloss = None
    idx = 0
    for f in files:
        curr_path = os.path.join(figpath, f)

        if os.path.isfile(curr_path) and curr_path.endswith("loss.txt"):
            idx += 1
            lossfile = open(curr_path, "r")
            losses = np.loadtxt(lossfile, delimiter=" ")

            if avgloss is None:
                avgloss = losses
            else:
                avgloss += losses

            # do the plotting bit
            fig = plt.figure()
            fig.suptitle("training MMD^2 losses")

            plt.semilogy(
                losses[:, 0],
                losses[:, dim],
                "--",
                color=("green" if loss else "red"),
                label="train loss",
            )
            plt.ylabel("train losses")
            plt.xlabel("train iteration")
            plt.savefig(curr_path[0:-4] + ".pdf")
            plt.close()

    # plot the avg
    avgloss /= idx
    fig = plt.figure()
    fig.suptitle("Avg training MMD^2 losses")

    plt.semilogy(
        avgloss[:, 0],
        avgloss[:, dim],
        "--",
        color=("green" if loss else "red"),
        label="train loss",
    )
    plt.ylabel("train " + ("loss" if loss else "mof"))
    plt.xlabel("train iteration")
    plt.savefig(os.path.join(figpath, ("avg_losses.pdf" if loss else "avg_mof.pdf")))
    plt.close()


def gather_scores(figpath: str):
    """
    Read pickle scores and average them across all videos.
    """
    files = os.listdir(figpath)

    # Collect all pickles
    file_list = []
    for f in files:
        curr_path = os.path.join(figpath, f)
        if os.path.isfile(curr_path) and not curr_path.endswith("all_metrics.pickle"):
            file_list.append(curr_path)
    file_list.sort()

    # Read all the metrics
    metrics = {}
    metrics["video"] = []
    metrics["mof"] = []
    metrics["f1"] = []
    metrics["iou"] = []
    metrics["acc_bd"] = []
    for i, f in enumerate(file_list):
        if "pickle" in f:
            d = pickle.load(open(f, "rb"))
            metrics["video"] = f
            if isinstance(d["mof"], list):
                metrics["mof"] += d["mof"]
                metrics["f1"] += d["f1"]
                metrics["iou"] += d["iou"]
                try:
                    metrics["acc_bd"] += d["acc_bd"]
                except:
                    metrics["acc_bd"] += 0
            else:
                metrics["mof"].append(d["mof"])
                metrics["f1"].append(d["f1"])
                metrics["iou"].append(d["iou"])
                try:
                    metrics["acc_bd"].append(d["acc_bd"])
                except:
                    metrics["acc_bd"].append(0)
    print(
        figpath,
        " [",
        len(metrics["mof"]),
        "] Avg MOF:",
        f'{np.array(metrics["mof"]).mean()*100:.2f}%',
        "\tavg IOU:",
        f'{np.array(metrics["iou"]).mean()*100:.2f}%',
        "\tavg F1:",
        f'{np.array(metrics["f1"]).mean()*100:.2f}%',
        "\tavg Acc boundary:",
        f'{np.array(metrics["acc_bd"]).mean()*100:.2f}%',
    )
    with open(os.path.join(figpath, "all_metrics.pickle"), "wb") as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------
# --- Averaging per-frame scores --------------------------------
# ---------------------------------------------------------------

if __name__ == "__main__":
    gather_scores(sys.argv[1])
