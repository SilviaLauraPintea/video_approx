################################################################################
# The code builds on the code of:
# Sarfraz, Saquib, Vivek Sharma, and Rainer Stiefelhagen. "Efficient parameter-free clustering using first neighbor
# relations." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
# TW-Finch: https://github.com/ssarfraz/FINCH-Clustering/
# All credit for third party sofware/code is with the authors.
################################################################################
### Code modified and extended by Silvia L. Pintea

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import time
import os
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
import seaborn as sns

plt.switch_backend("agg")

from utils.auxiliary import check_make_dir


def save_segmentation_plot(
    eval_labels: List[int],
    gt_labels: List[int],
    figure_path: str,
    features_types: List[str] = [""],
    title: str = "",
    label_names: List[str] = [],
    mnist: bool = False,
):
    """
    Plotting the segmentation per video for the ground truth and the predictions.
    """
    fig, axs = plt.subplots(2, figsize=(20, 3))
    cols = max(np.max(gt_labels) + 1, np.max(eval_labels) + 1)

    if mnist:
        color = (
            np.array(
                [
                    [200, 200, 0],
                    [228, 118, 66],
                    [91, 74, 206],
                    [0, 200, 200],
                    [117, 212, 134],
                    # Extra
                    [255, 255, 255],
                    [255, 125, 125],
                    [125, 125, 255],
                    [125, 255, 125],
                    [255, 255, 125],
                    [255, 125, 255],
                    [125, 225, 255],
                ]
            )
            / 255.0
        )
    else:
        color = sns.color_palette("colorblind", n_colors=cols)

    seen_labels = []

    def draw_patch(label, i, gt):
        if label == -1:
            rect = patches.Rectangle(
                (i, 0), 1, 1, fill=True, color="whitesmoke", alpha=0.9
            )
        else:
            rect = patches.Rectangle(
                (i, 0),
                1,
                1,
                fill=True,
                color=color[label],
                alpha=1,
                label="_nolegend_"
                if label in seen_labels
                else label_names[len(seen_labels) - 1],
            )
            if label not in seen_labels and gt:
                seen_labels.append(label)
        return rect

    # Plot the Ground Truth --------------
    axs[0].set_xlim([0, len(gt_labels)])
    for patch1 in map(
        draw_patch, gt_labels, range(len(gt_labels)), [True] * len(gt_labels)
    ):
        axs[0].add_patch(patch1)

    if len(label_names) > 0:
        axs[0].legend(label_names)
        box = axs[0].get_position()
        axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot the eval labels ---------------
    axs[1].set_xlim([0, len(eval_labels)])
    for patch2 in map(
        draw_patch,
        eval_labels,
        range(len(eval_labels)),
        [False] * len(eval_labels),
    ):
        axs[1].add_patch(patch2)

    if len(label_names) > 0:
        box = axs[1].get_position()
        axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    labels = ["GT"] + features_types

    # Set the axis properties and do the plotting.
    for i, ax in enumerate(axs):
        ax.set_ylim([0, 1])
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(False)
        ax.text(
            -1,
            ax.get_ylim()[1] / 2,
            labels[i],
            ha="right",
            va="center",
            fontsize=16,
            weight="bold",
        )
    fig.suptitle(title.replace("\t", " "))
    plt.savefig(figure_path, bbox_extra_artists=(lgd,))
    plt.clf()
    plt.cla()
    plt.close(fig)


def orderlabel(group: np.array):
    """
    Little hack to reorder the segment labeling when some segment-ids are missing in the predictions.
    """
    unig = np.unique(group)

    for i, k in enumerate(unig.tolist()):
        indi = np.where(group == k)[0]
        group[indi.astype(int)] = i + 1e5  # hack to not mistakenly merge classes
    group = group - 1e5
    return group.astype(int)


def estimate_cost_matrix(gt_labels: List[int], cluster_labels: List[int]):
    """
    Estimates the alignment cost between ground truth and predictions.
    NOTE: This assumes the predictions are continuous numbers (i.e. no missing segment-ids).
    """
    # Make sure the lengths of the inputs match:
    if len(gt_labels) != len(cluster_labels):
        print("The dimensions of the gt_labls and the pred_labels do not match")
        return -1

    L_gt = np.unique(np.array(gt_labels))
    L_pred = np.unique(np.array(cluster_labels))

    nClass_pred = L_pred.shape[0]
    dim_1 = max(nClass_pred, np.max(L_gt) + 1)
    profit_mat = np.zeros((nClass_pred, dim_1))

    for i in L_pred:
        idx = np.where(cluster_labels == i)[0]
        gt_selected = np.array(gt_labels)[np.array(idx).astype(int)]
        for j in L_gt:
            profit_mat[i][j] = np.count_nonzero(gt_selected.astype(int) == j)
    return -profit_mat


def boundary_accuracy(pred: np.array, gt_labels: np.array):
    """
    Computes the boundary accuracy: i.e. a boundary = whenever there is segment id switch.
    """
    dgt = gt_labels[1:] - gt_labels[:-1]
    dpred = pred[1:] - pred[:-1]
    dgt = (dgt != 0).astype(int)
    dpred = (dpred != 0).astype(int)

    boundaries = dgt != 0
    acc = metrics.accuracy_score(dgt[boundaries], dpred[boundaries])
    return acc


def evaluate(
    n_clusters: int,
    in_gt: np.array,
    in_pred: np.array,
    video_idx: int,
    plot_path: str,
    label_names: List[str],
    verbose: bool = True,
    plot: bool = True,
    ignore: List[int] = [],
    acc_boundary: bool = False,
):
    """
    Computes the evaluation scores and plots the ground truth segmentation versus the predictions.
    """
    check_make_dir(plot_path)
    if len(ignore) > 0:
        mask = None
        for igno in ignore:
            if mask is None:
                mask = in_gt != igno
            else:
                mask = mask & (in_gt != igno)

        gt_labels = in_gt[mask]
        pred = in_pred[mask]
    else:
        gt_labels = in_gt
        pred = in_pred

    # We need to resort the predictions because we may have missing classes
    pred = orderlabel(pred)

    start_time = time.time()
    # Find best assignment through Hungarian Method
    cost_matrix = estimate_cost_matrix(
        gt_labels=gt_labels.tolist(), cluster_labels=pred.tolist()
    )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # decode the predicted labels
    y_pred = col_ind[pred]

    # Calculate the metrics (External libraries)
    cur_acc = metrics.accuracy_score(gt_labels, y_pred)
    f1_macro = metrics.f1_score(gt_labels, y_pred, average="macro")  # F1-Score
    iou = np.sum(metrics.jaccard_score(gt_labels, y_pred, average=None)) / n_clusters

    stats = f"Evaluation on {video_idx}: accuracy = {cur_acc} IoU = {iou} and f1 ={f1_macro}"
    if verbose:
        print(stats)
    print("Hungarian matching time ", time.time() - start_time)

    # Write the pickles
    pickle_file = os.path.join(plot_path, "video" + str(video_idx) + ".pickle")
    pmetrics = {}
    pmetrics["mof"] = cur_acc
    pmetrics["f1"] = f1_macro
    pmetrics["iou"] = iou
    pmetrics["nmi"] = 0
    pmetrics["ari"] = 0
    pmetrics["pred"] = y_pred
    pmetrics["gt"] = gt_labels
    if acc_boundary:  # Only compute boundary accuracy for MNIST
        pmetrics["acc_bd"] = boundary_accuracy(
            pred=np.array(y_pred), gt_labels=np.array(gt_labels)
        )

    with open(pickle_file, "wb") as handle:
        pickle.dump(pmetrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do the plotting
    if plot:
        start_time = time.time()
        fig_path = os.path.join(plot_path, "seg_" + str(video_idx) + ".pdf")
        save_segmentation_plot(
            eval_labels=y_pred.tolist(),
            gt_labels=gt_labels.tolist(),
            figure_path=fig_path,
            features_types=["Ours"],
            title=stats,
            label_names=label_names,
            mnist=acc_boundary,  # use different colors for MNIST
        )
        print("Plotting time ", time.time() - start_time)
    return pmetrics
