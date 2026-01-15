### Author: Silvia L. Pintea
# Calls the different baselines.
# ABD and SAM are not official implementations and we could not replicate the author's results.
# (We report in the article the author's results.)

import random
import os
import math
import pickle
import numpy as np
import argparse
import time

from utils.eval_script import evaluate
from utils.auxiliary import get_dataset, check_make_dir


"""# Define all parameters"""
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--datapath",
    default="",
    type=str,
    help="Path to the dataset",
)
parser.add_argument(
    "--input_channels",
    type=int,
    default=-1,
    help="The input channels is defined in the data reading.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=-1,
    help="The batch size is defined wrt the video length.",
)
parser.add_argument(
    "--set",
    default="test",
    type=str,
    choices=["test", "val", "train"],
    help="Dataset split to use (if defined). All video data is used by default.",
)
parser.add_argument(
    "--per_vid",
    default=False,
    action="store_true",
    help="If the number of segments is per vid or per activity",
)
parser.add_argument(
    "--rand_seg",
    default=False,
    action="store_true",
    help="If we randomly change the number of segments is per vid",
)
parser.add_argument(
    "--dataset",
    default="fs",
    type=str,
    choices=["bf", "mnist", "fs", "yti", "da", "mp2", "he"],
    help="Dataset name to use",
)
parser.add_argument(
    "--run",
    default="twfinch",
    type=str,
    choices=["twfinch", "abd", "sam"],
    help="Model to run. Since ABD/SAM are not official implementation we do not report on them",
)

args = parser.parse_args()
# Dataset setting
args.mapping = "mid"
args.label_names = []
args.ignore = ""


# SAM and ABD method are nearly identical according to their paper, but no code is provided.
# The reproductions here are not to be used to compare with these papers.
if args.run.startswith("twfinch"):
    from twfinch import start_run
elif args.run.startswith("abd"):
    from abd import start_run
elif args.run.startswith("sam"):
    from sam import start_run

args.figpath = os.path.join(
    "results", ("RND" if args.rand_seg else "") + "_" + args.dataset + "_" + args.run
)
check_make_dir(args.figpath)
random.seed(args.seed)
np.random.seed(args.seed)


def run(args):
    """
    Calls a specific algorithm per video and write the predictions.
    """

    start = time.time()
    print("starting the processing...")
    is_data = True
    args.vidx = 0
    while is_data:
        (args, is_data, data, gt) = get_dataset(args=args, vidx=args.vidx, smooth=False)
        print(args)

        if not is_data:
            break

        args.pickle_file = os.path.join(
            args.figpath, "video" + str(args.vidx) + ".pickle"
        )
        isthere = os.path.exists(args.pickle_file)
        if isthere:
            args.vidx += 1
            continue

        print(
            "[",
            args.vidx,
            "] New approx-frame size is... ",
            args.approx_size,
        )

        # Run the segmentation
        solved = start_run(args=args, data=data, labels=gt)

        # Do the Hungarian assignment and the visualization
        evaluate(
            n_clusters=args.approx_size,
            in_gt=gt.reshape(-1).astype(int),
            in_pred=solved.reshape(-1).astype(int),
            video_idx=str(args.vidx),
            plot_path=args.figpath,
            label_names=args.label_names,
            verbose=True,
            plot=True,
            ignore=args.ignore,
            acc_boundary=(args.dataset.startswith("mnist")),
        )
        args.vidx += 1
        end = time.time()
        print("Time elapsed per video:", end - start)


if __name__ == "__main__":
    print(args)
    run(args)
