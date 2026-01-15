### Author: Silvia L. Pintea
# This is the main call: It calls the segmentation code per video for different datasets, and stores the segmentation as a pickle file.
# Edit the different configs as needed.

import os
import time
import numpy as np


from utils.auxiliary import visualize_approx, get_dataset, check_make_dir, set_seed
from utils.eval_script import evaluate
from mmd.train import start_run
from mmd.kernel import init_kernel
from mmd.preparation import get_normalization, init_model


"""# Define all parameters"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--learning_rate",
    default=5e-2,
    type=float,
    help="Learning rate, for ntk is always 5e-2",
)
parser.add_argument(
    "--dataset",
    default="",
    type=str,
    choices=["bf", "mnist", "fs", "yti", "da", "mp2", "he"],
    help="Dataset name to use",
)
parser.add_argument(
    "--kernel",
    default="combi",
    type=str,
    choices=["spl", "ntk", "combi", "combi-sphere", "sphere", "sntk", "nngp"],
    help="Kernel to use",
)
parser.add_argument("--figpath", default="./results/", type=str, help="output plots")
parser.add_argument(
    "--iter",
    default=int(1e2),
    type=int,
    help="Video id to be processed",
)
parser.add_argument(
    "--batch_size",
    default=-1,
    type=int,
    help="Batch size",
)
parser.add_argument(
    "--smooth",
    default=2,
    type=float,
    help="Feature smoothing",
)
parser.add_argument(
    "--set",
    default="test",
    type=str,
    choices=["test", "val"],
    help="The set to use",
)
parser.add_argument(
    "--what",
    default="mmd",
    type=str,
    choices=["mmd", "uniform", "kmeans"],
    help="The method to run",
)
parser.add_argument(
    "--input_channels",
    default=-1,
    type=int,
)
parser.add_argument(
    "--datapath",
    default="",
    type=str,
    help="The name comes from the datasets loading",
)
parser.add_argument(
    "--depth",
    default=1,
    type=int,
    help="The MLP depth before NTK",
)
parser.add_argument(
    "--activation",
    default="relu",
    type=str,
    choices=["relu", "tanh", "cos"],
    help="The activation function in the MLP",
)
parser.add_argument(
    "--wd",
    default=1e-3,
    type=float,
    help="Weight decay is fixed",
)
parser.add_argument(
    "--ls",
    default=-1,
    type=float,
    help="Length-scale parameter in the Gaussian kernel: (<0) median, or 1e+2, 1e+1, etc.",
)
parser.add_argument("--plot", default=False, action="store_true")
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

args = parser.parse_args()
args.ignore = []


def run(args):
    """
    Runs the complete pipeline.
    Check the args for the different running settings.
    """
    set_seed(args.seed)
    check_make_dir(args.figpath)

    vidx = 0
    is_data = True
    while is_data:
        # Load all video data --------
        (args, is_data, v_train, gts) = get_dataset(args=args, vidx=vidx, smooth=True)
        if not is_data:
            break

        args.pickle_file = os.path.join(args.figpath, "video" + str(vidx) + ".pickle")
        isthere = os.path.exists(args.pickle_file)

        args.batch_size = min(512, int(v_train.shape[0] // (args.approx_size)))
        print("Batch size:", args.batch_size)

        if isthere:
            vidx += 1
            continue

        # Just gets the normalization without normalizing
        v_means, v_stds = get_normalization(v_train=v_train)

        # Define Kernel -----
        if args.kernel.startswith("combi"):
            a_kernel = init_kernel(args=args, kern="ntk")
            if args.kernel.endswith("shpere"):
                kernel_fn = {"spl": None, "sphere": a_kernel}  # spl first
            else:
                kernel_fn = {"spl": None, "ntk": a_kernel}  # spl first
        elif args.kernel.startswith("spl"):
            kernel_fn = {"spl": None}
        elif (
            args.kernel.startswith("ntk")
            or args.kernel.startswith("sphere")
            or args.kernel.startswith("sntk")
        ):
            a_kernel = init_kernel(args=args, kern="ntk")
            kernel_fn = {args.kernel: a_kernel}
        elif args.kernel.startswith("nngp"):
            a_kernel = init_kernel(args=args, kern="nngp")
            kernel_fn = {"nngp": a_kernel}

        opt_state, update_fn, loss_fn, params = init_model(
            args=args,
            v_train=v_train,
            v_means=v_means,
            v_stds=v_stds,
            kernel_fn=kernel_fn,
        )

        # Do the labeling --------
        start_time = time.time()
        solved, v_approx = start_run(
            args=args,
            gts=gts,
            v_train=v_train,
            v_means=v_means,
            v_stds=v_stds,
            opt_state=opt_state,
            params=params,
            update_fn=update_fn,
            loss_fn=loss_fn,
            video_idx=str(vidx),
        )
        print("Training time elapsed: ", time.time() - start_time)

        # Hungarian assignment ---------
        start_time = time.time()
        evaluate(
            n_clusters=args.approx_size,
            in_gt=gts.reshape(
                -1,
            ).astype(int),
            in_pred=solved.reshape(
                -1,
            ).astype(int),
            video_idx=str(vidx),
            plot_path=args.figpath,
            label_names=args.label_names,
            verbose=True,
            plot=args.plot,
            ignore=args.ignore,
            acc_boundary=(args.dataset.startswith("mnist")),
        )
        print("Eval time elapsed: ", time.time() - start_time)

        # MNIST visualization stuff ----------
        if args.dataset.startswith("mnist") and not args.rand_seg:
            visualize_approx(
                args=args,
                v_approx=v_approx,
                v_train=v_train,
                v_stds=v_stds,
                v_means=v_means,
                gts=gts,
                vidx=vidx,
            )
        vidx += 1


if __name__ == "__main__":
    print(args)
run(args)
