### Author: Silvia L Pintea
# [NOTE]: This is Not an official implementation.
# It is provided simply as an effort for reproducibility.
# Results do NOT match the paper https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Fast_and_Unsupervised_Action_Boundary_Detection_for_Action_Segmentation_CVPR_2022_paper.pdf (CVPR'22)
# (Note: We do not use it in the paper).


import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics


def smooth_data(x, K, alpha):
    N = x.shape[0]
    L = math.ceil(alpha * N / K)

    kernel = np.ones((L,)) / L
    smooth = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, "same"), axis=0, arr=x
    )
    return smooth


def NMS(x, L, ismin, extremes=True):
    """
    Pick the max/min function location per window.
    x - 1D function.
    L - the window size.
    """
    boundary = []
    for i in range(L // 2, x.shape[0] - L // 2, L):
        window = x[i - L // 2 : i + L // 2]
        if ismin:
            pos = np.argmin(window)
        else:
            pos = np.argmax(window)
        boundary.append(pos + i - L // 2)

    window = x[-L:]
    if ismin:
        pos = np.argmin(window)
    else:
        pos = np.argmax(window)
    boundary.append(pos + x.shape[0] - L - 1)

    # add the extremes
    if extremes:
        if boundary[0] != 0:
            boundary = [0] + boundary
        if boundary[-1] != x.shape[0]:
            boundary = boundary + [x.shape[0]]

    # Get unique values and sort
    boundary = np.unique(boundary)
    boundary.sort()
    return boundary.tolist()


def merge(G, f_c, uni):
    """
    Merge the 2 most similar clusters given the similarity matrix G.
    G   - cluster similarity matrix
    f_c - current frame to cluster assignment
    uni - unique cluster numbers in <f_c>
    """
    val = np.max(G)
    (x, y) = np.where(G == val)

    if x.shape[0] > 1:
        ind = x.tolist()
    else:
        ind = [x, y]

    # Get unique clusters:
    cval1 = uni[ind[1]]
    cval2 = uni[ind[0]]

    # Merge the 2 most similar clusters
    f_c[f_c == cval1] = cval2
    return f_c, np.unique(f_c).size


def start_run(args, data, labels, alpha=0.3, plot=True, verbose=False):
    """
    Does the main loop of the ABD algorithm.
    data  - is the data matrix,
    labels  - is the labels matrix, used for plotting only
    alpha - smoothing hyperparameter
    """
    N = data.shape[0]
    K = args.approx_size

    # Smooth the data within a window L and get neighboring similarity
    data = smooth_data(data, K, alpha)
    s_data = getSimilarity(data, plot, labels, args.figpath, args.vidx)

    # Get boundaries in (alpha L/N) windows ==========================
    L = math.ceil(alpha * N / K)
    B = NMS(s_data, L, ismin=True)

    # Define the initial cluster assignment
    K_now = len(B) - 1
    f_c = np.zeros((N,), dtype=np.int32)
    for k in range(0, K_now):
        f_c[B[k] : B[k + 1]] = k

    # Keep merging until desired number of clusters =================
    while K_now > K:
        X = []

        if verbose:
            print("Partition of: {} clusters".format(K_now))

        # Compute the avg features per cluster
        uni = np.unique(f_c)
        for ki in uni.tolist():
            mask = f_c == ki
            X.append(np.mean(data[mask, :], axis=0))

        # Compute spatial similarity over clusters
        newX = np.array(X)
        G = getSimilarityGraph(np.array(newX))

        # Merge the 2 most similar clusters
        f_c, K_now = merge(G, f_c, uni)
    return f_c


def getSimilarityGraph(X):
    """
    Computes the cosine similarity among all clusters.
    X - average cluster features.
    """
    x_sim = metrics.pairwise.cosine_similarity(X, X)
    np.fill_diagonal(x_sim, 0)
    return x_sim


def getSimilarity(X, plot, labels, figpath, idx):
    """
    Compute cosine similarity with neighboring frames.
    X       - original video features.
    plot    - True/False.
    labels  - Segmentation GT for plotting.
    """
    N = X.shape[0]
    x_sim = []
    for i in range(1, N):
        simi = metrics.pairwise.cosine_similarity(X[i - 1 : i, :], X[i : i + 1, :])
        x_sim.append(simi)
    x_sim.append(x_sim[-1])
    x_sim = np.array(x_sim).reshape(X.shape[0], -1)

    if plot:
        cmap = plt.cm.get_cmap("tab20")
        color = np.array(cmap(labels / (max(labels) + 1)))
        plt.scatter(np.arange(0, x_sim.shape[0]), x_sim, c=color, s=1)
        plt.xlabel("time (s)")
        plt.ylabel("sim(x)")
        plt.title("Video frame similarities")
        plt.grid(True)
        plt.savefig(os.path.join(figpath, "simi" + str(idx) + ".png"))
        plt.close()
    return x_sim
