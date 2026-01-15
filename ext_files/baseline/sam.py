### Author: Silvia L Pintea
# [NOTE]: This is Not an official implementation.
# Not official implementation. Provided simply as an effort for reproducibility.
# Results do NOT match the paper https://ojs.aaai.org/index.php/AAAI/article/view/28445 (AAAI'24)
# (Note: We do not use it in the paper).


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


def start_run(args, data, labels, alpha=0.3, lmbda=1e-3, plot=True, verbose=False):
    """
    Does the main loop of the ABD algorithm.
    data  - is the data matrix,
    labels  - is the labels matrix, used for plotting only
    alpha - smoothing hyperparameter
    lmbda   - a weight hyperparameter for the temporal impact.
    """

    N = data.shape[0]
    K = args.approx_size

    # Smooth the data within a window L and get neighboring similarity
    data = smooth_data(data, K, alpha)
    s_data = getSimilarity(data, plot, labels)

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
        X, t = [], []

        if verbose:
            print("Partition of: {} clusters".format(K_now))

        # Compute the avg features per cluster
        uni = np.unique(f_c)
        for ki in uni.tolist():
            ind = np.where(f_c == ki)[0]
            X.append(np.mean(data[ind, :], axis=0))
            t.append(np.mean(ind / N))

        # Compute spatial similarity over clusters
        G = getSimilarityGraph(np.array(X), np.array(t), lmbda)

        # Merge the 2 most similar clusters
        f_c, K_now = merge(G, f_c, uni)

    return f_c


def getSimilarityGraph(X, t, lmbda):
    """
    Computes the cosine similarity among all clusters.
    X       - average cluster features.
    t       - the normalized time indexes
    lmbda   - a weight hyperparameter for the temporal impact.
    """

    x_sim = metrics.pairwise.cosine_similarity(X, X)
    np.fill_diagonal(x_sim, 0)

    assert X.shape[0] == t.shape[0]
    N = X.shape[0]
    x_sim = metrics.pairwise.cosine_similarity(X, X)
    x_sim = np.arccos(np.clip(x_sim, 0.0, 1.0)) * 180.0 / math.pi
    t_sim = metrics.pairwise.pairwise_distances(
        t.reshape(N, 1), t.reshape(N, 1), metric="l1"
    )

    sig = np.std(x_sim)
    g = np.exp(-lmbda * t_sim) * np.exp(-x_sim / sig)
    np.fill_diagonal(g, 0)
    return g


def getSimilarity(X, plot, labels):
    """
    Compute cosine similarity with neighboring frames.
    X       - original video features.
    plot    - True/False.
    labels  - Segmentation GT for plotting.
    """

    N = X.shape[0]
    s_data = []
    for i in range(1, N):
        simi = metrics.pairwise.cosine_similarity(X[i - 1 : i, :], X[i : i + 1, :])
        simi = np.arccos(np.clip(simi, 0.0, 1.0)) * 180.0 / math.pi
        s_data.append(simi)
    s_data.append(s_data[-1])
    s_data = np.array(s_data)

    # Exp and norm
    sig = np.std(s_data)
    s_data = np.exp(-(s_data / sig))

    if plot:
        cmap = plt.cm.get_cmap("tab20")
        color = np.array(cmap(labels / (max(labels) + 1)))
        plt.scatter(np.arange(0, s_data.shape[0]), s_data, c=color, s=1)
        plt.xlabel("time (s)")
        plt.ylabel("sim(x)")
        plt.title("Video frame distances")
        plt.grid(True)
        plt.savefig("results/simi.png")
        plt.close()
    return s_data
