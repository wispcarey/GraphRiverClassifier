# batch_active_learning.py
"""
Authors: James, Bohan, and Zheng

Functions for doing batch active learning and coreset selection
"""

import timeit
import os

# python 3.8 (used in google colab) needs typing.List, typing.Dict and typing.Tuple
from typing import Optional, List, Dict, Tuple
from collections.abc import Iterable

import graphlearning.active_learning as al
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import time
import os, glob

# SKLearn imports
from sklearn.utils import check_random_state

# Scipy imports
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sparse

import ActiveLearning.bal_utils as bal_utils
from ActiveLearning.AcqFunctions import model_change_supervised, uc_supervised

################################################################################
## Default Parameters

DENSITY_RADIUS: float = 0.2
BATCH_SIZE: int = 15

ACQUISITION_FUNCTIONS: List[str] = ["uc", "vopt", "mc", "mcvopt"]
AL_METHODS: List[str] = ["local_max", "random", "topn_max", "acq_sample", "global_max"]
AL_METHOD_NAMES = ["LocalMax", "Random", "TopMax", "Acq_sample", "Sequential"]

MAX_NEW_SAMPLES_PROPORTIONS: Dict[str, float] = {
    "mstar": 0.15,
    "open_sar_ship": 0.35,
    "fusar": 0.68,
}

MAX_NEW_SAMPLES_DICT: Dict[str, int] = {
    name: int(bal_utils.SAR_DATASET_SIZE_DICT[name] * MAX_NEW_SAMPLES_PROPORTIONS[name])
    for name in bal_utils.AVAILABLE_SAR_DATASETS
}

BALOutputType = Tuple[np.ndarray, List[int], np.ndarray, float]


################################################################################
### coreset functions
def get_poisson_weighting(G, train_ind, tau=1e-8, normalization='combinatorial'):
    n = G.num_nodes
    F = np.zeros(n)
    F[train_ind] = 1
    if normalization == 'combinatorial':
        F -= np.mean(F)
    else:
        F -= np.mean(G.degree_matrix(p=0.5) * F) * G.degree_vector() ** (0.5)

    L = G.laplacian(normalization=normalization)
    if tau > 0.0:
        L += tau * sparse.eye(L.shape[0])

    w = gl.utils.conjgrad(L, F, tol=1e-5)
    w -= np.min(w, axis=0)

    return w


def plusplus_coreset(graph, num_points=10, random_state=None, method='dijkstra', eik_p=1.0, tau=0.1,
                     ofs=0.2, q=1.0, knn_dist=None, kernel='gaussian', plot=False, X=None, initial=None):
    n = graph.num_nodes
    # if want to use 0/1 edge weights, specify kernel = 'uniform'
    if kernel == 'uniform':
        G = gl.graph(graph.adjacency())
    else:
        G = graph

    all_inds = np.arange(G.num_nodes)
    if random_state is None:
        random_state = 0
    random_state = check_random_state(random_state)

    if method == 'peikonal':
        # print("Preparing knn density estimator for p-eikonal")
        if knn_dist is not None:
            alpha = 2.
            d = np.max(knn_dist, axis=1)
            kde = (d / d.max()) ** (-1)
            f = kde ** (-alpha)
        else:
            print("No knn dist info provided, defaulting to just f = 1")
            f = 1.0
    # randomly select initial point for coreset
    if initial:
        indices = initial
        if len(initial) > 1:
            if method == 'dijkstra':
                dists = G.dijkstra(indices[:-1])
            elif method == 'peikonal':
                dists = G.peikonal(indices[:-1], p=eik_p, f=f)
            elif method == 'poisson':
                dists = np.full(n, np.inf)
                for i in range(len(initial) - 1):
                    w = get_poisson_weighting(G, [indices[i]], tau=tau)
                    dists_new = 1. / (ofs + w)
                    np.minimum(dists, dists_new, out=dists)
        else:
            dists = np.full(n, np.inf)
        j = 0
    else:
        indices = np.array([random_state.randint(n)])
        dists = np.full(n, np.inf)
        j = 1

    # while still have budget to add to the coreset, propagate dijkstra out
    while j < num_points:
        j += 1
        x = indices[-1]
        if method == 'dijkstra':
            dist_to_x = G.dijkstra([x])
        elif method == 'peikonal':
            dist_to_x = G.peikonal([x], p=eik_p, f=f)
        elif method == 'poisson':
            w = get_poisson_weighting(G, [x], tau=tau)
            dist_to_x = 1. / (ofs + w)
        np.minimum(dists, dist_to_x, out=dists)
        if plot and X is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            p1 = ax1.scatter(X[:, 0], X[:, 1], c=dists)
            ax1.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax1.scatter(X[x, 0], X[x, 1], c='pink', marker='^', s=100)
            ax1.set_title("Sampling Probabilities")
            plt.colorbar(p1, ax=ax1)

            p2 = ax2.scatter(X[:, 0], X[:, 1], c=dists ** q)
            ax2.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax2.set_title("Updated Distances")
            plt.colorbar(p2, ax=ax2)
            plt.show()

        # sample next point proportionally to the q^th power of the distances
        if q != 1:
            vals = np.cumsum(dists ** q)
        else:
            vals = np.cumsum(dists)
        next_ind = np.searchsorted(vals, vals[-1] * random_state.uniform())
        indices = np.append(indices, next_ind)
    return indices


def plusplus_coreset_sim(graph, num_points=10, random_state=None, method='poisson', tau=0.1,
                         q=1.0, kernel='gaussian', plot=False, X=None):
    n = graph.num_nodes

    # if want to use 0/1 edge weights, specify kernel = 'uniform'
    if kernel == 'uniform':
        G = gl.graph(graph.adjacency())
    else:
        G = graph

    all_inds = np.arange(G.num_nodes)

    # instantiate the random_state object for making random trials consistent
    if random_state is None:
        random_state = 0
    random_state = check_random_state(random_state)

    # randomly select initial point for coreset
    indices = np.array([random_state.randint(n)])

    similarities = np.zeros(n)  # initialize similarities to coreset vector
    # while still have budget to add to the coreset, propagate dijkstra out
    for j in range(1, num_points):
        x = indices[-1]
        if method == 'poisson':
            sim_to_x = get_poisson_weighting(G, [x], tau=tau)
        else:
            raise NotImplementedError()

        np.maximum(similarities, sim_to_x, out=similarities)

        if plot and X is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            p1 = ax1.scatter(X[:, 0], X[:, 1], c=similarities)
            ax1.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax1.scatter(X[x, 0], X[x, 1], c='pink', marker='^', s=100)
            ax1.set_title("Updated Similarities")
            plt.colorbar(p1, ax=ax1)

            p2 = ax2.scatter(X[:, 0], X[:, 1], c=np.exp(-similarities ** q))
            ax2.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax2.set_title("Sampling Probabilities")
            plt.colorbar(p2, ax=ax2)
            plt.show()

        # sample next point proportionally to the q^th power of the distances
        if q != 1:
            vals = np.cumsum(np.exp(-similarities ** q))
        else:
            vals = np.cumsum(np.exp(-similarities))

        next_ind = np.searchsorted(vals, vals[-1] * random_state.uniform())
        indices = np.append(indices, next_ind)
    return indices


def density_determine_rad(
        graph: gl.graph,
        node: int,
        proportion: float,
        r_0: float = 1.0,
        tol: float = 0.02,
) -> float:
    """Returns the radius, 'r', required for B_r(x) to contain a fixed proportion,
        'proportion', of the nodes in the graph. This uses the bisection method
        and more efficient code could be written in c. Starts by picking
        boundary points for the bisection method. The final radius will satisfy
            -tol <= |B_r(x)| / |V(G)| - proportion <= tol

    Args:
        graph: Graph object.
        node: Node index.
        proportion: Proportion of data desired in B_r(x).
        r_0: Initial radius to try for bisection method. Defaults to 1.0.
        tol: Allowable error tolerance in proportion calculation. Defaults to 0.02.

    Returns:
        Radius r
    """

    num_nodes = graph.num_nodes
    rad = r_0
    dists = graph.dijkstra(bdy_set=[node], max_dist=rad)
    p_current = np.count_nonzero(dists < rad) * 1.0 / num_nodes

    iterations: int = 0
    r_low = 0.0
    r_high = 0.0
    # If within some tolerance of the proportion, just return
    if p_current >= proportion - tol and p_current <= proportion + tol:
        return p_current
    # If radius too large, initialize a, b for bisection
    elif p_current > proportion + tol:
        r_low = 0
        r_high = rad
    # If radius too small, repeatedly increase until we can use bisection
    else:
        while p_current < proportion - tol:
            rad *= 1.5
            dists = graph.dijkstra(bdy_set=[node], max_dist=rad)
            p_current = np.count_nonzero(dists < rad) * 1.0 / num_nodes
        r_low = 0.66 * rad
        r_high = rad

    # Do bisection method to get answer
    while p_current < proportion - tol or p_current > proportion + tol:
        rad = (r_low + r_high) / 2.0
        p_current = np.count_nonzero(dists < rad) * 1.0 / num_nodes

        if p_current > proportion + tol:
            r_high = rad
        elif p_current < proportion - tol:
            r_low = rad
        else:
            return rad

        iterations += 1
        if iterations >= 50:
            print("Too many iterations. Density radius did not converge")
            return rad
    return rad


def coreset_dijkstras(
        graph: gl.graph,
        rad: float,
        data: Optional[np.ndarray] = None,
        initial: Optional[List[int]] = None,
        density_info: Tuple[bool, float, float] = (False, DENSITY_RADIUS, 1.0),
        similarity: str = "euclidean",
        knn_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        plot_steps: bool = False,
) -> List[int]:
    """Runs the Dijkstra's Annulus Coreset (DAC) method outlined in the paper. The
        algorithm uses inner radius which is half of rad. When using density
        radius, the inner radius makes half the proportion of data lie in that
        ball.

    Args:
        graph: Graph object.
        rad: Fixed radius to use in DAC method.
        data: Embedded data. Defaults to None.
        initial: Initial points in coreset. Defaults to None.
        density_info:
            Boolean to decide whether to use density radius.
            Float to determine the density radius.
            Initial spatial radius to try in density_determine_rad.
            Defaults to (False, DENSITY_RADIUS, 1.0).
        similarity:
            Similarity metric to use.
            Refer to gl.weightmatrix.knn for supported inputs.
            Defaults to "euclidean".
        knn_data: Precomputed knn_data. Defaults to None.
        plot_steps:
            Plots all stages of the algorithm.
            Uses first two dimensions of data for plotting.
            Defaults to False.

    Returns:
        Coreset computed from DAC
    """

    perim: List[int] = []
    if initial is None:
        initial = []
    coreset = initial.copy()

    rad_low = rad / 2.0
    rad_high = rad

    use_density, proportion, r_0 = density_info

    # Once all points have been seen, we end this
    points_seen = np.zeros(graph.num_nodes)

    knn_val = graph.weight_matrix[0].count_nonzero()

    # Use distances without a kernel applied
    if knn_data:
        w_dist = gl.weightmatrix.knn(
            data, knn_val, similarity=similarity, kernel="distance", knn_data=knn_data
        )
    else:
        w_dist = gl.weightmatrix.knn(
            data, knn_val, similarity=similarity, kernel="distance"
        )
    # Construct graph from raw distances
    graph_raw_dist = gl.graph(w_dist)

    # Construct the perimeter from the initial set
    for node in initial:
        if use_density:
            rad_low = density_determine_rad(graph_raw_dist, node, proportion / 2.0, r_0)
            rad_high = density_determine_rad(graph_raw_dist, node, proportion, r_0)
        else:
            # Calculate perimeter from new node
            tmp1 = graph_raw_dist.dijkstra(bdy_set=[node], max_dist=rad_high)
            tmp2 = tmp1 <= rad_high
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
            tmp4 = (tmp1 <= rad_low).nonzero()[0]

            # Get rid of points in perimeter too close to new_node
            for x in tmp4:
                if x in perim:
                    perim.remove(x)

            # Add in points in the perimeter of new_node but unseen by old points
            for x in tmp3:
                if x not in perim and points_seen[x] == 0:
                    perim.append(x)

            points_seen[tmp2] = 1

    # If no initial set, the initialize first point
    if len(coreset) == 0:
        # Generate coreset
        new_node = np.random.choice(graph_raw_dist.num_nodes, size=1).item()
        coreset.append(new_node)
        if use_density:
            rad_low = density_determine_rad(
                graph_raw_dist, new_node, proportion / 2.0, r_0
            )
            rad_high = density_determine_rad(graph_raw_dist, new_node, proportion, r_0)
        # Calculate perimeter
        tmp1 = graph_raw_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
        tmp2 = tmp1 <= rad_high
        tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
        # Update perim
        perim = list(tmp3)
        # Update points seen
        points_seen[tmp2] = 1

        if plot_steps and data is not None:
            _dac_plot_fun(data, points_seen, coreset, perim)

    # Generate the coreset from the remaining stuff
    iterations = 0

    # Terminate if we have seen all points and the perimeter is empty
    while np.min(points_seen) == 0 or len(perim) > 0:
        # If perimeter is empty, jump to a new, unseen node
        if len(perim) == 0:
            avail_nodes = (points_seen == 0).nonzero()[0]
            new_node = np.random.choice(avail_nodes, size=1).item()
            coreset.append(new_node)
            if use_density:
                rad_low = density_determine_rad(
                    graph_raw_dist, new_node, proportion / 2.0, r_0
                )
                rad_high = density_determine_rad(
                    graph_raw_dist, new_node, proportion, r_0
                )
            # Calculate perimeter
            tmp1 = graph_raw_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
            tmp2 = tmp1 <= rad_high
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]

            # Update perim and points seen
            perim = list(tmp3)
            points_seen[tmp2] = 1
        else:
            # Select a new node from the perimeter
            new_node = np.random.choice(perim, size=1).item()
            coreset.append(new_node)
            if use_density:
                rad_low = density_determine_rad(
                    graph_raw_dist, new_node, proportion / 2.0, r_0
                )
                rad_high = density_determine_rad(
                    graph_raw_dist, new_node, proportion, r_0
                )

            # Calculate perimeter from new node
            tmp1 = graph_raw_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
            tmp2 = tmp1 <= rad_high
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
            tmp4 = (tmp1 <= rad_low).nonzero()[0]

            # Get rid of points in perimeter too close to new_node
            for x in tmp4:
                if x in perim:
                    perim.remove(x)

            # Add in points in the perimeter of new_node but unseen by old points
            for x in tmp3:
                if x not in perim and points_seen[x] == 0:
                    perim.append(x)

            points_seen[tmp2] = 1

        if plot_steps and data is not None:
            _dac_plot_fun(data, points_seen, coreset, perim)

        if iterations >= 1000:
            break
        iterations += 1
    return coreset


def _dac_plot_fun(
        data: np.ndarray, points_seen: np.ndarray, coreset: List[int], perim: List[int]
) -> None:
    """Function for plotting the intermediate steps of the DAC algorithm. It first checks if
        the dataset is from a square. This indicates that it will use the
        parameters to make nice plots for figures in the paper (eg. larger
        red dots). If it is the square dataset, the plots are saved. The plots
        are always displayed when this function is called.

    Args:
        data: Raw data. Each datapoint must be in 2 dimensions.
        points_seen: Points which have already been seen.
        coreset: Points contained in the coreset.
        perim: Points in the perimeter.
    """
    unit_x_len = np.abs(np.max(data[:, 0]) - np.min(data[:, 0]) - 1) < 0.05
    unit_y_len = np.abs(np.max(data[:, 1]) - np.min(data[:, 1]) - 1) < 0.05
    square_dataset = unit_x_len and unit_y_len

    # The following is for the square dataset
    if square_dataset:
        # Save the initial dataset also
        if len(coreset) == 1:
            plt.scatter(data[:, 0], data[:, 1])
            plt.axis("square")
            plt.axis("off")
            plt.savefig("DAC Plots/coreset0.png", bbox_inches="tight")
            plt.show()
        # If not initial, do this
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(data[points_seen == 1, 0], data[points_seen == 1, 1], c="k")
        plt.scatter(data[coreset, 0], data[coreset, 1], c="r", s=100)
        plt.scatter(data[perim, 0], data[perim, 1], c="y")
        plt.axis("square")
        plt.axis("off")
        plt.savefig(
            "DAC Plots/coreset" + str(len(coreset)) + ".png", bbox_inches="tight"
        )
    else:
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(data[points_seen == 1, 0], data[points_seen == 1, 1], c="k")
        plt.scatter(data[coreset, 0], data[coreset, 1], c="r")
        plt.scatter(data[perim, 0], data[perim, 1], c="y")

    plt.show()
    return


################################################################################
## util functions for batch active learning


def local_maxes_k_new(
        knn_ind: np.ndarray,
        acq_array: np.ndarray,
        k: int,
        top_num: int,
        thresh: int = 0,
) -> np.ndarray:
    """Function to compute the k local maxes of the acquisition function.
        acq_array(v) >= acq_array(u) for all u in neighbors, then v is a local max

    Args:
        knn_ind: Indices for k-nearest neighbors of each point.
    acq_array: Computed acquisition values for each point.
        k: The number of neighbors to include in local_max calculation.
        top_num: The number of local maxes to include.
        thresh: The minimum acquisition value allowable to select a point. Defaults to 0.

    Returns:
        Array of indices for local maxes.
    """
    # Look at the k nearest neighbors
    local_maxes = np.array([])
    K = knn_ind.shape[1]
    if k > K or k == -1:
        k = K

    sorted_ind = np.argsort(acq_array)[::-1]
    local_maxes = np.append(local_maxes, sorted_ind[0])
    global_max_val = acq_array[sorted_ind[0]]
    neighbors = knn_ind[sorted_ind[0], :k]
    sorted_ind = np.setdiff1d(sorted_ind, neighbors, assume_unique=True)

    while len(local_maxes) < top_num and len(sorted_ind) > 0:
        current_max_ind = sorted_ind[0]
        neighbors = knn_ind[current_max_ind, :k]
        acq_vals = acq_array[neighbors]
        sorted_ind = np.setdiff1d(sorted_ind, neighbors, assume_unique=True)
        if acq_array[current_max_ind] >= np.max(acq_vals):
            if acq_array[current_max_ind] < thresh * global_max_val:
                break
            local_maxes = np.append(local_maxes, current_max_ind)

    return local_maxes.astype(int)

####
def local_maxes_k_new_optimized(knn_ind: np.ndarray, acq_array: np.ndarray, k: int, top_num: int, thresh: int = 0) -> np.ndarray:
    K = knn_ind.shape[1]
    k = min(k, K)

    sorted_indices = np.argsort(-acq_array)  # 直接按照降序排序
    local_maxes = []
    considered = set()

    for idx in sorted_indices:
        if len(local_maxes) >= top_num:
            break
        if idx in considered:
            continue

        neighbors = knn_ind[idx, :k]
        acq_vals = acq_array[neighbors]

        if acq_array[idx] >= np.max(acq_vals) and acq_array[idx] >= thresh * acq_array[sorted_indices[0]]:
            local_maxes.append(idx)
            considered.update(neighbors)

    return np.array(local_maxes, dtype=int)
#####

def random_sample_val(val: np.ndarray, sample_num: int) -> np.ndarray:
    """Turns val into a probability array and samples points from it.

    Args:
        val: Initial weights (typically acquisition values).
        sample_num: Number of points to sample.

    Returns:
        Indices to sample.
    """
    # Give all points some probability
    min_tol = 1.0 / len(val)
    val += min_tol - np.min(val)
    probs = val / np.sum(val)
    return np.random.choice(len(val), size=sample_num, replace=False, p=probs)


################################################################################
## implement batch active learning function


def batch_active_learning_experiment(
        X: np.ndarray,
        labels: np.ndarray,
        W: csr_matrix,
        coreset: List[int],
        new_samples: int,
        al_mtd: str,
        acq_fun: str,
        knn_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        method: str = "Laplace",
        use_prior: bool = False,
        batchsize: int = BATCH_SIZE,
        dist_metric: str = "angular",
        knn_size: int = bal_utils.KNN_NUM,
        q: int = 1,
        thresholding: int = 0,
        acc_eps: float = 0,
        adjust_eps_rate=None,
        class_acc_balance=False):
    """Function to run batch active learning experiments. Parameters starting with
        display_all_times are not changed in the experiments.

    Args:
        X: Embedded data. Defaults to None.
        labels: Labels for data.
        W: Weight matrix for the graph.
        coreset: List of points in the coreset.
        new_samples: Total number of points to label.
        al_mtd: Active learning method.
            "local_max": Local max method.
            "global_max": Sequential active learning.
            "acq_sample": Sample proportional to acq(x)^q.
            "random": Random sampling.
            "topn_max": Batchsize points with highest acquisition values.
        acq_fun: Acquisition function to use.
            "mc": Model change.
            "vopt": Variance optimality.
            "uc": Uncertainty.
            "mcvopt": MC + VOPT.
        knn_data: Precomputed knn data. Defaults to None.
        display_all_times: Prints more detailed time consumption.
        method: Graph learning method to use.
            "Laplace": laplace learning
            "rw_Laplace": reweighted laplace learning
            "Poisson": poisson learning
        use_prior: Use priors in graph learning method if true. Defaults to False.
        batchsize: Batch size used. Defaults to BATCH_SIZE.
        dist_metric: Distance metric for embedded data. Defaults to "angular".
        knn_size: Node degree in k-nearest neighbors graph. Defaults to bal_utils.KNN_NUM.
        q: Weighting for acq_sample. Defaults to 1.
        thresholding: Minimum acquisition value to accept. Defaults to 0.

    Returns:
        Final list of labeled points.
        Number of labels at each iteration.
        Model accuracy at each iteration.
        Total time taken (only valid if display == False).
    """

    if knn_data:
        knn_ind, _ = knn_data
    else:
        knn_ind, _ = gl.weightmatrix.knnsearch(
            X, knn_size, method="annoy", similarity=dist_metric
        )

    if al_mtd == "local_max":
        k, thresh = -1, 0

    list_num_labels = []
    list_acc = np.array([]).astype(np.float64)

    train_ind = coreset
    if use_prior:
        class_priors = gl.utils.class_priors(labels)
    else:
        class_priors = None

    if method == "Laplace":
        model = gl.ssl.laplace(W, class_priors=class_priors)
    elif method == "rw_Laplace":
        model = gl.ssl.laplace(W, class_priors, reweighting="poisson")
    elif method == "Poisson":
        model = gl.ssl.poisson(W, class_priors)
    else:
        raise ValueError(f"Invalid choice of method {method}.")

    if acq_fun not in ['uc', 'supuc']:
        print(f"For {acq_fun}, we need to calculate the low-rank eigen decomposition.")
        evals, evecs = model.graph.eigen_decomp(normalization='normalized', k=50)
        C = np.diag(1. / (evals + 1e-11))

    if acq_fun == "mc":
        acq_f = al.model_change(C.copy())
    elif acq_fun == "vopt":
        acq_f = al.v_opt(C.copy())
    elif acq_fun == "uc":
        acq_f = al.unc_sampling()
    elif acq_fun == "mcvopt":
        acq_f = al.model_change_vopt(C.copy())
    elif acq_fun == "supmc":
        acq_f = model_change_supervised(C=C.copy(), y_gt=labels, V=evecs.copy())
    elif acq_fun == "supuc":
        acq_f = uc_supervised(y_gt=labels)
    else:
        raise ValueError(f"Invalid choice of acquisition function {method}.")

    current_labeled_set = train_ind
    current_labels = labels[train_ind]

    # perform classification with GSSL classifier
    # print("start: laplace learning")
    classification = model.fit(current_labeled_set, current_labels)

    current_label_guesses = model.predict()
    # print("end: laplace learning")

    acc = gl.ssl.ssl_accuracy(
        current_label_guesses, labels, current_labeled_set
    )

    # record labeled set and accuracy value
    list_num_labels.append(len(current_labeled_set))
    list_acc = np.append(list_acc, acc)
    acc_diff = 1

    if al_mtd == "global_max":
        batchsize = 1

    remaining_samples: int = new_samples

    iteration: int = 0

    if adjust_eps_rate is not None:
        acc_eps_new = acc_eps * np.exp(- (100 - acc) / adjust_eps_rate)
    else:
        acc_eps_new = acc_eps

    while remaining_samples > 0 and acc_diff > acc_eps_new:
        t = time.time()

        # When you get to the last iteration, don't sample more points than desired
        batchsize = min(batchsize, remaining_samples)

        candidate_inds = np.setdiff1d(np.arange(len(X)), current_labeled_set)
        acq_vals = acq_f.compute(classification, candidate_inds)

        if class_acc_balance:
            class_accuracy = np.zeros(3)
            for i in range(3):
                cla_indx = candidate_inds[labels[candidate_inds] == i]
                class_accuracy[i] = np.sum(labels[cla_indx] == current_label_guesses[cla_indx]) / len(cla_indx)
            conditions = [labels[candidate_inds]==0, labels[candidate_inds]==1, labels[candidate_inds]==2]
            choices = [1, np.sqrt(class_accuracy[0]/class_accuracy[1]), np.sqrt(class_accuracy[0]/class_accuracy[2])]
            acq_weights = np.select(conditions, choices)
            acq_vals = acq_vals * acq_weights


        modded_acq_vals = np.zeros(len(X))
        modded_acq_vals[candidate_inds] = acq_vals

        # print("start: select batch")
        if al_mtd == "local_max":
            batch = local_maxes_k_new_optimized(knn_ind, modded_acq_vals, k, batchsize, thresh)
        elif al_mtd == "global_max":
            batch = candidate_inds[np.argmax(acq_vals)]
        elif al_mtd == "acq_sample":
            batch_inds = random_sample_val(acq_vals ** q, sample_num=batchsize)
            batch = candidate_inds[batch_inds]
        elif al_mtd == "random":
            batch = np.random.choice(candidate_inds, size=batchsize, replace=False)
        elif al_mtd == "topn_max":
            batch = candidate_inds[np.argsort(acq_vals)[-batchsize:]]
        # print("end: select batch")

        if thresholding > 0:
            max_acq_val = np.max(acq_vals)
            batch = batch[modded_acq_vals[batch] >= (thresholding * max_acq_val)]

        current_labeled_set = np.append(current_labeled_set, np.asarray(batch))
        current_labels = np.append(current_labels, labels[batch])

        # print("start: laplace learning")

        classification = model.fit(current_labeled_set, current_labels)
        current_label_guesses = model.predict()
        # print("end: laplace learning")
        acc = gl.ssl.ssl_accuracy(
            current_label_guesses, labels, current_labeled_set
        )

        list_num_labels.append(len(current_labeled_set))
        list_acc = np.append(list_acc, acc)
        acc_diff = np.abs(list_acc[-1] - list_acc[-2])
        if adjust_eps_rate is not None:
            acc_eps_new = acc_eps * np.exp(- (100 - acc) / adjust_eps_rate)
        else:
            acc_eps_new = acc_eps

        iteration += 1
        if isinstance(batch, Iterable):
            remaining_samples -= len(batch)
        else:
            remaining_samples -= 1

        print(f"Time: {time.time() - t}; Accuracy:{acc}.")

    labeled_ind = current_labeled_set

    return labeled_ind, list_num_labels, list_acc
