import numpy as np
import graphlearning as gl
import graphlearning.active_learning as al
import ActiveLearning.batch_active_learning as bal
import time
import warnings

import scipy.sparse.csgraph as csgraph
import scipy.sparse as sparse


def adaptive_K(X, k=25, verbose=False, similarity='angular'):
    num_comp = 2
    while num_comp > 1:
        k = int(k * 2)
        # knn_data = gl.weightmatrix.knnsearch(X, k, method='annoy', similarity=similarity)
        # W = gl.weightmatrix.knn(X, k, kernel='gaussian', knn_data=knn_data)
        W, knn_data = knn_sym_dist(X, k, epsilon='auto', similarity=similarity)
        num_comp, comp = csgraph.connected_components(W)
        if num_comp != 1:
            if verbose:
                print(f"Graph is not connected for k = {k}")

        return W, k, knn_data

def knn_sym_dist(data, k=20, epsilon='auto', similarity='angular'):
    knn_ind, knn_dist = gl.weightmatrix.knnsearch(data, k, similarity=similarity)

    # Restrict to k nearest neighbors
    n = knn_ind.shape[0]
    k = np.minimum(knn_ind.shape[1], k)
    knn_ind = knn_ind[:, :k]
    knn_dist = knn_dist[:, :k]
    knn_data = (knn_ind, knn_dist)

    # Self indices
    self_ind = np.ones((n, k)) * np.arange(n)[:, None]
    self_ind = self_ind.flatten()

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    Dist = sparse.coo_matrix((knn_dist.flatten(), (self_ind, knn_ind.flatten())), shape=(n, n)).tocsr()
    Dist = Dist + Dist.T.multiply(Dist.T > Dist) - Dist.multiply(Dist.T > Dist)

    if epsilon == 'auto':
        eps = Dist.max(axis=1).toarray().flatten()
    else:
        eps = epsilon * np.ones(n)
    if (eps < 1e-6).any():
        warnings.warn("Epsilon in KNN is very close to zero.", UserWarning)
    eps = np.maximum(eps, 1e-6)

    # weights
    rows, cols, values = sparse.find(Dist)
    W_values = np.exp(-4 * values * values / eps[rows] / eps[cols])

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((W_values, (rows, cols)), shape=(n, n)).tocsr()

    return W, knn_data


def adaptive_active_learning(all_features,
                             all_labels,
                             acc_eps=0.5,
                             acc_eps_adjust_rate=2,
                             al_batch_size=15,
                             sample_batch_size=65536,
                             initial_num=30,
                             use_pp_coreset=True,
                             prev_samples=True,
                             max_new_samples=3000,
                             acq_fun='uc',
                             use_prior=False,
                             class_acc_balance=False):
    total_num = len(all_features)
    num_batches = (total_num - 1) // sample_batch_size + 1

    for s_batch_ind in range(num_batches):
        print(f"Start the sampling step {s_batch_ind}.")
        t = time.time()
        start_ind = s_batch_ind * sample_batch_size
        end_ind = min((s_batch_ind + 1) * sample_batch_size, total_num)
        features = all_features[start_ind:end_ind]
        labels = all_labels[start_ind:end_ind]

        num_labels = 3

        if use_pp_coreset:
            W, k, knn_data = adaptive_K(features, k=25, verbose=False)
            G = gl.graph(W)
            initial = gl.trainsets.generate(labels, rate=1)
            num_points = initial_num - len(initial)
            train_ind = bal.plusplus_coreset(G, num_points=num_points, random_state=None, method='dijkstra', eik_p=1.0,
                                             tau=0.1, ofs=0.2, q=1.0, knn_dist=None, kernel='gaussian', plot=False,
                                             X=features, initial=initial)
        else:
            train_ind = gl.trainsets.generate(labels, rate=int(initial_num / num_labels))

        if s_batch_ind > 0 and prev_samples:
            features = np.concatenate((select_fvecs, features), axis=0)
            labels = np.concatenate((select_labels, labels), axis=0)
            train_ind = np.concatenate((np.arange(len(select_labels)), len(select_labels) + train_ind))

        W, k, knn_data = adaptive_K(features, k=25, verbose=False)
        print(f"k={k}.")

        labeled_ind, list_num_labels, list_acc = bal.batch_active_learning_experiment(
        X=features, labels=labels, W=W, coreset=train_ind, new_samples=max_new_samples,
        al_mtd='local_max', acq_fun=acq_fun, knn_data=knn_data, method="Laplace",
        use_prior=use_prior, batchsize=al_batch_size, dist_metric="angular", knn_size=k,
        q=1, thresholding=0, acc_eps=acc_eps, adjust_eps_rate=acc_eps_adjust_rate,
        class_acc_balance=class_acc_balance
        )

        if prev_samples or s_batch_ind == 0:
            if s_batch_ind == 0:
                new_sample_num = len(labeled_ind)
            else:
                new_sample_num = len(labeled_ind) - len(select_labels)
            select_fvecs = features[labeled_ind]
            select_labels = labels[labeled_ind]
        else:
            new_sample_num = len(labeled_ind)
            select_fvecs = np.concatenate((select_fvecs, features[labeled_ind]), axis=0)
            select_labels = np.concatenate((select_labels, labels[labeled_ind]), axis=0)

        result_string = f"""
Sample Batch [{s_batch_ind}]: 
Total Number of Samples [{len(select_fvecs)}]; 
Number of New Samples [{new_sample_num}]; 
Accuracy of Current Sample Batch [{list_acc[-1]}];
Time [{time.time() - t}].                           
"""
        print(result_string)

    return select_fvecs, select_labels
