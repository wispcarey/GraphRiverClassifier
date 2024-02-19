import numpy as np
from utils import one_hot_encode

class model_change_supervised():
    """Model Change
    ===================

    Active learning algorithm that selects points that will produce the greatest change in the model.

    Examples
    --------
    ```py
    import graphlearning.active_learning as al
    import graphlearning as gl
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(labels, rate=5)
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    # compute initial, low-rank (spectral truncation) covariance matrix
    evals, evecs = model.graph.eigen_decomp(normalization='normalized', k=50)
    C = np.diag(1. / (evals + 1e-11))
    AL = gl.active_learning.active_learner(model, gl.active_learning.model_change, train_ind, y[train_ind], C=C.copy(), V=evecs.copy())

    for i in range(10):
        query_points = AL.select_queries() # return this iteration's newly chosen points
        query_labels = y[query_points] # simulate the human in the loop process
        AL.update(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=y)
        plt.scatter(X[AL.labeled_ind,0],X[AL.labeled_ind,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1)
        plt.show()
        print(AL.labeled_ind)
        print(AL.labels)
    ```

    Reference
    ---------
    [1] Miller, K. and Bertozzi, A. L., “Model-change active learning in graph-based semi-supervised learning,”
    (Oct. 2021). arXiv: 2110.07739.

    [2] Karzand, M. and Nowak, R. D., “Maximin active learning in overparameterized model classes,” IEEE
    Journal on Selected Areas in Information Theory 1, 167–177 (May 2020).
    """

    def __init__(self, C, y_gt, V, gamma2=0.1 ** 2.):
        # y_gt is the ground truth for all labels
        assert (C.shape[0] == C.shape[1]) or (V is not None)
        self.C = C.copy()
        self.V = V
        self.gamma2 = gamma2
        self.y_gt = one_hot_encode(y_gt, n_classes=3)

        self.storage = 'trunc'

    def compute(self, u, candidate_ind):
        unc_terms = (np.sum(u[candidate_ind] - self.y_gt[candidate_ind], axis=1)) ** 2

        Cavk = self.C @ self.V[candidate_ind, :].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (self.gamma2 + np.array(
            [np.inner(self.V[k, :], Cavk[:, i]) for i, k in enumerate(candidate_ind)]))
        return unc_terms * col_norms / diag_terms

    def update(self, query_ind, query_labels):
        for k in query_ind:
            vk = self.V[k]
            Cavk = self.C @ vk
            ip = np.inner(vk, Cavk)
            self.C -= np.outer(Cavk, Cavk) / (self.gamma2 + ip)
        return

class uc_supervised():
    def __init__(self, y_gt):
        # y_gt is the ground truth for all labels
        self.y_gt = one_hot_encode(y_gt, n_classes=3)

    def compute(self, u, candidate_ind):
        unc_terms = (np.sum(u[candidate_ind] - self.y_gt[candidate_ind], axis=1)) ** 2

        return unc_terms
