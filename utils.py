from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import graphlearning as gl
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
import warnings

from scipy.ndimage import distance_transform_edt as distance_transform
from scipy.ndimage import binary_erosion

import matplotlib.pyplot as plt

import rasterio

from networks.CustomDatasets import CustomDataset_NLM
from networks.augmentations import CL_transform, CL_transform_cloud
from networks.networks import MLPEmbedding, SimpleCNN


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class NCropTransform:
    """Create N crops/views of the same image"""

    def __init__(self, transform, num_crops=2):
        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x):
        l = []
        for i in range(self.num_crops):
            l.append(self.transform(x))
        return l


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    if isinstance(model, list):
        parameters = []
        for m in model:
            parameters.extend(list(m.parameters()))
    else:
        parameters = model.parameters()

    optimizer = optim.SGD(parameters,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_loader(opt):
    dataset = np.load(opt.dataset_path, allow_pickle=True).item()
    image_list = []
    label_list = []
    for key in dataset.keys():
        image_list.append(dataset[key]['image'])
        label_list.append(dataset[key]['label'])
    image_array = np.concatenate(image_list, axis=0)
    label_array = np.concatenate(label_list, axis=0)

    if opt.method == 'SupCon' or opt.method == 'SimCLR':
        if opt.cloud_augmentation:
            transform = TwoCropTransform(CL_transform_cloud)
        else:
            transform = TwoCropTransform(CL_transform)
        drop_last = True
        shuffle = True
    elif opt.method == 'Val':
        transform = None
        drop_last = False
        shuffle = False
    else:
        raise ValueError(opt.method)

    dataset_NLM = CustomDataset_NLM(image_array,
                                    label_array,
                                    normalize_fun=None,
                                    margin=opt.margin,
                                    stepsize=opt.stepsize,
                                    gaussian_weight=True,
                                    flatten_x=False,
                                    transform=transform,
                                    down_sample_rate=opt.ratios,
                                    seed=opt.seed)
    print(f"Dataset Size: {len(dataset_NLM)}")

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset_NLM, batch_size=opt.batch_size, shuffle=shuffle,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=drop_last)

    return train_loader


def set_model(opt):
    if opt.model == 'cnn':
        model = SimpleCNN(n=2 * opt.margin + 1, bottleneck_dim=opt.ebd_dim)
    elif opt.model == 'mlp':
        input_dim = (2 * opt.margin + 1) ** 2 * 6
        model = MLPEmbedding(input_dim=input_dim, bottleneck_dim=opt.ebd_dim, encode_layers=3)

    if opt.cp_load_path != 'no':
        model_path = opt.cp_load_path
        model_dict = torch.load(model_path)
        try:
            result = model.load_state_dict(model_dict["model"])
            print(f"Successfully load model from {model_path}. Every key matches exactly.")
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_dict["model"].items():
                name = k.replace(".module", "")  # remove `module.`
                new_state_dict[name] = v
            result = model.load_state_dict(new_state_dict, strict=False)
            missing_keys = result.missing_keys
            print("Missing keys:", missing_keys)
            unexpected_keys = result.unexpected_keys
            print("Unexpected keys:", unexpected_keys)
            print(f"Successfully load model from {model_path}")
    else:
        print(f"Initize the model with random parameters")

    # criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available() & (opt.dev != 'cpu'):
        if torch.cuda.device_count() > 1:  # check for multiple GPU
            model.encoder = torch.nn.DataParallel(model.encoder)
        # model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True

    return model.to(torch.device(opt.dev))


def set_loader_from_tif(opt):
    with rasterio.open(opt.test_dataset_path) as dataset:
        image = dataset.read()

    if opt.test_dataset_type == 'Landsat 5':
        alpha, beta = np.ones(6), np.zeros(6)
    elif opt.test_dataset_type == 'Landsat 8':
        alpha = np.array([0.01856038, 0.02645001, 0.04259213, 0.08720809, 1.47270238, 0.9678328])
        beta = np.array([0.09252381, 0.0682249, 0.05301083, 0.11175688, 0.03906132, 0.01792211])
    else:
        raise ValueError(f"Invalid test_dataset_type {opt.test_dataset_type}.")
    image = (image.transpose(1, 2, 0)[:, :, :]).astype(float)
    image = image * alpha[None, None, :] + beta[None, None, :]
    # plt.imshow(image[:,:,[2,1,0]])
    # plt.axis("off")
    # plt.show()

    image_array = np.asarray([image])
    label_array = np.zeros((image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    dataset_NLM = CustomDataset_NLM(image_array,
                                    label_array,
                                    normalize_fun=None,
                                    margin=opt.margin,
                                    stepsize=1,
                                    gaussian_weight=True,
                                    flatten_x=False,
                                    transform=None,
                                    down_sample_rate=None,
                                    seed=None)
    print(f"Dataset Size: {len(dataset_NLM)}")

    train_loader = torch.utils.data.DataLoader(
        dataset_NLM, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=False)

    return train_loader, image


### custom laplace learning

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


def knn_connected(data, n_labels, k=20, similarity='euclidean'):
    knn_ind, knn_dist = gl.weightmatrix.knnsearch(data, k, similarity=similarity)

    # Restrict to k nearest neighbors
    n = knn_ind.shape[0]
    k = np.minimum(knn_ind.shape[1], k)
    knn_ind = knn_ind[:, :k]
    knn_dist = knn_dist[:, :k]
    knn_data = (knn_ind, knn_dist)

    # Weights
    eps = knn_dist[:, k - 1]
    print(np.min(eps))
    if (eps < 1e-10).any():
        print('eps is too small')
        warnings.warn("Epsilon in KNN is very close to zero.", UserWarning)
    eps += 0.15
    w = 4
    weights = np.exp(-w * knn_dist * knn_dist / eps[:, None] / eps[knn_ind])
    weights_v = -2 * w * np.exp(-w * knn_dist * knn_dist / eps[:, None] / eps[knn_ind]) / eps[:, None] / eps[knn_ind]

    # Flatten knn data and weights
    knn_ind = knn_ind.flatten()
    weights = weights.flatten()
    weights_v = weights_v.flatten()

    # Self indices
    self_ind = np.ones((n, k)) * np.arange(n)[:, None]
    self_ind = self_ind.flatten()

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((weights, (self_ind, knn_ind)), shape=(n, n)).tocsr()

    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)

    n_comp, labels = connected_components(W)
    print('number of components {}'.format(n_comp))
    nolab = np.setdiff1d(np.arange(n_comp), labels[:n_labels])

    for unlab_comp in nolab:  # for each connected component without a labeled sample
        ind = np.argwhere(labels == unlab_comp)[0]  # Get index of single element from the component
        dist = np.sum((data[:n_labels] - data[ind]) ** 2, axis=1)
        neighbor = np.argmin(dist)
        dist = np.amin(dist)
        if (np.exp(-w * dist / eps[ind] / eps[neighbor]) < 1e-8):
            print("Error - added weight is 0")

        weights = np.append(weights, np.exp(-w * dist / eps[ind] / eps[neighbor]))
        self_ind = np.append(self_ind, ind)
        knn_ind = np.append(knn_ind, neighbor)
    print('min weight is ', np.min(weights))

    W = sparse.coo_matrix((weights, (self_ind, knn_ind)), shape=(n, n)).tocsr()

    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)

    return W, knn_data


def one_hot_encode(labels, n_classes='auto'):
    # Number of labels and number of unique classes
    n_labels = len(labels)
    if n_classes == 'auto':
        n_classes = len(np.unique(labels))

    # Initialize the one-hot encoded matrix
    one_hot_matrix = np.zeros((n_labels, n_classes))

    # Set the appropriate elements to 1
    one_hot_matrix[np.arange(n_labels), labels] = 1

    return one_hot_matrix


def stable_conjgrad(A, b, x0=None, max_iter=1e5, tol=1e-10):
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0

    r = b - A @ x
    p = r
    rsold = np.sum(r ** 2, axis=0)

    err = 1
    i = 0
    while (err > tol) and (i < max_iter):
        i += 1
        Ap = A @ p
        alpha = np.zeros_like(rsold)
        alpha[rsold > tol ** 2] = rsold[rsold > tol ** 2] / np.sum(p * Ap, axis=0)[rsold > tol ** 2]
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.sum(r ** 2, axis=0)
        err = np.max(np.sqrt(rsnew))
        beta = np.zeros_like(rsold)
        beta[rsnew > tol ** 2] = rsnew[rsnew > tol ** 2] / rsold[rsnew > tol ** 2]
        p = r + beta * p
        rsold = rsnew

    if err > tol:
        print('max iter reached: ', i, ' iters')

    return x


def laplace(X, train_labels, knn_num=50, epsilon='auto', n_classes='auto', tau=1e-8, similarity='angular',
            train_inds=None, W=None, return_full=False):
    '''
    labeled indices are 0,1,2,...,k-1
    '''

    if W is None:
        W, _ = knn_sym_dist(X, k=knn_num, epsilon=epsilon, similarity=similarity)
    # W, _ = knn_connected(X, n_labels=3, k=50, similarity=similarity)
    L = sparse.csgraph.laplacian(W).tocsr()
    label_matrix = one_hot_encode(train_labels, n_classes)

    if train_inds is None:
        k = label_matrix.shape[0]

        Luu = L[k:, k:]  # Lower Right Corner - unlabelled with unlabelled
        Lul = L[k:, :k]  # Lower Left Rectangle - Labelled and Unlabelled
    else:
        N = W.shape[0]
        unlabeled_inds = np.delete(np.arange(N), train_inds)
        Luu = L[np.ix_(unlabeled_inds, unlabeled_inds)]
        Lul = L[np.ix_(unlabeled_inds, train_inds)]

    m = Luu.shape[0]

    Luu = Luu + sparse.spdiags(tau * np.ones(m), 0, m, m).tocsr()

    M = Luu.diagonal()
    M = sparse.spdiags(1 / np.sqrt(M + 1e-10), 0, m, m).tocsr()

    Pred = stable_conjgrad(M * Luu * M,
                           -M * Lul @ label_matrix)  #
    Pred = M * Pred

    if return_full:
        output = np.zeros((W.shape[0], label_matrix.shape[1])).astype(float)
        if train_inds is None:
            output[:k] = label_matrix
            output[k:] = Pred
        else:
            output[train_inds] = label_matrix
            output[unlabeled_inds] = Pred
        return output
    else:
        return Pred

## evaluation
def boundary_accuracy(pred_labels, gt_labels, d):
    if not (pred_labels.ndim == 2 and gt_labels.ndim == 2 and pred_labels.shape == gt_labels.shape):
        raise ValueError("pred_labels and gt_labels must be 2D numpy arrays of the same shape.")

    edges = np.zeros_like(gt_labels, dtype=bool)
    for c in np.unique(gt_labels):
        class_mask = (gt_labels == c)
        eroded_mask = binary_erosion(class_mask, structure=np.ones((3, 3)), border_value=1)
        edges |= (class_mask ^ eroded_mask)

    dt = distance_transform(~edges)
    boundary_mask = dt <= d

    correct_predictions = np.sum((pred_labels == gt_labels) & boundary_mask)
    total_boundary_pixels = np.sum(boundary_mask)

    return total_boundary_pixels, correct_predictions

def class_accuracy(pred_labels, gt_labels):
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("pred_labels and gt_labels must be numpy arrays of the same shape.")

    unique_classes = np.unique(gt_labels)
    class_counts = np.zeros_like(unique_classes, dtype=int)
    correct_pred_counts = np.zeros_like(unique_classes, dtype=int)

    for i, c in enumerate(unique_classes):
        class_mask = (gt_labels == c)
        class_counts[i] = np.sum(class_mask)
        correct_pred_counts[i] = np.sum((pred_labels == c) & class_mask)

    return class_counts, correct_pred_counts