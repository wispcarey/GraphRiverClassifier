import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset_NLM(Dataset):
    def __init__(self,
                 X_list:np.array,
                 Y_list:np.array,
                 normalize_fun=None,
                 margin=3,
                 stepsize=1,
                 gaussian_weight=False,
                 flatten_x=False,
                 transform=None,
                 down_sample_rate=None,
                 seed=None):
        # self.X_list = [np.moveaxis(X, -1, 0) for X in X_list]  # Change to CHW format
        self.X_list = X_list
        if normalize_fun is not None:
            self.X_list = [normalize_fun(X) for X in self.X_list]
        self.Y_list = Y_list
        self.margin = margin
        self.len_list = [X.shape[0] * X.shape[1] for X in self.X_list]
        # self.total_len = sum(self.len_list)
        self.cum_len_list = np.cumsum(self.len_list)
        self.m_list = [X.shape[0] for X in self.X_list]
        self.n_list = [X.shape[1] for X in self.X_list]
        self.c_list = [X.shape[2] for X in self.X_list]
        # self.stepsize = stepsize
        self.gaussian_weight = gaussian_weight
        self.flatten_x = flatten_x
        self.transform = transform

        if down_sample_rate is not None:
            self.down_sample_rate = down_sample_rate
            self.seed = seed
            select_inds = self.downsample_indices()
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(select_inds)
            self.select_inds = select_inds[::stepsize]
        else:
            self.select_inds = np.arange(0, np.prod(np.asarray(self.Y_list).shape), stepsize)

        if gaussian_weight:
            self.gaussian_weight_matrix = self.create_gaussian_weight_matrix()

    def downsample_indices(self):
        # Uncomment the next line if you want to set a fixed seed for reproducibility
        # np.random.seed(seed)

        # Ensure A is a numpy array
        A = np.asarray(self.Y_list)

        # Flatten A
        flattened_A = A.flatten()

        # Store the downsampled indices
        downsampled_indices = []

        # Create a random number generator with the given seed
        rng = np.random.default_rng(seed=self.seed)

        for i in range(3):  # For each value (0, 1, 2)
            # Find all indices where the value is i
            indices = np.where(flattened_A == i)[0]

            # Downsample
            downsample_size = int(len(indices) * self.down_sample_rate[i])
            downsampled_indices.extend(rng.choice(indices, size=downsample_size, replace=False))

        return downsampled_indices

    def create_gaussian_weight_matrix(self):
        u = np.zeros((2 * self.margin + 1, 2 * self.margin + 1))
        u[self.margin, self.margin] = 1
        G = gaussian_filter(u, self.margin / 2, mode='constant', cval=0)
        G = G / G[self.margin, self.margin]
        return G[np.newaxis, :, :]

    def __len__(self):
        return len(self.select_inds)

    def __getitem__(self, idx0):
        # idx = idx0 * self.stepsize  # multiply idx with stepsize
        idx = self.select_inds[idx0]

        # Find the image this idx belongs to
        image_idx = np.searchsorted(self.cum_len_list, idx + 1)
        # Adjust the index relative to this image
        if image_idx > 0:
            idx = idx - self.cum_len_list[image_idx - 1]

        X = self.X_list[image_idx]

        Y = self.Y_list[image_idx]

        m, n = self.m_list[image_idx], self.n_list[image_idx]

        x_idx, y_idx = divmod(idx, n)

        x_start, x_end = max(0, x_idx - self.margin), min(m, x_idx + self.margin + 1)
        y_start, y_end = max(0, y_idx - self.margin), min(n, y_idx + self.margin + 1)

        patch = np.pad(X[x_start:x_end, y_start:y_end, :],
                       ((self.margin - x_idx if x_idx - self.margin < 0 else 0,
                         x_idx + self.margin + 1 - m if x_idx + self.margin + 1 > m else 0),
                        (self.margin - y_idx if y_idx - self.margin < 0 else 0,
                         y_idx + self.margin + 1 - n if y_idx + self.margin + 1 > n else 0),
                        (0, 0)),
                       mode='reflect')

        # x = torch.from_numpy(patch).float().permute((1,2,0))
        # x = torch.from_numpy(patch).float()
        x = patch
        if self.transform:
            x = self.transform(x)
        else:
            x = transforms.ToTensor()(x)

        if self.gaussian_weight:
            if isinstance(x, torch.Tensor):
                x = x * torch.from_numpy(self.gaussian_weight_matrix)
            else:
                x_new = [xi * torch.from_numpy(self.gaussian_weight_matrix) for xi in x]
                x = x_new

        if self.flatten_x:
            if isinstance(x, torch.Tensor):
                x = x.flatten()
            else:
                x_new = [xi.flatten() for xi in x]
                x = x_new

        y = torch.tensor(Y[x_idx, y_idx], dtype=torch.long)

        return x, y
