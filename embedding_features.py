import torch
import numpy as np

from utils import set_loader, set_model
from cli import parse_option

import os

# --num_workers 4 --batch_size 2048 --margin 4 --model cnn --method Val --stepsize 1 --cp_load_path save/SupCon/SupCon_temp/ckpt_epoch_75.pth --down_sample_rate None --dataset_path RiverPIXELS/train_RiverPIXELS.npy

def encode_features(opt, suffix):
    model = set_model(opt)

    val_loader = set_loader(opt)
    device = opt.dev

    encoded_data = None
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.float()
            if encoded_data is None:
                encoded_data = model(data.to(device)).cpu().numpy()
                all_labels = labels.numpy()
            else:
                encoded_data = np.concatenate((encoded_data, model(data.to(device)).cpu().numpy()))
                all_labels = np.concatenate((all_labels, labels.numpy()))

    folder_path = os.path.dirname(opt.dataset_path)
    file_name = os.path.basename(opt.dataset_path)
    file_name, _ = os.path.splitext(file_name)
    save_path = os.path.join(folder_path, f"{file_name}_ebd_features_{suffix}.npy")

    output_dic = {'feature': encoded_data, 'label': all_labels}
    np.save(save_path, output_dic)

if __name__ == "__main__":
    suffix = ''
    opt = parse_option()
    opt.method = 'Val'
    opt.stepsize = 1
    opt.down_sample_rate = None
    encode_features(opt, suffix)
