from networks.augmentations import *
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random


def show_transformed_images(transform, img):
    # 应用变换
    # if img.ndim == 3:
    #     img = img.transpose((2, 0, 1)).copy()
    print(img.shape)
    transformed_img = transform(img)

    if isinstance(transformed_img, torch.Tensor):
        transformed_img = transformed_img.permute(1, 2, 0).numpy()

    # 显示图片
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f'Original Image {img.shape}.')
    plt.imshow(img[:, :, :3])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Transformed Image {transformed_img.shape}')
    plt.imshow(transformed_img[:, :, :3])
    plt.axis('off')

    # print((img - transformed_img)[:,:,0])

    plt.show()


if __name__ == "__main__":
    import os

    # 在任何其他导入之前设置环境变量
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    np_array = np.random.rand(9, 9)
    np.fill_diagonal(np.flipud(np_array), 1)

    img = np.repeat(np_array[:, :, np.newaxis], 6, axis=2)

    CL_transform = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(p=0.5),
        RandomPixelAugmentation(v=0.25, min_val=0.1, max_val=0.5, p=0.5),
        CenterCropResize(alpha_min=0.4, alpha_max=0.8, p=0.5),
        Identity(),
        transforms.ToTensor()
    ])
    show_transformed_images(CL_transform, img)

    # show_transformed_images(lambda x: TranslateX(x, 0.3))
    # show_transformed_images(transform)
