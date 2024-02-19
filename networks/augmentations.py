from PIL import Image
import numpy as np
from torchvision import transforms
import random
from scipy import ndimage


class RandomPixelAugmentation(object):
    def __init__(self, v, min_val, max_val, p=1.0):
        self.v = v
        self.min_val = min_val
        self.max_val = max_val
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                np_img = img.copy()
            else:
                np_img = np.array(img, dtype=np.float32) / 255.0

            h, w, c = np_img.shape
            max_num_pixels_to_augment = int(self.v * h * w)
            num_pixels_to_augment = np.random.randint(max_num_pixels_to_augment) + 1
            # print(num_pixels_to_augment)

            for _ in range(num_pixels_to_augment):
                x = np.random.randint(0, h)
                y = np.random.randint(0, w)
                increment = np.random.uniform(self.min_val, self.max_val)
                np_img[x, y, :] = np.clip(np_img[x, y, :] + increment, 0, 1)
                # np_img[x, y, :] = np_img[x, y, :] + increment

            if isinstance(img, np.ndarray):
                return np_img
            else:
                return Image.fromarray((np_img * 255).astype(np.uint8))

        return img


class CenterCropResize(object):
    def __init__(self, alpha_min, alpha_max, p=1.0):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            alpha = np.random.uniform(self.alpha_min, self.alpha_max)

            if isinstance(img, np.ndarray):
                h, w, c = img.shape
                new_h, new_w = int(h * alpha), int(w * alpha)
                cropped_img = img[h // 2 - new_h // 2:h // 2 + new_h // 2 + 1, w // 2 - new_w // 2:w // 2 + new_w // 2 + 1]

                # Resize each channel separately
                resized_img = np.zeros((h, w, c))
                for i in range(c):
                    channel = cropped_img[:, :, i]
                    resized_channel = ndimage.zoom(channel,
                                                   (float(h) / channel.shape[0],
                                                    float(w) / channel.shape[1]),
                                                   order=1)
                    resized_img[:, :, i] = resized_channel
                img = resized_img
            else:
                width, height = img.size
                new_width = int(width * alpha)
                new_height = int(height * alpha)
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = (width + new_width) // 2
                bottom = (height + new_height) // 2
                img = img.crop((left, top, right, bottom))
                img = img.resize((width, height), Image.ANTIALIAS)

        return img


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                # 如果是NumPy数组，使用NumPy的翻转
                return np.fliplr(img)
            elif isinstance(img, Image.Image):
                # 如果是PIL图像，使用PIL的方法
                return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return np.flipud(img)
            elif isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.randint(-30, 30)
            if isinstance(img, np.ndarray):
                return self.rotate_np(img, angle)
            elif isinstance(img, Image.Image):
                return img.rotate(angle)
        return img

    @staticmethod
    def rotate_np(img, angle):
        # 旋转NumPy数组
        return ndimage.rotate(img, angle, reshape=False)


class Identity(object):
    def __call__(self, img):
        return img.copy()


CL_transform_cloud = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(p=0.5),
        RandomPixelAugmentation(v=0.1, min_val=1, max_val=1, p=0.5),
        CenterCropResize(alpha_min=0.4, alpha_max=0.8, p=0.5),
        Identity(),
        transforms.ToTensor()
    ])

CL_transform = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(p=0.5),
        CenterCropResize(alpha_min=0.4, alpha_max=0.8, p=0.5),
        Identity(),
        transforms.ToTensor()
    ])
