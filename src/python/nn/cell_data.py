from torchvision.datasets import CIFAR10
from torchvision import transforms
import pandas as pd
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from numpy import pi
import torchvision.transforms.functional as TF
import random


class CAM32(CIFAR10):
    def __init__(
        self,
        data_path: str,
        data_info: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_size: bool = False,
        everyone: bool = False,
        split: tuple = ("train", 0.2, 42),
    ) -> None:

        super(CIFAR10, self).__init__(
            ".", transform=transform, target_transform=target_transform
        )
        data = np.load(data_path)
        label = pd.read_csv(data_info, index_col=0).reset_index(drop=True)

        label["Label"] = label["Label"].astype(int)

        if not everyone:
            tag = "train" if train else "test"
            idx = label[label["fold"] == tag].index
            if train:
                y_subset = np.array(label.loc[idx, "orderedLabel"])
                train_idx, val_idx, _, _ = train_test_split(
                    idx,
                    y_subset,
                    test_size=split[1],
                    random_state=split[2],
                    shuffle=True,
                    stratify=y_subset,
                )
                idx = train_idx if split[0] == "train" else val_idx
        else:
            idx = label.index

        self.data = data[idx]
        self.targets = np.array(label.loc[idx, "orderedLabel"])
        self.table = label.loc[idx].reset_index().copy()

        self.classes = np.unique(label["orderedLabel"])

        self.return_size = return_size

        self.h = self.table.Height
        self.w = self.table.Width

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.return_size:
            h, w = self.h.loc[index], self.w.loc[index]
            h = np.array(h)[..., np.newaxis].astype("float32")
            w = np.array(w)[..., np.newaxis].astype("float32")
        else:
            h, w = np.zeros(target.shape), np.zeros(target.shape)

        if self.transform is not None:
            img, h, w = self.transform(img, h, w)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_size:
            return img, h, w, target
        else:
            return img, target


def my_rotation(image, h, w, p, angle_min=-90, angle_max=90):
    if random.random() > 1 - p:
        angle = random.randint(angle_min, angle_max)
        image = TF.rotate(image, angle)
        cos_theta = np.cos(angle / 180 * pi)
        sin_theta = np.abs(np.sin(angle / 180 * pi))
        f_h = h * cos_theta + w * sin_theta
        f_w = h * sin_theta + w * cos_theta
        h = f_h.round()
        w = f_w.round()
    # more transforms ...
    return image, h, w


class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img, h, w):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        sh, sw = img.size
        i, j, newh, neww = self.get_params(img, self.scale, self.ratio)
        h, w = (h * newh / sh).round(), (w * neww / sw).round()
        return (
            TF.resized_crop(
                img,
                i,
                j,
                newh,
                neww,
                self.size,
                self.interpolation,
            ),
            h,
            w,
        )


resize_crop_object = MyRandomResizedCrop(32, scale=(0.8, 1.0))
resize_crop_object128 = MyRandomResizedCrop(128, scale=(0.8, 1.0))


def my_resizecrop(image, h, w, p, size=32):
    if random.random() > 1 - p:
        if size == 32:
            image, h, w = resize_crop_object(image, h, w)
        elif size == 128:
            image, h, w = resize_crop_object128(image, h, w)

    return image, h, w


def size_transform(x, h, w, p, size=32):
    x, h, w = my_rotation(x, h, w, p)
    x, h, w = my_resizecrop(x, h, w, p, size)
    return x, h, w

training_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        ),
    ]
    )

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        ),
    ]
)

class PairTransform:
    def __init__(self, train_transform=True, pair_transform=True, size=32):
        self.train_transform = train_transform
        if self.train_transform is True:
            self.transform = training_transforms
        else:
            self.transform = test_transforms
        self.pair_transform = pair_transform
        self.size = size
    def __call__(self, x, h=0, w=0):
        if self.pair_transform is True:
            if self.train_transform:
                x1, h1, w1 = size_transform(x.copy(), h.copy(), w.copy(), 0.8, self.size)
                x2, h2, w2 = size_transform(x, h, w, 0.8, self.size)
            else:
                x1, x2 = x, x.copy()
            y1 = self.transform(x1)
            y2 = self.transform(x2)
            return (y1, y2), (h1, h2), (w1, w2)
        else:
            if self.train_transform:
                x, h, w = size_transform(x, h, w, 0.8, self.size)
            return self.transform(x), h, w


def setup_data(
    data_path,
    data_info,
    inject_size,
    batch_size,
    workers,
    ssl=True,
):
    
    size = np.load(data_path).shape[1]
    seed = 42
    split = 0.2
    if ssl:
        train_transform = PairTransform(train_transform=True, size=size)
    else:
        train_transform = PairTransform(train_transform=True, pair_transform=False, size=size)
    test_transform = PairTransform(
        train_transform=False,
        pair_transform=False,
    )
    train_data = CAM32(
        data_path=data_path,
        data_info=data_info,
        train=True,
        transform=train_transform,
        return_size=inject_size,
        split=("train", split, seed),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    memory_data = CAM32(
        data_path=data_path,
        data_info=data_info,
        train=True,
        transform=test_transform,
        return_size=inject_size,
        split=("train", split, seed),
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    val_data = CAM32(
        data_path=data_path,
        data_info=data_info,
        train=True,  # confusing but we split the training data for validation
        transform=test_transform,
        return_size=inject_size,
        split=("val", split, seed),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_data = CAM32(
        data_path=data_path,
        data_info=data_info,
        train=False,
        transform=test_transform,
        return_size=inject_size,
        everyone=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, memory_loader, val_loader, test_loader


# if __name__ == "__main__":
#     import matplotlib.pylab as plt
#     data_path = "./to_ignore/ConSep/latest_data/consep_tinycells.npy"
#     data_info = "./to_ignore/ConSep/latest_data/consep.csv"
#     inject_size = True
#     split = 0.2
#     seed = 42
#     train_transform = Transform(train_transform = False)
#     train_data = CAM32(data_path=data_path,
#                         data_info=data_info,
#                         train=True,
#                         transform=train_transform,
#                         return_size=inject_size,
#                         split=("train", split, seed))
#     def f(img):
#         img = np.array(img)
#         return img
#     fig, axes = plt.subplots(10, 3, figsize=(10, 20))
#     t = MyRandomResizedCrop(size=32, scale=(0.5, 1.5), ratio=(0.5, 2.0))
#     for i in range(10):
#         img, h, w, y = train_data.__getitem__(i)
#         img = (img - img.min()) / (img.max() - img.min()) * 255
#         img = np.array(img).astype(np.uint8)
#         img = np.rollaxis(img, 0, img.ndim)
#         axes[i, 0].imshow(img)
#         axes[i, 0].set_title(f'h={h}, w={w}', fontstyle='italic')
#         rimg, rh, rw = my_rotation(transforms.ToPILImage()(img), h, w, p=1)
#         axes[i, 1].imshow(rimg)
#         axes[i, 1].set_title(f'h={rh}, w={rw}', fontstyle='italic')
#         timg, th, tw = t(transforms.ToPILImage()(img), h, w)
#         axes[i, 2].imshow(timg)
#         axes[i, 2].set_title(f'h={th}, w={tw}', fontstyle='italic')
#     plt.show()
