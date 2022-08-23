# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
from ast import arg
from os.path import join
import torch
from torch import nn
import copy
import argparse
import numpy as np
from typing import List

from lightly.data import LightlyDataset
from lightly.data import MoCoCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from cell_data import PairTransform, CAM32
from model import fetch_backbone
import pandas as pd
from test import test_moco, get_encoding_moco
from tqdm import tqdm, trange


class MyLightlyDataset(LightlyDataset):
    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        fname = self.index_to_filename(self.dataset, index)
        samples = self.dataset.__getitem__(index)
        if len(samples) == 2:
            sample, target = samples
            return sample, target, fname
        else:
            sample, h, w, target = samples
            return (
                sample,
                target,
                fname,
                h,
                w,
            )


class MyMoCoCollateFunction(MoCoCollateFunction):
    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.

        Args:
            batch:
                A batch of tuples of images, labels, and filenames which
                is automatically provided if the dataloader is built from
                a LightlyDataset.

        Returns:
            A tuple of images, labels, and filenames. The images consist of
            two batches corresponding to the two transformations of the
            input images.

        Examples:
            >>> # define a random transformation and the collate function
            >>> transform = ... # some random augmentations
            >>> collate_fn = BaseCollateFunction(transform)
            >>>
            >>> # input is a batch of tuples (here, batch_size = 1)
            >>> input = [(img, 0, 'my-image.png')]
            >>> output = collate_fn(input)
            >>>
            >>> # output consists of two random transforms of the images,
            >>> # the labels, and the filenames in the batch
            >>> (img_t0, img_t1), label, filename = output

        """
        batch_size = len(batch)

        # list of transformed images
        transforms = [
            self.transform(batch[i % batch_size][0]).unsqueeze_(0)
            for i in range(2 * batch_size)
        ]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]
        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0),
        )

        if len(batch[0]) > 3:
            h = torch.Tensor(np.array([item[3] for item in batch]))
            w = torch.Tensor(np.array([item[3] for item in batch]))
            return transforms, labels, fnames, h, w
        return transforms, labels, fnames


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--data_path",
    default="./tinycells.npy",
    type=str,
)
parser.add_argument(
    "--data_info",
    default="./info.csv",
    type=str,
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "-mb",
    "--memory-bank",
    default=4096,
    type=int,
    metavar="N",
    help="memory bank size (default: 4096)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="seed for initializing training. ",
)
parser.add_argument(
    "--gpu",
    default=None,
    type=int,
    help="GPU id to use.",
)
parser.add_argument(
    "--inject_size",
    dest="inject_size",
    action="store_true",
)
parser.add_argument(
    "--no_size",
    dest="inject_size",
    action="store_false",
)
parser.add_argument(
    "--ks",
    default=3,
    type=int,
    help="kernel size for the first layer",
)
parser.add_argument(
    "--name",
    default="ModelLambda",
    type=str,
    help="Output name",
)
parser.add_argument(
    "--output",
    default="./",
    type=str,
    help="Output folder",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim",
    default=128,
    type=int,
    help="feature dimension (default: 128)",
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t",
    default=20.0,
    type=float,
    help="softmax temperature (default: 0.07)",
)

# options for moco v2
parser.add_argument(
    "--mlp",
    action="store_true",
    help="use mlp head",
)
parser.add_argument(
    "--aug-plus",
    action="store_true",
    help="use moco v2 data augmentation",
)
parser.add_argument(
    "--cos",
    action="store_true",
    help="use cosine lr schedule",
)
parser.add_argument(
    "--k",
    default=40,
    type=int,
    help="Top k most similar images used to predict the label",
)

args = parser.parse_args()

def cam_loader(
    data_path,
    data_info,
    img_size,
    train,
    transform,
    size,
    split,
    batch_size,
    num_workers,
    everyone=False,
):
    cam = CAM32(
        data_path=data_path,
        data_info=data_info,
        train=train,
        transform=transform,
        return_size=size,
        split=split,
        everyone=everyone,
    )
    dataset = MyLightlyDataset.from_torch_dataset(cam)
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    collate_fn = MyMoCoCollateFunction(
        input_size=img_size,
        cj_prob=0,
        random_gray_scale=0,
        gaussian_blur=0,
        vf_prob=0,
        hf_prob=0,
        rr_prob=0,
        min_scale=1,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=not train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def cam_memory_train(
    data_path,
    data_info,
    img_size,
    train,
    transform,
    size,
    split,
    batch_size,
    num_workers,
    everyone=False,
):
    cam = CAM32(
        data_path=data_path,
        data_info=data_info,
        train=train,
        transform=transform,
        return_size=size,
        split=split,
        everyone=everyone,
    )
    dataset = MyLightlyDataset.from_torch_dataset(cam)
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    collate_fn = MyMoCoCollateFunction(
        input_size=img_size,
        cj_prob=0,
        random_gray_scale=0,
        gaussian_blur=0,
        vf_prob=0,
        hf_prob=0,
        rr_prob=0,
        min_scale=1,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(128, 128, 64)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x, h=0, w=0):
        query = self.backbone(x, h, w).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x, h=0, w=0):
        key = self.backbone_momentum(x, h, w).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key


device = "cuda" if torch.cuda.is_available() else "cpu"

name = args.arch
inject_size = args.arch == "ModelSDRN" or args.inject_size
data_path = args.data_path
data_info = args.data_info
batch_size = args.batch_size
num_workers = args.workers

imgsize = np.load(data_path).shape[1]

srn = fetch_backbone(name, 2, args.ks, inject_size, device)
model = MoCo(srn)

model.to(device)


train_cam = cam_loader(
    data_path,
    data_info,
    imgsize,
    train=True,
    transform=PairTransform(train_transform=True, pair_transform=False, size=imgsize),
    size=inject_size,
    split=("train", 0.2, 42),
    batch_size=batch_size,
    num_workers=num_workers,
)

c = len(train_cam.dataset.dataset.classes)

train_cam_memory = cam_memory_train(
    data_path,
    data_info,
    imgsize,
    train=True,
    transform=PairTransform(train_transform=True, pair_transform=False, size=imgsize),
    size=inject_size,
    split=("train", 0.2, 42),
    batch_size=batch_size,
    num_workers=num_workers,
)


val_cam = cam_loader(
    data_path,
    data_info,
    imgsize,
    train=False,
    transform=PairTransform(train_transform=False, pair_transform=False, size=imgsize),
    size=inject_size,
    split=("val", 0.2, 42),
    batch_size=batch_size,
    num_workers=num_workers,
)

test_cam = cam_loader(
    data_path,
    data_info,
    imgsize,
    train=False,
    transform=PairTransform(train_transform=False, pair_transform=False, size=imgsize),
    size=inject_size,
    split=(),
    batch_size=batch_size,
    num_workers=num_workers,
    everyone=True,
)

criterion = NTXentLoss(memory_bank_size=args.memory_bank)
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
results = {"train_loss": [], "test_acc@1": [], "test_acc@3": []}
save_name_pre = "{}_{}_{}_{}".format(
    args.name,
    args.arch,
    args.lr,
    args.weight_decay,
)

best_acc = 0.0
print("Starting Training")
for epoch in trange(1, args.epochs + 1):
    total_loss = 0
    for batch in tqdm(train_cam, leave=False):

        if inject_size:
            (x_query, x_key), _, _, h, w = batch
        else:
            (x_query, x_key), _, _ = batch
        update_momentum(model.backbone, model.backbone_momentum, m=args.moco_m)
        update_momentum(
            model.projection_head,
            model.projection_head_momentum,
            m=args.moco_m,
        )
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        if inject_size:
            query = model(x_query, h, w)
            key = model.forward_momentum(x_key, h, w)
        else:
            query = model(x_query)
            key = model.forward_momentum(x_key)
        loss = criterion(query, key)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 5 == 0:
        results["train_loss"].append(float(total_loss))
        val_acc_1, val_acc_3 = test_moco(
            model.backbone,
            train_cam_memory,
            val_cam,
            args.moco_t,
            args.k,
            c,
            epoch,
            args.epochs,
            inject_size=inject_size,
            gpu=device,
        )
        results["test_acc@1"].append(val_acc_1)
        results["test_acc@3"].append(val_acc_3)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(5, epoch + 1, 5))

        data_frame.to_csv(
            join(args.output, "{}_statistics.csv".format(save_name_pre)),
            index_label="epoch",
        )
        if val_acc_1 > best_acc:
            best_acc = val_acc_1
            best_acc3 = val_acc_3
            torch.save(
                model.state_dict(),
                join(args.output, "{}_model.pth".format(save_name_pre)),
            )
    if epoch % 50 == 0:
        torch.save(
            model.state_dict(),
            join(args.output, "{}_model_{}.pth".format(save_name_pre, epoch)),
        )
    avg_loss = total_loss / len(train_cam)
    print(f"Epoch: {epoch:>02}, loss: {avg_loss:.5f}")


model.load_state_dict(
    torch.load(join(args.output, "{}_model.pth".format(save_name_pre)))
)
embeddings = get_encoding_moco(
    model.backbone, test_cam, inject_size=inject_size, gpu=device
)

print(
    "test1: {:.4f}; test3: {:.4f}, best_acc: {:.4f}".format(
        results["test_acc@1"][-1], results["test_acc@3"][-1], best_acc
    )
)
pd.DataFrame(embeddings.cpu()).to_csv(f"{save_name_pre}_moco.csv")
res = {
    "validation_accuracy_knn": [best_acc],
    "validation_accuracy_knn3": [best_acc3],
    "name": [f"{args.name}"],
}
pd.DataFrame.from_dict(res).to_csv(f"{args.name}_training_statistics.csv")
