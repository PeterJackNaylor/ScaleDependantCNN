import argparse
from cell_data import setup_data
import torch
import torch.nn as nn
import pandas as pd
import torchvision
from tqdm import tqdm


def options():
    parser = argparse.ArgumentParser(
        description="Train a supervised NN on cell crops",
    )
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
        "--output",
        default="./",
        type=str,
        help="Output folder",
    )
    parser.add_argument(
        "--name",
        default="ModelLambda",
        type=str,
        help="Output name",
    )
    parser.set_defaults(corr_neg_one=False)
    args = parser.parse_args()
    return args


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def pretrained_resnet(gpu):
    model = torchvision.models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Identity()

    if gpu:
        model.cuda()
    return model


def step_function(inject_size, gpu=True):
    if not inject_size:

        def step(net, datatuple, encoding=False):
            x, y = datatuple
            if gpu:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            x = torch.nn.Upsample(size=(224, 224))(x)
            y_encoding = net(x)
            return y_encoding, y

    else:

        def step(net, datatuple, encoding=False):
            x, h, w, y = datatuple
            if gpu:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                h, w = h.cuda(non_blocking=True), w.cuda(non_blocking=True)
            x = torch.nn.Upsample(size=(224, 224))(x)
            y_encoding = net(x)
            y_encoding = torch.cat([y_encoding, h, w], dim=1)
            return y_encoding, y

    return step


def get_encoding(net, loader, inject_size, gpu=False):
    net.eval()
    step = step_function(inject_size, gpu)
    embeddings = []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(loader, desc="evaluating"):
            # Trick to reuse step to get encoding,
            # the second element in step is y
            embedding, _ = step(net, data_tuple, encoding=False)
            embeddings.append(embedding[0])  # because net outputs two things
    embeddings = torch.cat(embeddings)
    return embeddings


def main():

    gpu = torch.cuda.is_available()

    opt = options()
    data_inject_size = opt.inject_size

    _, _, _, test_loader = setup_data(
        opt.data_path, opt.data_info, data_inject_size, 1, 1, ssl=False
    )

    model = pretrained_resnet(gpu)

    embeddings = get_encoding(
        model,
        test_loader,
        inject_size=data_inject_size,
        gpu=gpu,
    )

    pd.DataFrame(embeddings.cpu()).to_csv("{}_pretrained.csv".format(opt.name))

    res = {"name": ["{}".format(opt.name)]}
    pd.DataFrame.from_dict(res).to_csv(
        "{}_pretrained_training_statistics.csv".format(opt.name)
    )


if __name__ == "__main__":
    main()
