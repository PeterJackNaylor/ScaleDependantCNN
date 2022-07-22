import argparse
import os
from os.path import join

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format

from train import train_bt
from test import test_bt, get_encoding_bt
from cell_data import setup_data

from model import fetch_model_ssl

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def options():
    parser = argparse.ArgumentParser(description="Train SimCam")
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
        "--feature_dim",
        default=128,
        type=int,
        help="Feature dim for latent vector",
    )
    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="Temperature used in softmax",
    )
    parser.add_argument(
        "--k",
        default=200,
        type=int,
        help="Top k most similar images used to predict the label",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "--output",
        default="./",
        type=str,
        help="Output folder",
    )
    parser.add_argument(
        "--model_name",
        default="ModelSRN",
        type=str,
        help="Model_name",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of workers",
    )
    parser.add_argument(
        "--name",
        default="ModelLambda",
        type=str,
        help="Output name",
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--wd",
        default=1e-6,
        type=float,
        help="Weight decay",
    )
    # for barlow twins

    parser.add_argument(
        "--lmbda",
        default=0.005,
        type=float,
        help="Lambda that controls the on- and off-diagonal terms",
    )
    parser.add_argument(
        "--corr_neg_one",
        dest="corr_neg_one",
        action="store_true",
    )
    parser.add_argument(
        "--corr_zero",
        dest="corr_neg_one",
        action="store_false",
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
    parser.set_defaults(corr_neg_one=False)

    # args parse
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    use_amp = False  # mixed precision boolean

    gpu = torch.cuda.is_available()

    opt = options()

    print(opt)
    data_inject_size = opt.model_name == "ModelSDRN" or opt.inject_size

    # setup data
    train_loader, memory_loader, val_loader, test_loader = setup_data(
        opt.data_path,
        opt.data_info,
        data_inject_size,
        opt.batch_size,
        opt.workers,
        ssl=True,
    )
    # model setup and optimizer config
    model = fetch_model_ssl(
        opt.model_name,
        opt.inject_size,
        opt.feature_dim,
        gpu,
    )
    if data_inject_size:
        fake_input = (
            torch.randn(1, 3, 32, 32),
            torch.Tensor([[32]]),
            torch.Tensor([[32]]),
        )
    else:
        fake_input = (torch.randn(1, 3, 32, 32),)

    if gpu:
        model = model.cuda()
        fake_input = (i.cuda() for i in fake_input)

    flops, params = profile(model, inputs=fake_input)
    flops, params = clever_format([flops, params])
    print("# Model Params: {} FLOPs: {}".format(params, flops))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    c = len(train_loader.dataset.classes)

    # training loop
    results = {"train_loss": [], "test_acc@1": [], "test_acc@3": []}

    corr_neg_one_str = "neg_corr_" if opt.corr_neg_one else ""
    save_name_pre = "{}{}_{}_{}_{}".format(
        corr_neg_one_str, opt.lmbda, opt.feature_dim, opt.batch_size, opt.name
    )

    outpath = opt.output
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(1, opt.epochs + 1):
        train_loss = train_bt(
            model,
            train_loader,
            optimizer,
            data_inject_size,
            opt.batch_size,
            epoch,
            opt.epochs,
            opt.lmbda,
            opt.corr_neg_one,
            scaler,
            use_amp,
            gpu,
        )

        if epoch % 5 == 0:
            results["train_loss"].append(train_loss)
            test_acc_1, test_acc_3 = test_bt(
                model,
                memory_loader,
                val_loader,
                opt.temperature,
                opt.k,
                c,
                epoch,
                opt.epochs,
                inject_size=data_inject_size,
                gpu=gpu,
            )
            results["test_acc@1"].append(test_acc_1)
            results["test_acc@3"].append(test_acc_3)
            # save statistics
            data_frame = pd.DataFrame(
                data=results,
                index=range(5, epoch + 1, 5),
            )
            data_frame.to_csv(
                join(outpath, "{}_statistics.csv".format(save_name_pre)),
                index_label="epoch",
            )
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                best_acc3 = test_acc_3
                torch.save(
                    model.state_dict(),
                    join(outpath, "{}_model.pth".format(save_name_pre)),
                )
        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                join(outpath, "{}_model_{}.pth".format(save_name_pre, epoch)),
            )

    model.load_state_dict(
        torch.load(join(outpath, "{}_model.pth".format(save_name_pre)))
    )
    embeddings = get_encoding_bt(
        model, test_loader, inject_size=data_inject_size, gpu=gpu
    )

    print(
        "test1: {:.4f}; test3: {:.4f}, best_acc: {:.4f}".format(
            results["test_acc@1"][-1], results["test_acc@3"][-1], best_acc
        )
    )
    pd.DataFrame(embeddings.cpu()).to_csv("{}_ssl.csv".format(save_name_pre))
    res = {
        "validation_accuracy_knn": [best_acc],
        "validation_accuracy_knn3": [best_acc3],
        "name": ["{}".format(opt.name)],
    }
    pd.DataFrame.from_dict(res).to_csv(
        "{}_ssl_training_statistics.csv".format(opt.name)
    )
