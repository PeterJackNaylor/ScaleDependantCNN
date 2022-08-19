import os
from os.path import join
import argparse
from train import train
from test import test, get_encoding
from cell_data import setup_data
from thop import profile, clever_format
from model import fetch_model
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


def options():
    parser = argparse.ArgumentParser(
        description="Train supervised NN on cell",
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
        "--temperature",
        default=100.0,
        type=float,
        help="Temperature used in softmax",
    )
    parser.add_argument(
        "--k",
        default=40,
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
        "--workers",
        default=1,
        type=int,
        help="Number of workers",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        help="Number of sweeps over the dataset to train",
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
        "--model_name",
        default="ModelSRN",
        type=str,
        help="Model_name",
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


def main():

    gpu = torch.cuda.is_available()

    opt = options()
    use_amp = False  # mixed precision boolean

    data_inject_size = opt.model_name == "ModelSDRN" or opt.inject_size

    train_loader, memory_loader, val_loader, test_loader = setup_data(
        opt.data_path,
        opt.data_info,
        data_inject_size,
        opt.batch_size,
        opt.workers,
        ssl=False,
    )
    # model setup and optimizer config
    model = fetch_model(
        opt.model_name, len(train_loader.dataset.classes), opt.inject_size, gpu
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

    criterion = nn.CrossEntropyLoss()
    results = {
        "train_loss": [],
        "test_acc@1": [],
        "test_acc@3": [],
        "test_accknn@1": [],
        "test_accknn@3": [],
    }
    outpath = opt.output
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(1, opt.epochs + 1):
        train_loss = train(
            model,
            train_loader,
            optimizer,
            data_inject_size,
            criterion,
            opt.batch_size,
            epoch,
            opt.epochs,
            scaler,
            use_amp,
            gpu,
        )
        if epoch % 5 == 0:
            results["train_loss"].append(train_loss.cpu().detach().numpy())
            test_acc_1, test_acc_3, testknn_acc_1, testknn_acc_3 = test(
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
            results["test_accknn@1"].append(testknn_acc_1)
            results["test_accknn@3"].append(testknn_acc_3)
            # save statistics
            data_frame = pd.DataFrame(
                data=results,
                index=range(5, epoch + 1, 5),
            )
            data_frame.to_csv(
                join(outpath, "{}_statistics.csv".format(opt.name)),
                index_label="epoch",
            )
            if testknn_acc_1 > best_acc:
                best_acc = bestknn_acc = testknn_acc_1
                bestknn_acc_3 = testknn_acc_3
                best_acc1 = test_acc_1
                best_acc3 = test_acc_3
                torch.save(
                    model.state_dict(),
                    join(outpath, "{}_model.pth".format(opt.name)),
                )
        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                join(outpath, "{}_model_{}.pth".format(opt.name, epoch)),
            )
    weights = join(outpath, "{}_model.pth".format(opt.name))
    model.load_state_dict(torch.load(weights))
    _, _, embeddings = get_encoding(
        model, test_loader, inject_size=data_inject_size, gpu=gpu
    )
    print(
        "acc1: {:.4f}; acc3: {:.4f}, knn_1: {:.4f}, knn_3: {:.4f}".format(
            float(best_acc1),
            float(best_acc3),
            float(bestknn_acc),
            float(bestknn_acc_3),
        )
    )
    pd.DataFrame(embeddings.cpu()).to_csv("{}_supervised.csv".format(opt.name))

    res = {
        "validation_accuracy": [best_acc1],
        "validation_accuracy3": [best_acc3],
        "validation_accuracy_knn": [bestknn_acc],
        "validation_accuracy_knn3": [bestknn_acc_3],
        "name": ["{}".format(opt.name)],
    }
    pd.DataFrame.from_dict(res).to_csv(
        "{}_supervised_training_statistics.csv".format(opt.name)
    )


if __name__ == "__main__":
    main()
