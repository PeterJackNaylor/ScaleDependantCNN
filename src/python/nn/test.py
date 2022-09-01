from tqdm import tqdm
import torch
from train import step_function

from evaluation_function import knn_evaluation


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top
    predictions for the specified values of k
    In top-5 accuracy you give yourself
    credit for having the right answer
    if the right answer appears in your
    top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/
    how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g.
    scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g.
    (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your
    models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1)
    and the true label was bird (3) you get zero
    but if it were either cat or dog you'd
    accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...]
    depending on your topk input
    """
    with torch.no_grad():
        maxk = max(topk)
        # max number labels we will consider
        # in the right choices for out model

        _, y_pred = output.topk(k=maxk, dim=1)
        # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()
        # [B, maxk] -> [maxk, B] Expects input to be <= 2-D
        # tensor and transposes dimensions 0 and 1.

        target_reshaped = target.view(1, -1).expand_as(
            y_pred
        )  # [B] -> [B, 1] -> [maxk, B]
        correct = y_pred == target_reshaped
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )  # [k, B] -> [kB]
            fiwtmt = flattened_indicator_which_topk_matched_truth
            tot_correct_topk = fiwtmt.float().sum(dim=0, keepdim=True)
            # [kB] -> [1]
            topk_acc = tot_correct_topk  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        # list of topk accuracies for entire batch [topk1, topk2, ... etc]
        return list_topk_accs


def test(
    net,
    memory_data_loader,
    test_data_loader,
    temperature,
    k,
    c,
    epoch,
    epochs,
    inject_size=False,
    gpu=False,
):
    net.eval()
    step = step_function(inject_size, gpu)
    total_top1, total_top1_knn, total_top3, total_top3_knn, total_num = (
        0.0,
        0.0,
        0.0,
        0.0,
        0,
    )
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc="Feature extracting"):
            pred, target, feature = step(net, data_tuple, encoding=True)
            target_bank.append(target)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        # [N]
        feature_labels = (
            torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            y_pred, y, feature = step(net, data_tuple, encoding=True)

            bs = y.size(0)
            res = accuracy(y_pred, y, (1, 3))
            total_top1 += float(res[0].cpu())
            total_top3 += float(res[1].cpu())
            total_num += bs
            top1, top3 = knn_evaluation(
                feature_bank,
                feature_labels,
                feature,
                y,
                k=k,
                c=c,
                temperature=temperature,
                from_numpy=False,
            )
            total_top1_knn += top1 * bs
            total_top3_knn += top3 * bs
            prog_bar = (
                "Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@3:{:.2f}%"
                + "Accknn@1:{:.2f}% Accknn@3:{:.2f}%"
            )
            test_bar.set_description(
                prog_bar.format(
                    epoch,
                    epochs,
                    total_top1 / total_num * 100,
                    total_top3 / total_num * 100,
                    total_top1_knn / total_num * 100,
                    total_top3_knn / total_num * 100,
                )
            )

    return (
        total_top1 / total_num,
        total_top3 / total_num,
        total_top1_knn / total_num,
        total_top3_knn / total_num,
    )


def get_encoding(net, loader, inject_size, gpu=False):
    net.eval()
    step = step_function(inject_size, gpu)
    total_top1, total_top3, total_num = 0.0, 0.0, 0
    embeddings = []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(loader, desc="evaluating"):
            y_pred, y, embedding = step(net, data_tuple, encoding=True)
            bs = y.size(0)
            res = accuracy(y_pred, y, (1, 3))
            total_top1 += res[0]
            total_top3 += res[1]
            total_num += bs
            embeddings.append(embedding)
    embeddings = torch.cat(embeddings)
    return total_top1 / total_num, total_top3 / total_num, embeddings


def get_encoding_bt(net, loader, inject_size, gpu=False):
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


def test_bt(
    net,
    memory_data_loader,
    test_data_loader,
    temperature,
    k,
    c,
    epoch,
    epochs,
    inject_size=False,
    gpu=False,
):
    # test for one epoch, use weighted knn to
    # find the most similar images' label to assign the test image
    net.eval()
    step = step_function(inject_size, gpu)
    total_top1, total_top3, total_num = 0.0, 0.0, 0
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc="Feature extracting"):
            (feature, _), target = step(net, data_tuple, encoding=False)
            target_bank.append(target)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        # [N]
        feature_labels = (
            torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (feature, _), target = step(net, data_tuple, encoding=False)
            n = feature.size(0)
            total_num += n
            top1, top3 = knn_evaluation(
                feature_bank,
                feature_labels,
                feature,
                target,
                k=k,
                c=c,
                temperature=temperature,
                from_numpy=False,
            )
            total_top1 += top1 * n
            total_top3 += top3 * n
            test_bar.set_description(
                "Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@3:{:.2f}%".format(
                    epoch,
                    epochs,
                    total_top1 / total_num * 100,
                    total_top3 / total_num * 100,
                )
            )

    return total_top1 / total_num * 100, total_top3 / total_num * 100


def step_function_moco(inject_size, gpu=True):
    if not inject_size:

        def step(net, datatuple, encoding=False):
            (xq, xk), y, fname = datatuple
            if gpu:
                x, y = xq.cuda(non_blocking=True), y.cuda(non_blocking=True)
            y_pred = net(x)
            return y_pred, y

    else:

        def step(net, datatuple, encoding=False):
            (xq, xk), y, fname, h, w = datatuple
            if gpu:
                x, y = xq.cuda(non_blocking=True), y.cuda(non_blocking=True)
                h, w = h.cuda(non_blocking=True), w.cuda(non_blocking=True)
            y_pred = net(x, h, w)
            return y_pred, y

    return step


def test_moco(
    net,
    memory_data_loader,
    test_data_loader,
    temperature,
    k,
    c,
    epoch,
    epochs,
    inject_size=False,
    gpu=False,
):
    # test for one epoch, use weighted knn to find
    # the most similar images' label to assign the test image

    net.eval()
    step = step_function_moco(inject_size, gpu)
    total_top1, total_top3, total_num = 0.0, 0.0, 0
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc="Feature extracting"):
            feature, target = step(net, data_tuple)
            target_bank.append(target)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        # [N]
        feature_labels = (
            torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            feature, target = step(net, data_tuple)
            n = feature.size(0)
            total_num += n
            top1, top3 = knn_evaluation(
                feature_bank,
                feature_labels,
                feature,
                target,
                k=k,
                c=c,
                temperature=temperature,
                from_numpy=False,
            )
            total_top1 += top1 * n
            total_top3 += top3 * n
            test_bar.set_description(
                "Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@3:{:.2f}%".format(
                    epoch,
                    epochs,
                    total_top1 / total_num * 100,
                    total_top3 / total_num * 100,
                )
            )

    return total_top1 / total_num * 100, total_top3 / total_num * 100


def get_encoding_moco(net, loader, inject_size, gpu=False):
    net.eval()
    step = step_function_moco(inject_size, gpu)
    embeddings = []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(loader, desc="evaluating"):
            # Trick to reuse step to get encoding,
            # the second element in step is y
            embedding, _ = step(net, data_tuple)
            embeddings.append(embedding)  # because net outputs two things
    embeddings = torch.cat(embeddings)
    return embeddings
