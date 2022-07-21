import torch
from tqdm import tqdm


def step_function(inject_size, gpu=True):
    if not inject_size:

        def step(net, datatuple, encoding=False):
            x, y = datatuple
            if gpu:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if not encoding:
                y_pred = net(x)
                return y_pred, y
            else:
                y_pred, encoding = net(x, return_embedding=True)
                return y_pred, y, encoding

    else:

        def step(net, datatuple, encoding=False):
            x, h, w, y = datatuple
            if gpu:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                h, w = h.cuda(non_blocking=True), w.cuda(non_blocking=True)
            y_pred = net(x, h, w)
            if not encoding:
                y_pred = net(x, h, w)
                return y_pred, y
            else:
                y_pred, encoding = net(x, h, w, return_embedding=True)
                return y_pred, y, encoding

    return step


def step_function_bt(inject_size, gpu=True):
    if not inject_size:

        def step(net, datatuple):
            (pos_1, pos_2), _ = datatuple
            if gpu:
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(
                    non_blocking=True
                )
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            return feature_1, out_1, feature_2, out_2

    else:

        def step(net, datatuple):
            (pos_1, pos_2), (h1, h2), (w1, w2), _ = datatuple
            if gpu:
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(
                    non_blocking=True
                )
                h1, h2 = h1.cuda(non_blocking=True), h2.cuda(non_blocking=True)
                w1, w2 = w1.cuda(non_blocking=True), w2.cuda(non_blocking=True)
            feature_1, out_1 = net(pos_1, h1, w1)
            feature_2, out_2 = net(pos_2, h2, w2)
            return feature_1, out_1, feature_2, out_2

    return step


def train(
    net,
    loader,
    opti,
    inject_size,
    loss_fn,
    bs,
    epoch,
    epochs,
    scaler,
    use_amp,
    gpu=True,
):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(loader)
    step = step_function(inject_size, gpu)
    for data_tuple in train_bar:
        with torch.cuda.amp.autocast(enabled=use_amp):
            y_pred, y = step(net, data_tuple)
            loss = loss_fn(y_pred, y)

        scaler.scale(loss).backward()
        scaler.step(opti)
        scaler.update()
        opti.zero_grad(set_to_none=True)
        # loss.backward()

        # opti.step()
        total_loss += loss
        total_num += bs
        text = "Train Epoch [{}/{}] Loss: {:.4f}".format(
            epoch, epochs, total_loss / total_num
        )
        train_bar.set_description(text)
    return total_loss / total_num


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train_bt(
    net,
    loader,
    opti,
    inject_size,
    bs,
    epoch,
    epochs,
    lmbda,
    corr_neg_one,
    scaler,
    use_amp,
    gpu=True,
):
    # train for one epoch to learn unique features

    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(loader)
    step = step_function_bt(inject_size, gpu)
    for data_tuple in train_bar:
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, out_1, _, out_2 = step(net, data_tuple)
            # Barlow Twins

            # normalize the representations along the batch dimension
            out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
            out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)

            # cross-correlation matrix
            c = torch.matmul(out_1_norm.T, out_2_norm) / bs

            # loss
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            if corr_neg_one is False:
                # the loss described in the original Barlow Twin's paper
                # encouraging off_diag to be zero
                off_diag = off_diagonal(c).pow_(2).sum()
            else:
                # inspired by HSIC
                # encouraging off_diag to be negative ones
                off_diag = off_diagonal(c).add_(1).pow_(2).sum()
            loss = on_diag + lmbda * off_diag

        scaler.scale(loss).backward()
        scaler.step(opti)
        scaler.update()
        opti.zero_grad(set_to_none=True)

        total_num += bs
        total_loss += loss.item() * bs
        if corr_neg_one is True:
            off_corr = -1
        else:
            off_corr = 0
        prog_bar = (
            "Train Epoch: [{}/{}] Loss: {:.4f} off_corr:{} lmbda:{:.4f} bsz:{}"
        )
        train_bar.set_description(
            prog_bar.format(
                epoch,
                epochs,
                total_loss / total_num,
                off_corr,
                lmbda,
                bs,
            )
        )
    return total_loss / total_num
