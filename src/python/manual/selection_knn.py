import torch
import numpy as np
import pandas as pd
import sys
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier


class Net2(nn.Module):
    def __init__(self, num_class=2, size=128):
        super(Net, self).__init__()
        self.inner = nn.Linear(size, size * 2, bias=True)
        self.bn = nn.BatchNorm1d(size * 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(size * 2, num_class, bias=True)
        self.do = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.inner(x)
        # x = self.bn(x)
        x = self.do(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, num_class=2, size=128):
        super(Net, self).__init__()
        self.fc = nn.Linear(size, num_class, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


def fit_nn(X, y, lr=0.001, wd=1e-6, max_epochs=100, tol=1e-5):
    model = Net(size=X.shape[1], num_class=np.unique(y).shape[0])
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()
        model.cuda()

    last_loss = 1e10
    # scheduler = optim.lr_scheduler.StepLR(
    #   optimizer,
    #   step_size=100,
    #   gamma=0.1
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min"
    )  # , verbose=True)
    for epoch in range(max_epochs):
        # Clear gradient buffers because we don't want any
        # gradient from previous epoch to carry forward,
        # dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(X)

        # get loss for the predicted output
        loss = criterion(outputs, y)
        loss.backward()
        # print(loss)
        optimizer.step()
        scheduler.step(loss)
        # print(scheduler.get_last_lr())
        if abs(last_loss - loss) < tol:
            return model
        last_loss = loss
        # if loss > best_loss:
        #     print("Early stopping")
        #     return model
        # best_loss = loss
    # import pdb; pdb.set_trace()
    return model


def nn_linear(X, y, X2, y2, lr=0.1, wd=1e-5, max_epochs=100):
    X = Variable(torch.from_numpy(X)).float()
    y = Variable(torch.from_numpy(y))
    X2 = Variable(torch.from_numpy(X2)).float()
    y2 = Variable(torch.from_numpy(y2))

    model = fit_nn(X, y, lr, wd, max_epochs)
    model.eval()
    with torch.no_grad():
        model.cpu()
        y_pred = model(X)
        train_acc = torch.sum(y_pred.argmax(axis=1) == y)
        train_acc = train_acc / y.size(0)
        y_pred_test = model(X2)
        test_acc = torch.sum(y_pred_test.argmax(axis=1) == y2)
        test_acc = test_acc / y2.size(0)
    return train_acc, test_acc


def random_forest(X, y, X2, y2):

    model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2, random_state=0
    )
    model.fit(X, y)
    train_score = model.score(X, y)
    test_score = model.score(X2, y2)
    return train_score, test_score


def linear_model(X, y, X2, y2, C=1.0, max_iter=100):
    model = LogisticRegression(
        penalty="l2", multi_class="multinomial", C=1e-6, max_iter=max_iter
    )
    model.fit(X, y)
    train_score = model.score(X, y)
    test_score = model.score(X2, y2)
    return train_score, test_score


def knn_evaluation(
    X_train,
    y_train,
    X_test,
    y_test,
    k=20,
    c=2,
    temperature=0.5,
    from_numpy=True,
):

    if from_numpy:
        n = X_test.shape[0]
        ytrain = torch.from_numpy(y_train)
        ytest = torch.from_numpy(y_test)
        XTrain = torch.from_numpy(X_train).T
        XTest = torch.from_numpy(X_test)
    else:
        n = X_test.size(0)
        ytrain, ytest = y_train, y_test
        XTrain, XTest = X_train.t(), X_test
    sim_matrix = torch.mm(XTest, XTrain)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(ytrain.expand(n, -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / temperature).exp()
    # counts for each class
    one_hot_label = torch.zeros(n * k, c, device=sim_labels.device)
    # [B*K, C]

    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1).to(torch.int64), value=1.0
    )
    # weighted score ---> [B, C]

    pred_scores = torch.sum(
        one_hot_label.view(n, -1, c) * sim_weight.unsqueeze(dim=-1), dim=1
    )

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    total_top1 = torch.sum(
        (pred_labels[:, :1] == ytest.unsqueeze(dim=-1)).any(dim=-1).float()
    ).item()
    total_top3 = torch.sum(
        (pred_labels[:, :3] == ytest.unsqueeze(dim=-1)).any(dim=-1).float()
    ).item()
    return total_top1 / n, total_top3 / n


def add_feature(X, X2, y, y2, sf, c):
    indexes = []
    train_scores_index = []
    test_scores_index = []
    for i in range(X.shape[1]):
        if i not in sf:
            indexes.append(i)
            train_score, val_score = nn_linear(
                X[:, sf + [i]],
                y,
                X2[:, sf + [i]],
                y2,
            )
            # knn_score = knn_evaluation(X[:,sf+[i]], y, X2[:,sf+[i]], y2, c=c)
            train_scores_index.append(train_score)
            test_scores_index.append(val_score)
            # knn_scores_index.append(knn_score)
    best_feat = np.argmax(train_scores_index)
    knn_score, knn_score3 = knn_evaluation(
        X[:, sf + [indexes[best_feat]]],
        y,
        X2[:, sf + [indexes[best_feat]]],
        y2,
        c=c,
    )
    # best_feat = np.argmax(knn_scores_index)
    return (
        sf + [indexes[best_feat]],
        train_scores_index[best_feat],
        test_scores_index[best_feat],
        knn_score,
        knn_score3,
    )


def remove_feature(X, X2, y, y2, sf, rmf, c, prev):
    indexes = []
    train_scores = []
    val_scores = []
    for i in range(X.shape[1]):
        if i not in rmf:
            indexes.append(i)
            tmp_f = sf.copy()
            tmp_f.remove(i)
            train_score, val_score = nn_linear(
                X[:, tmp_f],
                y,
                X2[:, tmp_f],
                y2,
            )
            # score = knn_evaluation(X[:,tmp_f], y, X2[:,tmp_f], y2, c=c)
            train_scores.append(train_score)
            val_scores.append(val_score)

            # scores_index.append(score)
    best_feat = np.argmax(train_scores)
    # best_feat = np.argmax(scores_index)
    knn_score, knn_score3 = knn_evaluation(
        X[:, sf + [indexes[best_feat]]],
        y,
        X2[:, sf + [indexes[best_feat]]],
        y2,
        c=c,
    )
    sf.remove(indexes[best_feat])
    return (
        sf,
        train_scores[best_feat],
        val_scores[best_feat],
        knn_score,
        knn_score3,
        rmf + [indexes[best_feat]],
    )


def selection(X, X2, y, y2, c, stepwise="ascending"):
    n, p = X.shape
    if stepwise == "ascending":
        selected_features = []
        prev = 0
        rm_features = None
    elif stepwise == "descending":
        selected_features = list(range(p))
        rm_features = []
        prev, prev3 = knn_evaluation(X, y, X2, y2, c=c)
    scores = []
    test_scores = []
    knn_scores = []
    knn_scores3 = []
    if stepwise == "ascending":
        for i in trange(p):
            selected_features, score, test_s, knn_s, knn_s3, = add_feature(
                X,
                X2,
                y,
                y2,
                selected_features,
                c,
            )
            scores.append(score)
            test_scores.append(test_s)
            knn_scores.append(knn_s)
            knn_scores3.append(knn_s3)
            if score > prev:
                prev = score
            else:
                break

    elif stepwise == "descending":
        for i in trange(p - 1):
            (
                selected_features,
                score,
                test_score,
                knn_score,
                knn_score3,
                rm_features,
            ) = remove_feature(
                X,
                X2,
                y,
                y2,
                selected_features,
                rm_features,
                c,
                prev,
            )
            scores.append(score)
            test_scores.append(test_score)
            knn_scores.append(knn_score)
            knn_scores3.append(knn_score3)
            if score > np.max(scores) - 0.01:
                prev = score
            else:
                break
    return (
        selected_features,
        scores,
        test_scores,
        knn_scores,
        knn_scores3,
        rm_features,
    )


def load_data(path):
    x = pd.read_csv(path, index_col=0).reset_index(drop=True)

    x["orderedLabel"] = x["orderedLabel"].astype(int)

    c = np.unique(x["orderedLabel"])

    idx_train = x[x["fold"] == "train"].index
    idx_test = x[x["fold"] == "test"].index
    y_train = np.array(x.loc[idx_train, "orderedLabel"])
    y_test = np.array(x.loc[idx_test, "orderedLabel"])

    features = list(x.columns)

    for f in [
        "Centroid_x",
        "Centroid_y",
        "BBox_y_min",
        "BBox_y_max",
        "BBox_x_min",
        "BBox_x_max",
        "name",
        "patch",
        "orderedLabel",
        "Label",
        "fold",
    ]:
        features.remove(f)

    X_train = np.array(x.loc[idx_train, features])
    X_test = np.array(x.loc[idx_test, features])
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    return X_train, y_train, X_test, y_test, features, c


def main():

    X_train, y_train, X_test, y_test, feat, c = load_data(f"{sys.argv[1]}.csv")

    nc = len(c)
    outputs = selection(
        X_train,
        X_test,
        y_train,
        y_test,
        nc,
        stepwise=sys.argv[3],
    )
    selected_feats = outputs[0]
    train_score = outputs[1]
    test_score = outputs[2]
    knn_score = outputs[3]
    knn_score3 = outputs[4]
    rm_feat = outputs[5]
    print("Selected features:")
    selected_feat = np.array(feat)[selected_feats]
    print(selected_feat)
    np.save(f"selected_feat_{sys.argv[3]}.npy", selected_feat)

    if rm_feat and sys.argv[3]:
        print("Remove features:")
        rm_feat = np.array(feat)[rm_feat]
        print(rm_feat)
        np.save(f"removed_feat_{sys.argv[3]}.npy", rm_feat)
    print("Scores:")
    print("Train:", train_score)
    print("Test:", test_score)
    print("KNN acc1:", knn_score)
    print("KNN acc3:", knn_score3)
    np.save(f"train_score_{sys.argv[3]}.npy", train_score)
    np.save(f"test_score_{sys.argv[3]}.npy", test_score)
    np.save(f"knn_score_{sys.argv[3]}.npy", knn_score)
    np.save(f"knn_score3_{sys.argv[3]}.npy", knn_score3)


if __name__ == "__main__":
    main()
