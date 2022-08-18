import numpy as np
from torch import nn
import torch

from torch.autograd import Variable
import torch.optim as optim


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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min"
    )
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
        optimizer.step()
        scheduler.step(loss)

        if abs(last_loss - loss) < tol:
            return model
        last_loss = loss
    return model



def nn_linear_prediction(X, y, X2, y2, lr=0.1, wd=1e-5, max_epochs=100):
    X = Variable(torch.from_numpy(X)).float()
    y = Variable(torch.from_numpy(y))
    X2 = Variable(torch.from_numpy(X2)).float()
    y2 = Variable(torch.from_numpy(y2))

    model = fit_nn(X, y, lr, wd, max_epochs)
    model.eval()
    with torch.no_grad():
        model.cpu()
        y_pred = model(X)
        y_pred_test = model(X2)
    return y_pred, y_pred_test

def top_1_accuracy(y_pred, y_true):

    return torch.sum(y_pred.argmax(axis=1) == y_true) / y_true.size(0)

def nn_linear(X, y, X2, y2, lr=0.1, wd=1e-5, max_epochs=100):
    y_train_pred, y_test_pred = nn_linear_prediction(X, y, X2, y2, lr, wd, max_epochs)
    y = Variable(torch.from_numpy(y))
    y2 = Variable(torch.from_numpy(y2))

    train_acc = top_1_accuracy(y_train_pred, y)
    test_acc = top_1_accuracy(y_test_pred, y2)
    return train_acc, test_acc

def knn_evaluation_prediction(
    XTrain,
    ytrain,
    XTest,
    k=20,
    c=2,
    temperature=0.5,
):
    n = XTest.size(0)
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
    return pred_labels


def knn_evaluation(X_train, y_train, X_test, y_test, k=20, c=2, temperature=0.5, from_numpy=True):
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
    
    y_test_prediction = knn_evaluation_prediction(XTrain, ytrain, XTest, k, c, temperature)

    acc1 = torch.sum(
        (y_test_prediction[:, :1] == ytest.unsqueeze(dim=-1)).any(dim=-1).float()
    ).item()
    acc3 = torch.sum(
        (y_test_prediction[:, :3] == ytest.unsqueeze(dim=-1)).any(dim=-1).float()
    ).item()
    return acc1 / n, acc3 / n
    