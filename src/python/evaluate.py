import pandas as pd
import sys
import numpy as np
from evaluation_function import nn_linear_prediction

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

feats = pd.read_csv(sys.argv[2], index_col=0)
info = pd.read_csv(sys.argv[3], index_col=0).reset_index()

idx_train = info[info["fold"] == "train"].index
x_train = feats.loc[idx_train].values
y_train = info.loc[idx_train, "orderedLabel"].values

idx_test = info[info["fold"] == "test"].index
x_test = feats.loc[idx_test].values
y_test = info.loc[idx_test, "orderedLabel"].values

ytrain_pred, y_test_pred = nn_linear_prediction(x_train, y_train, x_test, y_test)

ytrain_pred = np.array(ytrain_pred)
y_test_pred = np.array(y_test_pred)

train_score = (ytrain_pred.argmax(axis=1) == y_train).mean()
test_score = (y_test_pred.argmax(axis=1) == y_test).mean()

labels = np.unique(y_train)
labels.sort()
labels = list(labels)

y_test_pred = np.array(y_test_pred)


results = pd.DataFrame(
    {
        "train_score": float(train_score),
        "test_score": float(test_score),
        "name": sys.argv[1],
    },
    index=[0],
)

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred.argmax(axis=1), labels=labels, average=None)

for i in range(len(precision)):
    results[f"precision_{labels[i]}"] = precision[i]
    results[f"recall_{labels[i]}"] = recall[i]
    results[f"fscore_{labels[i]}"] = fscore[i]
    results[f"support_{labels[i]}"] = support[i]

confusion_matrix_res = confusion_matrix(y_test, y_test_pred.argmax(axis=1), labels=labels)
first_ = ['[' + ','.join(map(str, l)) + ']' for l in confusion_matrix_res]
second_ = '[' + ','.join(map(str, first_)) + ']'
results["confusion_matrix"] = second_

results.to_csv("performance.csv")
