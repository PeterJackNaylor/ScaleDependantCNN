import pandas as pd
import sys
import numpy as np
from selection_knn import nn_linear, knn_evaluation, random_forest


feats = pd.read_csv(sys.argv[2], index_col=0)
info = pd.read_csv(sys.argv[3], index_col=0).reset_index()

idx_train = info[info["fold"] == "train"].index
x_train = feats.loc[idx_train].values
y_train = info.loc[idx_train, "orderedLabel"].values

idx_test = info[info["fold"] == "test"].index
x_test = feats.loc[idx_test].values
y_test = info.loc[idx_test, "orderedLabel"].values

train_score, test_score = nn_linear(x_train, y_train, x_test, y_test)
forest_train, forest_test = random_forest(x_train, y_train, x_test, y_test)
knn_score, knn_score3 = knn_evaluation(
    x_train, y_train, x_test, y_test, c=len(np.unique(y_train))
)

pd.DataFrame(
    {
        "train_score": float(train_score),
        "test_score": float(test_score),
        "knn_score": knn_score,
        "knn_score3": knn_score3,
        "forest_train": forest_train,
        "forest_test": forest_test,
        "name": sys.argv[1],
    },
    index=[0],
).to_csv("performance.csv")
