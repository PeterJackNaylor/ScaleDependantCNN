import torch
import numpy as np
import pandas as pd
import sys
from tqdm import trange
from sklearn.preprocessing import StandardScaler

from evaluation_function import nn_linear

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
            train_scores_index.append(train_score)
            test_scores_index.append(val_score)

    best_feat = np.argmax(train_scores_index)
    return (
        sf + [indexes[best_feat]],
        train_scores_index[best_feat],
        test_scores_index[best_feat],
    )


def remove_feature(X, X2, y, y2, sf, rmf, prev):
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
            train_scores.append(train_score)
            val_scores.append(val_score)

    best_feat = np.argmax(train_scores)
    sf.remove(indexes[best_feat])
    return (
        sf,
        train_scores[best_feat],
        val_scores[best_feat],
        rmf + [indexes[best_feat]],
    )


def selection(X, X2, y, y2, stepwise="ascending"):
    n, p = X.shape
    if stepwise == "ascending":
        selected_features = []
        prev = 0
        rm_features = None
    elif stepwise == "descending":
        selected_features = list(range(p))
        rm_features = []
        prev, prev3 = nn_linear(X, y, X2, y2)
    scores = []
    test_scores = []
    if stepwise == "ascending":
        for i in trange(p):
            selected_features, score, test_s = add_feature(
                X,
                X2,
                y,
                y2,
                selected_features,
            )
            scores.append(score)
            test_scores.append(test_s)
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
                rm_features,
            ) = remove_feature(
                X,
                X2,
                y,
                y2,
                selected_features,
                rm_features,
                prev,
            )
            scores.append(score)
            test_scores.append(test_score)
            if score > np.max(scores) - 0.01:
                prev = score
            else:
                break
    return (
        selected_features,
        scores,
        test_scores,
        rm_features,
    )


def load_data(path):
    x = pd.read_csv(path, index_col=0).reset_index(drop=True)

    x["orderedLabel"] = x["orderedLabel"].astype(int)

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
    return X_train, y_train, X_test, y_test, features


def main():

    X_train, y_train, X_test, y_test, feat = load_data(f"{sys.argv[1]}.csv")

    outputs = selection(
        X_train,
        X_test,
        y_train,
        y_test,
        stepwise=sys.argv[3],
    )
    selected_feats = outputs[0]
    train_score = outputs[1]
    test_score = outputs[2]
    rm_feat = outputs[3]
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

    np.save(f"train_score_{sys.argv[3]}.npy", train_score)
    np.save(f"test_score_{sys.argv[3]}.npy", test_score)


if __name__ == "__main__":
    main()
