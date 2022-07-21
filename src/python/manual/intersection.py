import sys
import numpy as np
import pandas as pd


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def main():

    sel1 = list(np.load(sys.argv[3]))
    sel2 = list(np.load(sys.argv[4]))
    feat_name = intersection(sel1, sel2)
    table = pd.read_csv(sys.argv[2])
    table[feat_name].to_csv(f"{sys.argv[1]}_interdata.csv")

    res = {
        "ascending_knn": [np.load(sys.argv[5])[-1]],
        "descending_knn": [np.load(sys.argv[6])[-1]],
        "ascending_selected": [len(sel1)],
        "descending_selected": [len(sel2)],
        "inter_selected": [len(feat_name)],
        "ascending_train": [np.load(sys.argv[7])[-1]],
        "descending_train": [np.load(sys.argv[8])[-1]],
        "ascending_validation": [np.load(sys.argv[9])[-1]],
        "descending_validation": [np.load(sys.argv[10])[-1]],
        "name": [sys.argv[1] + "_inter"],
    }
    fname = "{}_inter_training_statistics.csv".format(sys.argv[1])
    pd.DataFrame.from_dict(res).to_csv(fname)


if __name__ == "__main__":
    main()
