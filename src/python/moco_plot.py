import sys
import pandas as pd
import numpy as np

from paper_table import preproc
import plotly.express as px
def main():
    data = 'tnbc'
    tmp = preproc(sys.argv[1], data)
    tmp = tmp.reset_index()
    tmp["bs"] = tmp.name.apply(lambda x: x.split("_")[-2])
    tmp["MB"] = tmp.name.apply(lambda x: x.split("_")[-3])
    for var in ["test_score"]:
        tmp[var + "_std"] = np.array([
            f"{tmp[var].values[i] * 100:.1f}"
            + " \pm "
            + f"{tmp[var + '_std'].values[i] * 100:.1f}"
        for i in range(len(tmp[var + '_std'].values))])

    tmp = tmp[
        [
            "name",
            "bs",
            "MB",
            "lr",
            "wd",
            "test_score_std",
        ]
    ]
    tmp.columns = [
        "name",
        "BS",
        "MB",
        "LR",
        "WD",
        "Linear ({})".format(data),
    ]
    tmp = tmp.drop("name", axis=1)
    tmp["BS"] = tmp["BS"].astype(int)
    tmp["MB"] = tmp["MB"].astype(int)
    tmp = tmp.sort_values(["MB", "BS", "LR", "WD"])
    tmp.to_csv("moco_experiments.csv")



if __name__ == "__main__":
    main()
