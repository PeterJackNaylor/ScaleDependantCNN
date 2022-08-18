import sys
import pandas as pd
import numpy as np
from glob import glob


def return_backbone(name):
    padded = "_padded" if "padded" in name else ""
    if "ModelSRN" in name:
        return "SRN" + padded
    elif "ModelSDRN" in name:
        return "SDRN" + padded
    else:
        return "CS" + padded


def name_type(name):
    if "LAMB" in name:
        return "ssl"
    elif "moco" in name:
        return "moco"
    elif "Model" in name:
        return "supervised"
    else:
        return "cs"


def return_type(name):
    if "ssl" in name:
        return "BT"
    elif "moco" in name:
        return "MoCo"
    elif "supervised" in name:
        return "S"
    elif "pretrained" in name:
        return "pretrained"
    else:
        return "Us"


def inject_size_fn(name):
    if "inject_size" in name:
        return "Size"
    elif "cs" in name:
        return "Size"
    else:
        return ""


def return_title(name):
    if name == "knn_score":
        return "kNN"
    elif name == "test_score":
        return "Linear"
    elif name == "knn_score3":
        return "kNN-3"
    elif name == "forest_test":
        return "Non-linear"
    else:
        raise NotImplementedError("name unknown")


def extract_lr(name):

    if "ssl" in name:
        return float(name.split("_")[4])
    elif "moco" in name:
        return float(name.split("_")[4])
    elif "supervised" in name:
        return float(name.split("_")[4])
    else:
        return 0


def extract_wd(name):
    if "ssl" in name:
        return float(name.split("_")[5])
    elif "moco" in name:
        return float(name.split("_")[5])
    elif "supervised" in name:
        return float(name.split("_")[5])
    else:
        return 0


def extract_lambda(name):
    if "ssl" in name:
        return float(name.split("_")[-5].split("-")[1])
    elif "supervised" in name:
        return 0
    else:
        return 0


types = ["inter", "ascending", "descending", "union"]


def read_other_files(path, perf):
    new_table = []
    for f in glob(f"{path}/*training_statistics.csv"):
        if perf != f:
            data = pd.read_csv(f, index_col=0)
            el = [i for i in types if i in f]
            if el:
                data["name"] += "_" + el[0]
            tmp = data.groupby("name").mean()
            tmp = tmp * np.where(tmp < 1, 100, 1)
            new_table.append(tmp)
    return pd.concat(new_table)


def read_special_moco():
    table = pd.read_csv("./train_moco_statistics.csv")
    table["name"] = table["name"].astype(str) + "_moco"
    table = table.groupby("name").mean().reset_index()
    return table


def g(score, type):
    if type == "S":
        return score * 100
    else:
        return score


def h(name):
    if "cs" in name:
        data, type_ = name.split("cs")
        type_ = "_inter" if "_intersection" == type_ else type_
        name = data + type_ + type_ + "_cs"
    return name


def gg(name, t):
    if "moco" in name:
        return name
    else:
        return name + "_" + t

def preproc(performance, data):
    tmp = pd.read_csv(performance, index_col=0)
    tmp["name"] = tmp.name.apply(lambda x: h(x))
    tmp["data"] = tmp.name.apply(lambda x: x.split("_")[0].replace("cs", "").replace("padded", ""))
    tmp["backbone"] = tmp.name.apply(lambda x: return_backbone(x))
    tmp["type"] = tmp.name.apply(lambda x: return_type(x))
    tmp["inject_size"] = tmp.name.apply(lambda x: inject_size_fn(x))
    tmp["lr"] = tmp.name.apply(lambda x: extract_lr(x))
    tmp["wd"] = tmp.name.apply(lambda x: extract_wd(x))
    tmp["lambda"] = tmp.name.apply(lambda x: extract_lambda(x))
    tmp = tmp[tmp["data"] == data]

    # selection process
    tr = read_other_files("./", performance).reset_index()
    # training_results = read_special_moco()
    tr["data"] = tr.name.apply(lambda x: x.split("_")[0])
    tr["type"] = tr.name.apply(lambda x: name_type(x))
    tr["name"] = tr.apply(lambda x: gg(x["name"], x["type"]), axis=1)

    tr = tr[tr["data"] == data].set_index("name")
    tr = tr.drop(["data", "type"], axis=1)
    tmp_mean = (
        tmp.groupby(["name", "data", "backbone", "type", "inject_size"])
        .mean()
        .reset_index()
        .set_index("name")
    )
    tmp_std = (
        tmp.groupby(["name", "data", "backbone", "type", "inject_size"])
        .std()
        .reset_index()
        .set_index("name")
    )
    rm = ["data", "backbone", "type", "inject_size", "lr", "wd", "lambda"]
    tmp_std = tmp_std.drop(rm, axis=1)
    tmp_mean = tmp_mean.drop("data", axis=1)

    tmp_mean = tmp_mean.join(tr)

    list_col = [el + "_std" for el in tmp_std.columns]
    tmp_std.columns = list_col
    tmp_mean = tmp_mean.join(tmp_std)
    tmp_mean["validation_accuracy_knn"] = tmp_mean.apply(
        lambda x: g(x["validation_accuracy_knn"], x["type"]), axis=1
    )
    return tmp_mean

def merge_all(performance, data, average=True):

    tmp_mean = preproc(performance, data)
    grps = ["backbone", "type", "inject_size"]
    keys = list(tmp_mean.groupby(grps).mean().index)
    final_table = []
    for key in keys:
        backbone, type_, inject_size = key
        tmptmp = tmp_mean.loc[
            (
                (tmp_mean.backbone == backbone)
                & (tmp_mean.type == type_)
                & (tmp_mean.inject_size == inject_size)
            )
        ]
        if type_ in ["BT", "MoCo", "S"]:
            variable = "validation_accuracy_knn"
            tmpi = tmp[tmp["name"] == tmptmp.index[tmptmp[variable].argmax()]]
        elif type_ == "pretrained":
            variable = "train_score"
            tmpi = tmp[tmp["name"] == tmptmp.index[tmptmp[variable].argmax()]]
        else:
            variable = "train_score"
            tmpi = tmp[tmp["backbone"] == backbone]
            tmpi = tmp[tmp["name"] == tmptmp.index[tmptmp[variable].argmax()]]
        if average:
            tmpi_mean = tmpi.groupby(grps).mean().reset_index()
            tmpi_std = tmpi.groupby(grps).std().reset_index()
            for var in ["test_score", "knn_score"]:
                tmpi_mean[var + "_std"] = (
                    f"{tmpi_mean[var].values[0] * 100:.1f}"
                    + " \pm "
                    + f"{tmpi_std[var].values[0] * 100:.1f}"
                )
            tmpi_mean["name"] = tmpi.loc[tmpi.index[0], "name"].values[0]
            final_table.append(tmpi_mean)
        else:
            final_table.append(tmpi)
    results = pd.concat(final_table)
    return results


def main():
    tmp = pd.read_csv(sys.argv[1], index_col=0)
    tmp["data"] = tmp.name.apply(lambda x: x.split("_")[0].replace("cs", "").replace("padded", ""))
    available_data = list(tmp.data.unique())
    tabs = []
    for data in available_data:
        tmp = merge_all(sys.argv[1], data)
        tmp = tmp[
            [
                "name",
                "backbone",
                "type",
                "inject_size",
                "test_score_std",
                "knn_score_std",
            ]
        ]
        tmp.columns = [
            "name",
            "backbone",
            "type",
            "inject_size",
            "Linear ({})".format(data),
            "knn ({})".format(data),
        ]
        tabs.append(tmp)
    final = tabs[0]
    for i in range(1, len(tabs)):
        final.merge(tmp[i], on=["backbone", "type", "inject_size"])
        final.drop(["name_x", "name_y"], axis=1, inplace=True)
    final.to_csv("./paper_results.csv", index=False, sep=";")
    # allt.loc[[0, 1, 2, 3, 4, 9, 10, 5, 6, 11, 12, 7, 8, 13, 14], :]


if __name__ == "__main__":
    main()
