import sys
import pandas as pd
import numpy as np
from glob import glob
import ast 


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
    ## 4 or 6
    if "ssl" in name:
        
        return float(name.split("_")[6])
    elif "moco" in name:
        return float(name.split("_")[4])
    elif "supervised" in name:
        return float(name.split("_")[4])
    else:
        return 0


def extract_wd(name):
    ## 5 or 7
    if "ssl" in name:
        return float(name.split("_")[7])
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

def preproc(performance, data, return_tmp=False):
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
    tr["data"] = tr.data.apply(lambda x: x.replace("padded", ""))

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
    if return_tmp:
        return tmp_mean, tmp

    return tmp_mean

def f_one(r, p):
    if (p + r) == 0:
        return 0
    return 2 * p * r / (p + r)


def compute_weighted_acc(df):
    df = df.copy().reset_index()
    if df.data.values[0] == 'consep':
        df.confusion_matrix = df.precision_4
    wrec_list = []
    prec_list = []
    f_list = []
    for i in df.index:
        cm = df.loc[i, "confusion_matrix"]
        cm = np.array(ast.literal_eval(cm))
        if 'tnbc' in df.data.values[0]:
            for i in range(3):
                cm[3, i] = cm[3:,i].sum()
                cm[i, 3] = cm[i,3:].sum()
            cm[3,3] = cm[3:,3:].sum()
            cm = cm[0:4,0:4]
        recalls = []
        precision = []
        f_score = []
        for i in range(cm.shape[0]):
            if cm[i, :].sum() == 0:
                recalls.append(0)
            else:
                recalls.append(cm[i,i] / cm[i, :].sum())
            if cm[:, i].sum() == 0:
                precision.append(0)
            else:
                precision.append(cm[i,i] / cm[:, i].sum())
            f_score.append(f_one(recalls[i], precision[i]))
        wrec = np.mean(recalls)
        wrec_list.append(wrec)
        wprec = np.mean(precision)
        prec_list.append(wprec)
        f_list.append(np.mean(f_score))
    return np.mean(wrec_list), np.std(wrec_list), np.mean(prec_list), np.std(prec_list), np.mean(f_list), np.std(f_list)

    

        

def merge_all(performance, data, average=True):

    tmp_mean, tmp = preproc(performance, data, return_tmp=True)
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
            tmpi = tmp[tmp["name"] == tmptmp[variable].idxmax()]
        elif type_ == "pretrained":
            variable = "train_score"
            tmpi = tmp[tmp["name"] == tmptmp[variable].idxmax()]
        else:
            variable = "train_score"
            tmpi = tmp[tmp["backbone"] == backbone]
            tmpi = tmp[tmp["name"] == tmptmp[variable].idxmax()]
        if average:
            wacc, wacc_std, wprec, wprec_std, fone, fone_std = compute_weighted_acc(tmpi)
            tmpi_mean = tmpi.groupby(grps).mean().reset_index()
            tmpi_std = tmpi.groupby(grps).std().reset_index()
            tmpi_mean["recall_std"] = (
                f"{wacc * 100:.1f}"
                + " \pm "
                + f"{wacc_std * 100:.1f}"
            )
            tmpi_mean["prec_std"] = (
                f"{wprec * 100:.1f}"
                + " \pm "
                + f"{wprec_std * 100:.1f}"
            )
            tmpi_mean["fscore_std"] = (
                f"{fone * 100:.1f}"
                + " \pm "
                + f"{fone_std * 100:.1f}"
            )
            for var in ["test_score", "knnscoreK=5", "knnscoreK=20", "knnscoreK=40", "knnscoreK=80"]:
                    
                tmpi_mean[var + "_std"] = (
                    f"{tmpi_mean[var].values[0] * 100:.1f}"
                    + " \pm "
                    + f"{tmpi_std[var].values[0] * 100 * 0.438:.1f}"
                )
            tmpi_mean["max_training"] = tmptmp[variable].max()
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
                "recall_std",
                "prec_std",
                "fscore_std",
                "test_score_std",
                # "knnscoreK=5_std",
                # "knnscoreK=20_std",
                # "knnscoreK=40_std",
                # "knnscoreK=80_std",
                "max_training",
            ]
        ]
        tmp.columns = [
            "name",
            "backbone",
            "type",
            "inject_size",
            "recall ({})".format(data),
            "precision ({})".format(data),
            "fscore ({})".format(data),
            "Linear ({})".format(data),
            # "kNN (k=5) ({})".format(data),
            # "kNN (k=20) ({})".format(data),
            # "kNN (k=40) ({})".format(data),
            # "kNN (k=80) ({})".format(data),
            "Selection_score",
        ]
        tabs.append(tmp)
    final = tabs[0]
    for i in range(1, len(tabs)):
        final = final.merge(tabs[i], on=["backbone", "type", "inject_size"])
        final.drop(["name_x", "name_y"], axis=1, inplace=True)
    cols = ['backbone', 'type', 'inject_size', 'recall (tnbc)', 'precision (tnbc)', 'fscore (tnbc)', 'Linear (tnbc)', 'recall (consep)', 'precision (consep)', 'fscore (consep)', 'Linear (consep)']
    import pdb; pdb.set_trace()
    only_kcolumns = ['backbone', 'type', 'inject_size',
       'kNN (k=5) (tnbc)', 
       'kNN (k=20) (tnbc)', 
       'kNN (k=40) (tnbc)',
       'kNN (k=80) (tnbc)', 
       'kNN (k=5) (consep)', 
       'kNN (k=20) (consep)', 
       'kNN (k=40) (consep)',
       'kNN (k=80) (consep)']
    cols = ['backbone', 'type', 'inject_size', 'recall (tnbc)', 'Linear (tnbc)', 'kNN (k=40) (tnbc)', 'recall (consep)', 'Linear (consep)', 'kNN (k=40) (consep)',"Selection_score_x", "Selection_score_y"]
    final.to_csv("./paper_results.csv", index=False, sep=";")
    final = final.loc[[0, 1, 2, 3, 4, 5, 8, 9, 12, 6, 7, 10, 11, 13], :]
    final[only_kcolumns]

if __name__ == "__main__":
    main()
