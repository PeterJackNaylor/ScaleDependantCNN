import os
import numpy as np
from glob import glob
from optparse import OptionParser
from tqdm import tqdm
import pandas as pd

import skimage.io as io
import scipy.io

from extractor.extract_nuclei import bin_extractor
from extractor.feature_object import (
    PixelSize,
    MeanIntensity,
    Centroid,
    Elongation,
    Circularity,
    StdIntensity,
    Granulometri,
    LBP,
    Label,
    ChannelMeanIntensity,
    ChannelStdIntensity,
)


def create_mapping(lbl):
    old_idx = list(lbl[np.argsort(lbl)])
    return {old_idx[i]: i for i in range(len(old_idx))}


coordinates = [
    "Centroid_x",
    "Centroid_y",
    "Width",
    "Height",
    "BBox_x_min",
    "BBox_y_min",
    "BBox_x_max",
    "BBox_y_max",
]
list_f = []
for d in [0, 4]:
    list_f.append(PixelSize(f"Pixel_sum_{d}", d))
    list_f.append(MeanIntensity(f"Intensity_mean_{d}", d))
    cfname = "Channel_Intensity_mean_0{}_{}"
    op_names = [cfname.format(el, d) for el in range(3)]
    list_f.append(ChannelMeanIntensity(op_names, d))
    list_f.append(StdIntensity(f"Intensity_std_{d}", d))
    cfname = "Channel_Intensity_std_0_c{}_{}"
    op_names = [cfname.format(el, d) for el in range(3)]
    list_f.append(ChannelStdIntensity(op_names, d))
    list_f.append(LBP(["LPB"], d))
    list_f.append(
        Granulometri(
            [
                f"Grano_1_{d}",
                f"Grano_2_{d}",
                f"Grano_3_{d}",
                f"Grano_4_{d}",
                f"Grano_5_{d}",
            ],
            d,
            [1, 2, 3, 4, 5],
        )
    )

list_f.append(Elongation("Elongation", 0))
list_f.append(Circularity("Circularity", 0))
list_f.append(Centroid(coordinates, 0))
list_f.append(Label("Label", 0))


def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    Parameters
    ----------
    path: string, path for the creation of the folder to check/create
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def gene_data(p, datatype="tnbc"):
    if datatype == "tnbc":
        pattern = os.path.join(p, "Slide_*", "*.png")
        files = glob(pattern)
        for f in files:
            img = io.imread(f)[:, :, 0:3]
            gt_f = f.replace("Slide", "GT")
            gt = io.imread(gt_f)
            name = os.path.basename(f).split(".")[0]
            yield img, gt, name.split("_")[0], name.split("_")[1]
    elif datatype == "consep":
        files = glob(os.path.join(p, "Images", "*.png"))
        for f in files:
            img = io.imread(f)[:, :, 0:3]
            lbl = scipy.io.loadmat(
                f.replace("Images", "Labels").replace(".png", ".mat")
            )
            gt = lbl["inst_map"]
            gt_type = lbl["type_map"]
            basename = f.split("/")[-1].split(".")[0]
            yield img, gt, basename, gt_type


consep_mapping = {
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 5,
    6: 5,
    7: 5,
}

tnbc_mapping = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
}


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--folder",
        dest="folder",
        type="string",
        help="where to find the TNBC cell type dataset",
    )
    parser.add_option(
        "--type",
        dest="type",
        type="string",
        help="data type name",
        default="tnbc",
    )
    parser.add_option(
        "--marge",
        dest="marge",
        type="int",
        help="how much to reduce the image size",
    )
    parser.add_option(
        "--cell_marge",
        dest="cell_marge",
        type="int",
        help="how much to reduce the image size",
    )
    parser.add_option(
        "--cell_resize",
        dest="cell_resize",
        type="int",
        help="how much to reduce the image size",
    )
    parser.add_option(
        "--out_path",
        dest="out_path",
        type="str",
        help="output path name",
    )
    parser.add_option(
        "--name",
        dest="name",
        type="str",
        help="output name for the cell table",
    )
    parser.add_option(
        "--n_jobs",
        dest="n_jobs",
        type="int",
        default=8,
        help="Number of jobs",
    )

    (options, args) = parser.parse_args()

    n_jobs = int(options.n_jobs)
    table_list = []
    cell_list = []
    last_index = 0
    if options.type == "tnbc":
        n = len(glob(os.path.join(options.folder, "Slide_*", "*.png")))
    else:
        n = len(glob(os.path.join(options.folder, "Images", "*.png")))
    for rgb_, bin_, name, patch in tqdm(
        gene_data(options.folder, options.type), total=n
    ):
        list_f[-2].set_shift((-options.marge, -options.marge))
        table, cells = bin_extractor(
            rgb_,
            bin_,
            list_f,
            options.marge,
            cell_resize=options.cell_resize,
            cell_marge=options.cell_marge,
            pandas_table=True,
            n_jobs=n_jobs,
            cellclass_map=patch,
        )
        table["name"] = name
        if options.type == "tnbc":
            table["patch"] = patch
        else:
            table["patch"] = 0
        if table is not None:
            cell_list.append(cells)
            n = table.shape[0]
            table["index"] = range(last_index, n + last_index)
            table.set_index(["index"], inplace=True)

            last_index += n
            table_list.append(table)

    res = pd.concat(table_list, axis=0)
    res = res[(res.T != 0).any()]  # drop rows where that are only 0! :)
    all_cells = np.vstack(cell_list)
    all_cells = all_cells[res.index]

    if options.type == "tnbc":
        # drop necrosis and myoepithelial
        res["name"] = res["name"].astype(int)
        res = res[~(res["Label"].isin([8, 11]))]
        res["Label"] = res.apply(lambda x: tnbc_mapping[x["Label"]], axis=1)
        idx_train = res[~res["name"].isin([1, 9, 14])].index
        idx_test = res[res["name"].isin([1, 9, 14])].index
        res.loc[idx_train, "fold"] = "train"
        res.loc[idx_test, "fold"] = "test"
        all_cells = all_cells[res.index]
    elif options.type == "consep":
        label = options.name.split("_")[-1]
        res["fold"] = label
        # 4 classes like consep paper
        res["Label"] = res.apply(lambda x: consep_mapping[x["Label"]], axis=1)

    unique_label = res["Label"].unique()
    n_map = create_mapping(unique_label)
    res["orderedLabel"] = res.apply(lambda row: n_map[row["Label"]], axis=1)

    check_or_create(options.out_path)
    res.to_csv(os.path.join(options.out_path, options.name + ".csv"))
    fname = os.path.join(options.out_path, options.name + "_tinycells.npy")
    np.save(fname, all_cells)
