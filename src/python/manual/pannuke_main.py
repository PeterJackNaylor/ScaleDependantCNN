import os
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import pandas as pd
from skimage.morphology import dilation, erosion, square

from extractor.extract_nuclei import bin_extractor

from main import list_f


def create_mapping(lbl):
    old_idx = list(lbl[np.argsort(lbl)])
    return {old_idx[i]: i for i in range(len(old_idx))}


def generate_wsl(labelled_mat):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    Args:
        labelled_mat: 2-D labelled matrix.
    Returns:
        a 2-D labelled matrix where each integer component
        cooresponds to a seperation between two objects.
        0 refers to the backrgound.
    """
    se_3 = square(3)
    ero = labelled_mat.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se_3)
    ero[labelled_mat == 0] = 0

    grad = dilation(labelled_mat, se_3) - ero
    grad[labelled_mat == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad


def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    Parameters
    ----------
    path: string, path for the creation of the folder to check/create
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def gene_data(p, datatype="pannuke"):
    if datatype == "pannuke":
        for fold in ["fold1", "fold2", "fold3"]:
            rgbs = np.load(os.path.join(p, "images", fold, "images.npy"))
            masks = np.load(os.path.join(p, "masks", fold, "masks.npy"))
            names = np.load(os.path.join(p, "images", fold, "types.npy"))
            for i in range(rgbs.shape[0]):
                img = rgbs[i].astype("uint8")
                name = names[i]
                gt = masks[i].astype("uint8")
                all_int = []
                matching = {0: 0}
                for i in range(5):
                    elements = list(np.unique(gt[:, :, i]))
                    elements.remove(0)
                    all_int += elements
                    for el in elements:
                        matching[el] = i + 1
                new_gt = np.sum(gt[:, :, :5], axis=2)
                all_el = list(np.unique(new_gt))
                all_el.remove(0)
                overlap = [el for el in all_el if el not in all_int]
                for el in overlap:
                    new_gt[new_gt == el] = 0
                lines = generate_wsl(new_gt)
                new_gt[lines > 0] = 0
                g = np.vectorize(lambda x: matching[x])
                new_gt = g(new_gt)

                yield img, new_gt, name, fold


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
    mask_list = []
    last_index = 0
    n = 2656 + 2523 + 2722
    for rgb_, bin_, name, patch in tqdm(
        gene_data(options.folder, options.type), total=n
    ):
        list_f[-2].set_shift((-options.marge, -options.marge))
        table, cells, mask = bin_extractor(
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
        table["patch"] = patch
        if patch == "fold1" or patch == "fold2":
            table["fold"] = "train"
        elif patch == "fold3":
            table["fold"] = "test"

        if table is not None:
            cell_list.append(cells)
            mask_list.append(mask)
            n = table.shape[0]
            table["index"] = range(last_index, n + last_index)
            table.set_index(["index"], inplace=True)

            last_index += n
            table_list.append(table)

    res = pd.concat(table_list, axis=0)
    res = res[(res.T != 0).any()]  # drop rows where that are only 0! :)

    unique_label = res["Label"].unique()
    n_map = create_mapping(unique_label)
    res["orderedLabel"] = res.apply(lambda row: n_map[row["Label"]], axis=1)

    check_or_create(options.out_path)
    res.to_csv(os.path.join(options.out_path, options.name + ".csv"))
    all_cells = np.vstack(cell_list)
    all_masks = np.vstack(mask_list)
    fname = os.path.join(options.out_path, options.name + "_tinycells.npy")
    np.save(fname, all_cells)
    fname = os.path.join(options.out_path, options.name + "_tinymasks.npy")
    np.save(fname, all_masks)
