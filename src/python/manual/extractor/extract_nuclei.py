import numpy as np
import pandas as pd

from skimage.morphology import dilation, disk
from skimage.segmentation import watershed
from skimage.transform import resize
from skimage.measure import regionprops, label
from tqdm import tqdm

# from joblib import Parallel, delayed


def get_names(feat_list):
    """
    feat_list: list of feature defined in 'feature_object'
    Returns list of the feature names.
    """
    names = []
    for el in feat_list:
        if el.size != 1:
            for it in range(el.size):
                names.append(el._return_name()[it])
        else:
            names.append(el._return_name())
    return names


def needed_grown_region(list_feature):
    """
    Looks if any of the features needs a
    specific growing of the objects by dilation.
    """
    res = []
    for feat in list_feature:
        if feat._return_n_extension() not in res:
            res += [feat._return_n_extension()]
    return res


def needed_grown_region_dic(list_feature):
    """
    Looks if any of the features needs
    a specific growing of the objects by dilation.
    """
    res = {}
    for feat in list_feature:
        res_apply = feat._return_n_extension()
        if res_apply not in res.keys():
            res[res_apply] = [feat]
        else:
            res[res_apply] += [feat]
    return res


def grow_region(bin_image, n_pix):
    """
    Grows a region to fix size.
    """
    op = disk(n_pix)
    dilated_mask = dilation(bin_image, footprint=op)
    return watershed(dilated_mask, bin_image, mask=dilated_mask)


def dilate(bin, n_pix):
    op = disk(n_pix)
    dilated_mask = dilation(bin, footprint=op)
    return dilated_mask


def check_within_margin(rgb_image, marge, cell_prop):
    max_x, max_y = rgb_image.shape[0:2]
    x_c, y_c = cell_prop.centroid
    x_bounds = marge < x_c and x_c < max_x - marge
    y_bounds = marge < y_c and y_c < max_y - marge
    if x_bounds and y_bounds:
        return True
    else:
        return False


def get_crop(rgb_image, bin_image, cell_prop, d=0, dilate_bin=True):
    # d is the dilation resolution to pad to the image
    x_m, y_m, x_M, y_M = cell_prop.bbox
    r_rgb = rgb_image[(x_m - d):(x_M + d), (y_m - d):(y_M + d)]
    r_bin = bin_image[(x_m - d):(x_M + d), (y_m - d):(y_M + d)].copy()
    r_bin[r_bin != cell_prop.label] = 0
    r_bin[r_bin == cell_prop.label] = 1
    if d > 0 and dilate_bin:
        r_bin = dilate(r_bin, d)
    return r_rgb, r_bin


def get_names_dic(feat_list):
    """
    feat_list: list of feature defined in 'feature_object'
    Returns list of the feature names.
    """
    names = []
    for dilation_res in feat_list.keys():
        for el in feat_list[dilation_res]:
            if el.size != 1:
                for it in range(el.size):
                    names.append(el._return_name()[it])
            else:
                names.append(el._return_name())
    return names


def analyse_cell(
    cell,
    rgb_image,
    marge,
    p,
    features_grow_region_n,
    bin_image_copy,
    bin_image_label,
):
    c_array = np.zeros(shape=p)
    if check_within_margin(rgb_image, marge, cell):
        offset_all = 0
        for dilation_res in features_grow_region_n.keys():
            rgb_c, bin_c = get_crop(
                rgb_image,
                bin_image_copy,
                cell,
                d=dilation_res,
            )
            for _, feat in enumerate(features_grow_region_n[dilation_res]):
                ot = feat.size  # off_tmp
                c_array[(offset_all):(offset_all + ot)] = feat._apply_region(
                    rgb_c, bin_c, cell, bin_image_label
                )
                offset_all += ot
    return c_array

def compute_pad_width(bin_crop, maxsize):
    x, y = bin_crop.shape
    left0 = maxsize // 2 - x // 2
    right0 = maxsize - (left0 + x)
    left1 = maxsize // 2 - y // 2
    right1 = maxsize - (left1 + y)
    return (left0, right0), (left1, right1)

def bin_extractor(
    rgb_image,
    bin_image,
    list_feature,
    marge=None,
    pandas_table=False,
    do_label=True,
    n_jobs=8,
    cell_resize=32,
    cell_marge=0,
    pad=True,
    cellclass_map="0",
    cellsize=128
):
    if pad:
        rgb_image = np.pad(
            rgb_image,
            pad_width=((marge, marge), (marge, marge), (0, 0)),
            mode="symmetric",
        )
        bin_image = np.pad(
            bin_image,
            pad_width=((marge, marge), (marge, marge)),
            mode="constant",
        )
    bin_image_copy = bin_image.copy()
    if do_label:
        bin_image_copy = label(bin_image_copy)

    p = 0
    for feat in list_feature:
        p += feat.size
    features_grow_region_n = needed_grown_region_dic(list_feature)
    if isinstance(cellclass_map, str):
        cellclass = bin_image.copy()
    else:
        cellclass = cellclass_map
        if pad:
            cellclass = np.pad(
                cellclass,
                pad_width=((marge, marge), (marge, marge)),
                mode="constant",
            )

    def task(cell):
        return analyse_cell(
            cell,
            rgb_image,
            marge,
            p,
            features_grow_region_n,
            bin_image_copy,
            cellclass,
        )

    cell_descriptors = []
    cell_list = regionprops(bin_image_copy)
    # cell_descriptors = Parallel(n_jobs=n_jobs)
    # (delayed(task)(i) for i in cell_list)
    cell_descriptors = [task(cell) for cell in tqdm(cell_list, leave=False)]

    if cell_descriptors:
        cell_matrix = np.stack(cell_descriptors)
        cell_matrix = cell_matrix[cell_matrix.sum(axis=1) != 0]
    else:
        cell_matrix = np.zeros(shape=(0, p))

    if pandas_table:
        names = get_names_dic(features_grow_region_n)
        cell_matrix = pd.DataFrame(cell_matrix, columns=names)

    if cell_matrix.shape[0]:

        def task_resize(c):
            if check_within_margin(rgb_image, marge, c):
                rgb_c, bin_c = get_crop(
                    rgb_image,
                    bin_image_copy,
                    c,
                    d=cell_marge,
                    dilate_bin=False
                )
                r_rgb = resize(
                    rgb_c, (cell_resize, cell_resize, 3), preserve_range=True
                ).astype("uint8")
                pad_width = compute_pad_width(bin_c, cellsize)
                r_bin = np.pad(bin_c, pad_width, constant_values=0)
                return r_rgb, r_bin
            else:
                return None

        # cell_array = Parallel(n_jobs=n_jobs)
        # (delayed(task_resize)(i) for i in cell_list)
        cell_mask_array = [task_resize(i) for i in tqdm(cell_list, leave=False)]
        cell_array = [el[0] for el in cell_mask_array if el is not None]
        cell_mask = [el[1] for el in cell_mask_array if el is not None]
        cell_array = [el for el in cell_array if el is not None]
        cell_array = np.stack(cell_array)
    else:
        cell_array = np.zeros(
            shape=(0, cell_resize, cell_resize, 3),
            dtype="uint8",
        )
    return cell_matrix, cell_array, cell_mask
