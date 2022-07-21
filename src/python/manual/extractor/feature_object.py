import numpy as np

from skimage.color import rgb2gray
from skimage.io import imsave
from skimage.morphology import disk, opening
from skimage.feature import local_binary_pattern


class Feature(object):
    """
    Generic python object for feature extraction from
    a binary image.
    You can also override the name attribute.
    Parameters
    ----------
    name: string, name of the feature extractor
    n_extensions: integer, size of dilation to
        perform to the object prior to extraction
    -------
    """

    def __init__(self, name, n_extension):
        self.name = name
        self.n_extension = n_extension
        self.size = 0
        self.GetSize()

    def _return_size(self):
        return self.size

    def _return_name(self):
        return self.name

    def _return_n_extension(self):
        return self.n_extension

    def _apply_region(self, RGB, BIN, cell):
        raise NotImplementedError

    def GetSize(self):
        raise NotImplementedError


def Pixel_size(img):
    img[img > 0] = 1
    return np.sum(img)


def Mean_intensity(RGB, BIN):
    gray_rgb = rgb2gray(RGB)
    BIN = BIN > 0
    return np.mean(gray_rgb[BIN])


def Std_intensity(RGB, BIN):
    gray_rgb = rgb2gray(RGB)
    BIN = BIN > 0
    return np.std(gray_rgb[BIN])


def OutSideBBandBin(regionp, RGB, marge):
    max_x, max_y = RGB.shape[0:2]
    x_m, y_m, x_M, y_M = regionp.bbox
    shift_xm = marge - x_m if x_m - marge < 0 else 0
    shift_xM = (x_M + marge) - max_x if x_M + marge > max_x else 0
    shift_ym = marge - y_m if y_m - marge < 0 else 0
    shift_yM = (y_M + marge) - max_y if y_M + marge > max_y else 0
    x_m = max(x_m - marge, 0)
    x_M = min(x_M + marge, max_x)
    y_m = max(y_m - marge, 0)
    y_M = min(y_M + marge, max_y)

    bin = regionp.image.astype(np.uint8)
    bin_x, bin_y = bin.shape
    shape_bin = (x_M - x_m, y_M - y_m)
    bin_res = np.zeros(shape=shape_bin, dtype="uint8")

    if marge - shift_xM == 0 and marge - shift_yM == 0:
        bin_res[(marge - shift_xm):, (marge - shift_ym):] = bin
    elif marge - shift_xM == 0:
        bin_res[
            (marge - shift_xm):,
            (marge - shift_ym):-(marge - shift_yM),
        ] = bin
    elif marge - shift_yM == 0:
        bin_res[
            (marge - shift_xm):-(marge - shift_xM),
            (marge - shift_ym):,
        ] = bin
    else:
        bin_res[
            (marge - shift_xm):-(marge - shift_xM),
            (marge - shift_ym):-(marge - shift_yM),
        ] = bin
    inv_bin = 1 - bin_res
    return x_m, x_M, y_m, y_M, inv_bin


class PixelSize(Feature):
    def _apply_region(self, RGB, BIN, cell, BIN_L):
        return Pixel_size(BIN)

    def GetSize(self):
        self.size = 1


class MeanIntensity(Feature):
    def _apply_region(self, RGB, BIN, cell, BIN_L):
        return Mean_intensity(RGB, BIN)

    def GetSize(self):
        self.size = 1


class ChannelMeanIntensity(Feature):
    def _apply_region(self, RGB, BIN, cell, BIN_L):
        bin_b = BIN > 0
        val = []
        for c in range(RGB.shape[2]):
            cha_crop = RGB[:, :, c]
            value = np.mean(cha_crop[bin_b]) / 255
            val.append(value)
        return val

    def GetSize(self):
        self.size = 3


class StdIntensity(Feature):
    def _apply_region(self, RGB, BIN, cell, BIN_L):
        return Std_intensity(RGB, BIN)

    def GetSize(self):
        self.size = 1


class ChannelStdIntensity(Feature):
    def _apply_region(self, RGB, BIN, cell, BIN_L):
        bin_c = BIN > 0
        val = []
        for c in range(RGB.shape[2]):
            cha_crop = RGB[:, :, c]
            value = np.std(cha_crop[bin_c]) / 255
            val.append(value)
        return val

    def GetSize(self):
        self.size = 3


class LBP(Feature):
    def __init__(
        self,
        name,
        n_extension,
        radius=[1, 3],  # ,3,5],
        n_points=None,
        methods=["ror"],  # "default", 'uniform', 'nri_uniform', 'var'],
        quantiles=[el for el in range(10, 100, 10)],
    ):
        self.name = []
        for r in radius:
            for m in methods:
                for q in quantiles:
                    self.name.append(
                        "{}__m_{}__r_{}__q_{}__d__{}".format(
                            name,
                            m,
                            r,
                            q,
                            n_extension,
                        )
                    )
        self.n_extension = n_extension
        self.size = len(radius) * len(methods) * len(quantiles)
        self.radius = radius
        self.n_points = [6 * r for r in radius] if not n_points else n_points
        self.methods = methods
        self.quantiles = quantiles

    def _apply_region(self, RGB, BIN, cell, BIN_L):
        BIN_c = BIN > 0
        tmp = rgb2gray(RGB)
        output = np.zeros(self.size)
        ind = 0
        for r, n_points in zip(self.radius, self.n_points):
            for m in self.methods:
                lbp = local_binary_pattern(tmp, n_points, r, m)
                for q in self.quantiles:
                    output[ind] = np.percentile(lbp[BIN_c], q)
                    ind += 1

        return output


class Granulometri(Feature):
    def __init__(self, name, n_extension, sizes=[1, 2, 5, 7]):
        self.name = name
        self.n_extension = n_extension
        self.size = len(sizes)
        self.sizes = sizes
        self.GetSize(sizes)

    def GetSize(self, list_):
        self.size = len(list_)

    def _apply_region(self, RGB, BIN, cell, BIN_L):
        tmp = rgb2gray(RGB)
        BINb = BIN > 0
        sum_total = np.sum(tmp[BINb])
        values = np.zeros(shape=self.size)
        for i, s in enumerate(self.sizes):
            se = disk(s)
            opn_img = opening(tmp, se)
            diff_img = tmp - opn_img
            values[i] = np.sum(diff_img[BINb]) / sum_total
            tmp = opn_img
        return values


class Circularity(Feature):
    def _apply_region(self, RGB, BIN, cell, BIN_L):
        circularity = (cell.perimeter ** 2) / (4 * np.pi * cell.area)
        return circularity

    def GetSize(self):
        self.size = 1


class Elongation(Feature):
    def _apply_region(self, RGB, BIN, cell, BIN_L):
        if cell.major_axis_length != 0:
            elongation = cell.minor_axis_length / cell.major_axis_length
            return elongation
        else:
            return 0

    def GetSize(self):
        self.size = 1


class Centroid(Feature):
    def __init__(self, name, n_extension):
        self.name = name
        self.n_extension = n_extension
        self.size = 0
        self.shift_value = (0, 0)
        self.GetSize()

    def _apply_region(self, RGB, BIN, cell, BIN_L):
        centers = cell.centroid
        bbox = cell.bbox
        h, w, z = RGB.shape
        coordinates = (
            centers[1] + self.shift(0),
            centers[0] + self.shift(1),
            w,
            h,
            bbox[1] + self.shift(0),
            bbox[0] + self.shift(1),
            bbox[3] + self.shift(0),
            bbox[2] + self.shift(1),
        )
        return coordinates

    def GetSize(self):
        self.size = 8

    def set_shift(self, couple):
        self.shift_value = couple

    def shift(self, integer):
        return self.shift_value[integer]


class Label(Feature):
    def __init__(self, name, n_extension):
        self.name = name
        self.n_extension = n_extension
        self.GetSize()
        return

    def _apply_region(self, RGB, BIN, cell, BIN_L):
        x, y = cell.coords[0]
        label = BIN_L[int(x), int(y)]
        if label == 0:
            import pdb

            pdb.set_trace()
        return label

    def GetSize(self):
        self.size = 1


class Save(Feature):
    """Not usable yet, need to figure out a way for the diff name"""

    def _apply_region(self, RGB, BIN, cell):
        bin = cell.image.astype(np.uint8)
        x_m, y_m, x_M, y_M = cell.bbox
        img_crop = RGB[x_m:x_M, y_m:y_M]
        imsave("test_rgb.png", img_crop)
        imsave("test.png", bin)

    def GetSize(self):
        self.size = 0
