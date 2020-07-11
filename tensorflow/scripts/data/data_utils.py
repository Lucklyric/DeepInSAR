import numpy as np
import json
import glob
import re
import matplotlib.pyplot as plt
import os

class InSARDB(object):
    def __init__(self, name, noisy_path, noisy_ext, rslc_path, rslc_ext, filt_path, filt_ext, coh_path, coh_ext, width):
        self.name = name
        self.noisy_files = glob.glob("%s/*%s" % (noisy_path, noisy_ext))
        self.noisy_files.sort()
        self.filt_files = glob.glob("%s/*%s" % (filt_path, filt_ext))
        self.filt_files.sort()
        self.coh_files = glob.glob("%s/*%s" % (coh_path, coh_ext))
        self.coh_files.sort()
        self.rslc_path = rslc_path
        self.rslc_ext = rslc_ext
        self.width = width
        assert len(self.noisy_files) > 0, "length of noisy_files is 0"
        assert len(self.filt_files) > 0, "length of filt_files is 0"
        assert len(self.coh_files) > 0, "length of coh_files is 0"
        assert len(self.noisy_files) == len(self.filt_files) == len(
            self.coh_files), "length of files different => noisy %d filt %d, coh %d" % (len(self.noisy_files), len(self.filt_files), len(self.coh_files))

    def get_rslcs_paths(self, ifg_file_path):
        file_name = ifg_file_path.split("/")[-1]
        tokens = re.split("_|\.", file_name)
        slc1_path = "%s/%s%s" % (self.rslc_path, tokens[0], self.rslc_ext)
        slc2_path = "%s/%s%s" % (self.rslc_path, tokens[1], self.rslc_ext)
        return slc1_path, slc2_path


def parse_json_db(file):
    dbs = []
    with open(file) as f:
        json_data = json.load(f)
        for db in json_data:
            dbs.append(
                InSARDB(
                    name=db["name"],
                    noisy_path=db["noisy_path"],
                    noisy_ext=db["noisy_ext"],
                    filt_path=db["filt_path"],
                    filt_ext=db["filt_ext"],
                    coh_path=db["coh_path"],
                    coh_ext=db["coh_ext"],
                    rslc_path=db["rslc_path"],
                    rslc_ext=db["rslc_ext"],
                    width=db["width"]))
    return dbs

def readShortComplex(fileName, width=1):
    return np.fromfile(fileName, '>i2').astype(np.float).view(np.complex).reshape(-1, width)


def readFloatComplex(fileName, width=1):
    return np.fromfile(fileName, '>c8').astype(np.complex).reshape(-1, width)


def readFloat(fileName, width=1):
    return np.fromfile(fileName, '>f4').astype(np.float).reshape(-1, width)


def writeShortComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.copy().view(np.float).astype('>i2').tofile(out_file)
    out_file.close()


def writeFloatComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>c8').tofile(out_file)
    out_file.close()


def writeFloat(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>f4').tofile(out_file)
    out_file.close()


def readFloatComplexRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            size_of_file = os.path.getsize(fileName)
            height = size_of_file / 8 / width
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(8 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(8 * patch_size), dtype=">c8").astype(np.complex))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height

def readShortFloatComplexRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            size_of_file = os.path.getsize(fileName)
            # print(size_of_file)
            height = size_of_file / 4 / width
            # print(height)
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(4 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(4 * patch_size), dtype=">i2").astype(np.float).view(np.complex))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height

def readFloatRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
            size_of_file = os.path.getsize(fileName)
            height = size_of_file / 4 / width
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(4 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(4 * patch_size), dtype=">f4").astype(np.float))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height

def ap2one(ifg):
    ifg_phase = np.angle(ifg)
    ifg = 1. * np.exp(1j * ifg_phase)
    return ifg, np.real(ifg), np.imag(ifg)

def normalize_slc_by_tanhmz(img, norm=False):
    phase = np.angle(img)
    points = img
    a = np.abs(points)
    shape = a.shape
    a = a.flatten()
    # a = a**0.15
    mad = np.median(np.abs(a - np.median(a)))
    mz = 0.6745 * ((a - np.median(a)) / mad)
    mz = (np.tanh(mz / 7) + 1) / 2
    if norm:
        mz = (mz - mz.min()) / (mz.max() - mz.min())
    mz = mz.reshape(shape)
    return mz * np.exp(1j * phase)


# #usage
# Z_processed = saturate_outlier(z)
def normalize_under_folder(path, width=1000, norm=False):
    slc_names = glob.glob(path)
    slc_names.sort()
    print("total %d" % (len(slc_names)))
    i = 1
    for slc_name in slc_names:
        slc = readShortComplex(slc_name, width=width)
        slc = normalize_slc_by_tanhmz(slc, norm=norm)
        writeFloatComplex(slc_name + ".bar.norm", slc.flatten())
        print("Finhsed %d" % (i))
        i += 1


def visualizaion_under_folder(path, width, title, xlabel, ylabel):
    slc_names = glob.glob(path)
    slc_names.sort()
    print("total %d" % (len(slc_names)))
    i = 1
    for slc_name in slc_names:
        slc = readShortComplex(slc_name, width=width)
        if width > 1000:
            slc = slc[250:1250, 250:1250]
        plt.figure()
        plt.hist(np.abs(slc).flatten(), 260, log=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(slc_name + ".hist.png")
        plt.close()
        plt.figure()
        plt.imsave(slc_name + ".png", np.log(np.abs(slc)), cmap="gray")
        plt.close()
        nslc = normalize_slc_by_tanhmz(slc,True)
        plt.figure()
        plt.hist(np.abs(nslc).flatten(), 260, range=[0, 1])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(slc_name + ".bar.norm.hist.png")
        plt.close()
        plt.figure()
        plt.imsave(slc_name + ".bar.png", np.abs(nslc), cmap="gray")
        plt.close()
        print("Finhsed %d" % (i))
        i += 1
