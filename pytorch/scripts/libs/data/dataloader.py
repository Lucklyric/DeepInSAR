import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob 

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

class SimInSARDB(Dataset):
    def __init__(self, patch_size, sim_db_root_dir, fake_length, db_width):
        self.all_noisy_ifg_paths = glob.glob('{}/**/*.noisy'.format(sim_db_root_dir), recursive=True)
        self.all_noisy_ifg_paths.sort()
        self.length = fake_length
        self.patch_size = patch_size
        self.db_width = db_width

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % len(self.all_noisy_ifg_paths)
        
        noisy_path, clean_path, slc1_path, slc2_path, coh_path = self.parse_path(self.all_noisy_ifg_paths[idx])

        ifg, rs, cs, h = readFloatComplexRandomPathces(noisy_path, width=self.db_width, patch_size=self.patch_size, num_sample=1)
        slc1, rs, cs, h = readFloatComplexRandomPathces(slc1_path, width=self.db_width, patch_size=self.patch_size, num_sample=1, rows=rs, cols=cs, height=h)
        slc2, rs, cs, h = readFloatComplexRandomPathces(slc2_path, width=self.db_width, patch_size=self.patch_size, num_sample=1, rows=rs, cols=cs, height=h)
        filt, rs, cs, h = readFloatComplexRandomPathces(clean_path, width=self.db_width, patch_size=self.patch_size, num_sample=1, rows=rs, cols=cs, height=h)
        coh, rs, cs, h = readFloatRandomPathces(coh_path, width=self.db_width, patch_size=self.patch_size, num_sample=1, rows=rs, cols=cs, height=h)

        return {'ifg': np.asarray(ifg), 'slc1': np.asarray(slc1), 'slc2': np.asarray(slc2), 'clean': np.asarray(filt), 'coh': np.asarray(coh)}

    def parse_path(self, noisy_path):
        noisy_path_tokens = noisy_path.split('/')
        pair_name = noisy_path_tokens[-1].split('.')[0]

        clean_path = "/{}/{}.filt".format(os.path.join(*(noisy_path_tokens[:-1])), pair_name)
        coh_path = clean_path + '.coh'

        slc1_name = pair_name.split('_')[0]
        slc2_name = pair_name.split('_')[1]

        slc1_path = "/{}/{}.rslc.bar.norm".format(os.path.join(*(noisy_path_tokens[:-1])), slc1_name)
        slc2_path = "/{}/{}.rslc.bar.norm".format(os.path.join(*(noisy_path_tokens[:-1])), slc2_name)

        assert os.path.isfile(slc1_path), "slc1 not exisit"
        assert os.path.isfile(slc2_path), "slc1 not exisit"
        assert os.path.isfile(noisy_path), "noisy not exisit"
        assert os.path.isfile(clean_path), "clean not exisit"
        assert os.path.isfile(coh_path), "coh not exisit"

        return noisy_path, clean_path, slc1_path, slc2_path, coh_path




