import numpy as np
from pathlib import Path
import h5py
from math import floor, ceil

from torch.utils.data import Dataset


class JEDIRAMDataset(Dataset):
    """
    This dataset class loads the entire dataset into memory. This
    can result in long load times and can be a problem if your machine
    doesn't have a lot of memory.
    """

    FILE_SIZE = 10000

    def __init__(self, data_dir, train=True, size=None):
        self.size = size

        if train:
            split_dir = Path(data_dir) / "train"
            max_size = len(list(split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 630000
            else:
                assert self.size <= max_size
        else:
            split_dir = Path(data_dir) / "val"
            max_size = len(list(split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 260000
            else:
                assert self.size <= max_size

        data_list = []
        label_list = []
        for file_path in split_dir.glob("jetImage_*_150p_*_*0000_JEDI.h5"):
            f = h5py.File(file_path, mode="r")
            data_list.append(np.array(f.get("X")))
            label_list.append(np.array(f.get("Y")))

        self.X = np.concatenate(data_list, axis=0)
        self.Y = np.concatenate(label_list, axis=0)

    def __getitem__(self, index):
        return self.X[index, ::], self.Y[index, ::]

    def __len__(self):
        return self.size  # len(self.X)


class JEDIDataset(Dataset):
    """
    This dataset class uses the file structure in the JEDI-net data
    in order to only load the necessary files when they are needed.
    This is useful if your machine can't load the entire dataset into
    memory at once.
    """

    JET_IMAGE_SIZE = 90000
    FILE_SIZE = 10000

    def __init__(self, data_dir, train=True, size=None):
        self.train = train
        self.size = size

        if self.train:
            self.split_dir = Path(data_dir) / "train"
            max_size = len(list(self.split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 630000
            else:
                assert self.size <= max_size
        else:
            self.split_dir = Path(data_dir) / "val"
            max_size = len(list(self.split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 260000
            else:
                assert self.size <= max_size

    def __getitem__(self, index):
        if not self.train:
            index += 630000  # train set size is 630000 so val index needs to start from there

        i_a = index / self.JET_IMAGE_SIZE
        i_jet_image = floor(i_a)
        i_b = index % self.JET_IMAGE_SIZE  # index within the jet image

        i_low = floor(i_b / self.FILE_SIZE) * self.FILE_SIZE
        i_high = ceil(i_b / self.FILE_SIZE) * self.FILE_SIZE
        if i_low == i_high:
            i_high += self.FILE_SIZE  # when there is no remainder ceil()==floor()

        i_file = i_b % self.FILE_SIZE  # index within file

        file_name = "jetImage_{}_150p_{}_{}_JEDI.h5".format(i_jet_image, i_low, i_high)
        assert file_name in [str(path.name) for path in self.split_dir.glob("*JEDI*")], print(file_name)
        file = h5py.File(str(self.split_dir / file_name))

        return np.array(file["X"][i_file]), np.array(file["Y"][i_file])

    def __len__(self):
        return self.size


class TinyJEDIDataset(Dataset):
    """
    This dataset class loads a samll fraction of the total JEDI-net dataset.
    """

    FILE_SIZE = 10000

    def __init__(self, data_dir, train=True, size=None):
        if size is not None:
            assert size <= self.FILE_SIZE, "maximum size of TinyJEDIDataset is {}".format(self.FILE_SIZE)
            self.size = size
        else:
            self.size = self.FILE_SIZE
        self.train = train
        if self.train:
            self.file_name = Path(data_dir) / "train/jetImage_0_150p_0_10000_JEDI.h5"
        else:
            self.file_name = Path(data_dir) / "val/jetImage_7_150p_0_10000_JEDI.h5"

    def __getitem__(self, index):
        file = h5py.File(str(self.file_name))

        return np.array(file["X"][index]), np.array(file["Y"][index])

    def __len__(self):
        return self.size
