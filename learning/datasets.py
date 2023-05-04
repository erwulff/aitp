import torch
import numpy as np
from pathlib import Path
import h5py
from math import floor, ceil
from tqdm import tqdm

from torch.utils.data import Dataset, IterableDataset, DistributedSampler
from torch.distributed import get_rank, get_world_size


class MetaJediDataset(Dataset):
    CLASS_LABELS = ["gluon", "light quark", "W", "Z", "top"]
    FILE_SIZE = 10000

    def __init__(self, data_dir, train, size=None, transform=None):
        self.size = size
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.size


class JEDIIterableDataset(IterableDataset):
    """
    This dataset is based on the torch.utils.data.IterableDataset class and is
    suitable for distributed training and/or multi-process data loading.
    """
    CLASS_LABELS = ["gluon", "light quark", "W", "Z", "top"]
    FILE_SIZE = 10000
    def __init__(self, data_dir, train, size=None, transform=None):
        self.size = size
        self.transform = transform

        if train:
            self.split_dir = Path(data_dir) / "train"
            max_size = len(list(self.split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 630000
            else:
                assert self.size <= max_size, "max_size is {}, self.size is {}".format(max_size, self.size)
        else:
            self.split_dir = Path(data_dir) / "val"
            max_size = len(list(self.split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 260000
            else:
                assert self.size <= max_size, "max_size is {}, self.size is {}".format(max_size, self.size)

        self.shards = list(self.split_dir.glob("*JEDI*"))  # shard filenames
        # self.shards = list(Path("/mnt/ceph/users/ewulff/data/clic_pt").glob("samples*.pt"))  # shard filenames


    def _convert_to_pt(self, save_dir):
        for i_shard, shard in enumerate(tqdm(self.shards, total=len(self.shards))):
            samples = []
            with h5py.File(str(shard), mode="r") as loaded_shard:
                for i_sample in range(len(loaded_shard['X'])):
                    sample = torch.tensor(loaded_shard['X'][i_sample]), torch.tensor(loaded_shard['Y'][i_sample])
                    samples.append(sample)
            torch.save(samples, Path(save_dir) / f"samples{i_shard}.pt")


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        num_workers = worker_info.num_workers if worker_info is not None else 1  # cpus feeding data to gpu
        worker_id = worker_info.id if worker_info is not None else 0  # cpu worker id
        world_size = get_world_size()  # number of gpus
        process_rank = get_rank()  # gpu worker rank

        sampler_rank = process_rank * num_workers + worker_id
        # sampler distributes shard indices over gpu and cpu workers
        sampler = DistributedSampler(self.shards,
            num_replicas=(num_workers * world_size),
            rank=sampler_rank,
            shuffle=False,
        )
        print(f"sampler of sampler_rank {sampler_rank} is of length {len(sampler)}")

        for i_shard in iter(sampler):
            shard = self.shards[i_shard]
            with h5py.File(str(shard), mode="r") as loaded_shard:
                for i_sample in range(len(loaded_shard['X'])):
                    sample = torch.tensor(loaded_shard['X'][i_sample]), torch.tensor(loaded_shard['Y'][i_sample])
                    if self.transform:
                        sample = self.transform(sample)
                    yield sample
            # samples = torch.load(shard)
            # for i_sample in range(len(samples)):
            #     yield samples[i_sample]

    def __len__(self):
        return self.size

class JEDIRAMDataset(MetaJediDataset):
    """
    This dataset class loads the entire dataset into memory. This
    can result in long load times and can be a problem if your machine
    doesn't have a lot of memory.
    """

    def __init__(self, data_dir, train, size=None, transform=None):
        super().__init__(data_dir, train=train, size=size, transform=transform)

        if train:
            split_dir = Path(data_dir) / "train"
            max_size = len(list(split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 630000
            else:
                assert self.size <= max_size, "max_size is {}, self.size is {}".format(max_size, self.size)
        else:
            split_dir = Path(data_dir) / "val"
            max_size = len(list(split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 260000
            else:
                assert self.size <= max_size, "max_size is {}, self.size is {}".format(max_size, self.size)

        data_list = []
        label_list = []
        for file_path in split_dir.glob("jetImage_*_150p_*_*0000_JEDI.h5"):
            f = h5py.File(file_path, mode="r")
            data_list.append(np.array(f.get("X")))
            label_list.append(np.array(f.get("Y")))

        self.X = torch.tensor(np.concatenate(data_list, axis=0))
        self.Y = torch.tensor(np.concatenate(label_list, axis=0))

    def __getitem__(self, index):
        sample = self.X[index, ::], self.Y[index, ::]
        if self.transform:
            sample = self.transform(sample)
        return sample


class JEDIDataset(MetaJediDataset):
    """
    This dataset class uses the file structure in the JEDI-net data
    in order to only load the necessary files when they are needed.
    This is useful if your machine can't load the entire dataset into
    memory at once.
    """

    JET_IMAGE_SIZE = 90000

    def __init__(self, data_dir, train, size=None, transform=None):
        super().__init__(data_dir, train=train, size=size, transform=transform)

        self.train = train
        self.size = size

        if self.train:
            self.split_dir = Path(data_dir) / "train"
            max_size = len(list(self.split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 630000
            else:
                assert self.size <= max_size, "max_size is {}, self.size is {}".format(max_size, self.size)
        else:
            self.split_dir = Path(data_dir) / "val"
            max_size = len(list(self.split_dir.glob("*JEDI*"))) * self.FILE_SIZE
            if self.size == None:
                self.size = 260000
            else:
                assert self.size <= max_size, "max_size is {}, self.size is {}".format(max_size, self.size)

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
        assert file_name in [str(path.name) for path in self.split_dir.glob("*JEDI*")], print(
            file_name, i_a, i_jet_image, i_b
        )
        file = h5py.File(str(self.split_dir / file_name))
        sample = torch.tensor(file["X"][i_file]), torch.tensor(file["Y"][i_file])
        if self.transform:
            sample = self.transform(sample)
        return sample


class TinyJEDIDataset(MetaJediDataset):
    """
    This dataset class loads a samll fraction of the total JEDI-net dataset.
    """

    def __init__(self, data_dir, train, size=None, transform=None):
        super().__init__(data_dir, train=train, size=size, transform=transform)

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
        sample = torch.tensor(file["X"][index]), torch.tensor(file["Y"][index])
        if self.transform:
            sample = self.transform(sample)
        return sample
