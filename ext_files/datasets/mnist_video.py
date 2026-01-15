### Author Silvia. L. Pintea
# Dataset class for reading "Moving5" raw video data (based on MNIST digists).

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import scipy
import os
import numpy as np
import pickle
import copy
import struct


class MnistVideo:
    def __init__(
        self,
        args: Any,
        set_name: str,
    ):
        self._args = args
        self._set_name = set_name
        self._avg_cls = 5

        # Path to files
        self._datapath = args.datapath
        alldataidx = self.list_data(args)

        # Pick only the set
        self._dataidx = self.pick_set(set_name, alldataidx)

    def list_data(self, args: Any):
        datafile = open(args.datapath, "rb")

        self._data, self._labels = pickle.load(datafile)
        alldataidx = np.arange(len(self._data))
        return alldataidx

    def pick_set(self, setname, dataidx: List[int]):
        np.random.seed(0)
        shuffidx = copy.deepcopy(dataidx)
        np.random.shuffle(shuffidx)

        # Split into val and train and test
        if setname.startswith("train"):
            return (shuffidx[0 : len(shuffidx) // 3]).tolist()
        elif setname.startswith("val"):
            return (shuffidx[len(shuffidx) // 3 : 2 * len(shuffidx) // 3]).tolist()
        if setname.startswith("test"):
            return (shuffidx[2 * len(shuffidx) // 3 : len(shuffidx)]).tolist()

    def read_one_data(self, idx: int):
        prefeat = self._data[self._dataidx[idx]]
        feat = prefeat.reshape(prefeat.shape[0], -1)
        cls = self._labels[self._dataidx[idx]]
        indexes = np.unique(cls, return_index=True)[1]
        subactions = [cls[index] for index in sorted(indexes)]
        gt = np.array([np.where(np.array(subactions) == gs)[0][0] for gs in cls])
        return feat, gt, subactions

    def __len__(self):
        return len(self._dataidx)

    def __getitem__(self, idx: int):
        spls, label, label_names = self.read_one_data(idx)
        spls = spls.reshape(spls.shape[0], -1)
        spls /= (
            np.linalg.norm(spls, axis=1, keepdims=True).repeat(spls.shape[1], axis=1)
            + 1e-7
        )

        video = {
            "samples": spls,
            "labels": np.array(label).astype(int),
            "idx": idx,
            "name": "video" + str(idx),
            "label_names": label_names,
            "avg_cls": self._avg_cls,
        }
        return video
