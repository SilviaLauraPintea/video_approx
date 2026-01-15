### Author Silvia. L. Pintea
# Dataset class for reading "Desktop Assembly" 512D features using the features provided in:
# Kumar, Sateesh, et al. "Unsupervised action segmentation by joint representation learning and online clustering."
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
# https://github.com/trquhuytin/TOT-CVPR22

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import scipy
import os
import numpy as np
import pickle
import copy
import struct
import cv2 as cv


class DAVideo:
    def __init__(
        self,
        args: Any,
        set_name: str,
    ):
        self._args = args
        self._ext = ".npy"
        self._datapath = args.datapath
        self._set_name = set_name
        self._ignore = []
        self._avg_cls = 23
        # Path to files
        self._datalist = self.list_data()

    def list_data(self):
        fpath = os.path.join(self._datapath, "features")
        files = [
            f
            for f in os.listdir(fpath)
            if os.path.isfile(os.path.join(fpath, f)) and (self._ext in f)
        ]
        files.sort()
        return files

    def __len__(self):
        return len(self._datalist)

    def read_one_data(self, idx):
        fpath = os.path.join(self._datapath, "features")
        features = np.load(open(os.path.join(fpath, self._datalist[idx]), "rb"))

        gtpath = os.path.join(self._datapath, "groundTruth")
        gt_str = []
        gt_file = open(os.path.join(gtpath, self._datalist[idx][0:-4]), "r")
        for line in gt_file:
            gt_str.append(line.strip())
        gt_str, feat = self.ignore_class(gt_str, features)

        subactions = np.unique(gt_str)
        gt = [np.where(np.array(subactions) == gs)[0][0] for gs in gt_str]

        assert len(gt) == feat.shape[0]
        return feat, np.array(gt).astype(int), subactions

    def ignore_class(self, labels, feats):
        if len(self._ignore):
            for icls in self._ignore:
                ignore = np.where(np.array(labels) == icls)[0].tolist()
                ignore_labels = np.delete(np.array(labels), ignore).tolist()
                ignore_feats = np.delete(feats, ignore, axis=0)
                labels = ignore_labels
                feats = ignore_feats
            return ignore_labels, ignore_feats
        else:
            return labels, feats

    def __getitem__(self, idx: int):
        spl, label, label_names = self.read_one_data(idx)
        spls = spl.reshape(spl.shape[0], spl.shape[-1])
        video = {
            "samples": spls,
            "labels": label,
            "label_names": label_names,
            "idx": idx,
            "name": self._datalist[idx],
            "avg_cls": self._avg_cls,
        }
        return video
