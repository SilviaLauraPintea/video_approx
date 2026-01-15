### Author Silvia. L. Pintea
# Dataset class for reading "Hollywood Extended" IDT (improved dense trajectory) features using the features provided in:
# Sarfraz, Saquib, et al. "Temporally-weighted hierarchical clustering for unsupervised action segmentation."
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
# https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import sys
import os
import numpy as np

import struct

np.set_printoptions(threshold=sys.maxsize)


class HEVideo:
    def __init__(
        self,
        args: Any,
        set_name: str,
    ):
        self._args = args
        self._set_name = set_name
        self._ignore_class = []
        self._avg_cls = 3
        # Path to files
        self._datapath = args.datapath
        self._datalist = self.list_data(args)

    def list_data(self, args):
        files = []
        feature_path = os.path.join(self._datapath, "features")
        files = [
            f
            for f in os.listdir(feature_path)
            if os.path.isfile(os.path.join(feature_path, f))
        ]
        files.sort()
        return files

    def read_one_data(self, idx: int):
        feat_path = os.path.join(self._datapath, "features")
        gt_path = os.path.join(self._datapath, "groundTruth")

        # Read the features
        feat = np.loadtxt(os.path.join(feat_path, self._datalist[idx]))
        gt_str = np.loadtxt(os.path.join(gt_path, self._datalist[idx]), dtype="U")
        gt_str, feat = self.ignore_class(gt_str, feat)

        subactions = np.unique(gt_str)
        gt = [int(np.where(np.array(subactions) == gs)[0][0]) for gs in gt_str]

        # Hack: the labels seem to always be a few frames too long/short
        if feat.shape[0] < len(gt):
            diff = len(gt) - feat.shape[0]
            del gt[-diff:]
        elif feat.shape[0] > len(gt):
            diff = feat.shape[0] - len(gt)
            gt = gt + [gt[-1]] * diff

        # Remap the GT to the action numbers
        return feat, np.array(gt).astype(int), np.unique(gt)

    def ignore_class(self, labels, feats):
        if len(self._ignore_class):
            for icls in self._ignore_class:
                ignore = np.where(np.array(labels) == icls)[0].tolist()
                ignore_labels = np.delete(np.array(labels), ignore).tolist()
                ignore_feats = np.delete(feats, ignore, axis=0)
                labels = ignore_labels
                feats = ignore_feats
            return ignore_labels, ignore_feats
        else:
            return labels, feats

    def __len__(self):
        return len(self._datalist)

    def __getitem__(self, idx: int):
        spl, label, label_names = self.read_one_data(idx)
        spls = spl.reshape(spl.shape[0], spl.shape[-1])

        name = self._datalist[idx]
        video = {
            "samples": spls,
            "labels": label,
            "idx": idx,
            "name": name,
            "label_names": label_names,
            "avg_cls": self._avg_cls,
        }
        return video
