### Author Silvia. L. Pintea
# Dataset class for reading "YouTube Instructions" 3000D features using the features provided in:
# Kukleva, Anna, et al. "Unsupervised learning of action classes with continuous temporal embedding."
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
# https://github.com/Annusha/unsup_temp_embed/blob/master/HOWTO_master.md

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import os
import numpy as np

import struct


class YTIVideo:
    def __init__(
        self,
        args: Any,
        set_name: str,
    ):
        self._args = args
        self._set_name = set_name
        self._ignore_class = []

        args.allavg = False
        if args.allavg:
            self._avg_cls = {
                "changing_tire": 8,
                "coffee": 8,
                "cpr": 8,
                "jump_car": 8,
                "repot": 8,
            }
        else:
            self._avg_cls = {
                "changing_tire": 10,
                "coffee": 8,
                "cpr": 6,
                "jump_car": 10,
                "repot": 7,
            }

        # Path to files
        self._datapath = args.datapath
        self._datalist = self.list_data(args)

    def list_data(self, args: Any):
        files = []
        feature_path = os.path.join(self._datapath, "features")
        dirs = os.listdir(feature_path)

        # Loop over subactions
        for d in dirs:
            curr_path = os.path.join(feature_path, d)
            files += [
                f
                for f in os.listdir(curr_path)
                if os.path.isfile(os.path.join(curr_path, f))
            ]
        files.sort()
        return files

    def read_one_data(self, idx: int):
        find_last = (self._datalist[idx]).rfind("_")
        actiondir = self._datalist[idx][0:find_last]

        feat_path = os.path.join(self._datapath, "features")
        gt_path = os.path.join(self._datapath, "groundTruth")

        feat = np.loadtxt(
            os.path.join(os.path.join(feat_path, actiondir), self._datalist[idx])
        )
        feat = feat.astype(np.float32)

        # Read the GTs
        gt_str = np.loadtxt(os.path.join(gt_path, self._datalist[idx][0:-4]), dtype="U")
        if len(self._ignore_class) > 0:
            gt_str, feat = self.ignore_class(gt_str, feat)

        subactions = np.unique(gt_str)
        gt = [np.where(np.array(subactions) == gs)[0][0] for gs in gt_str]
        return feat, gt, subactions, self._avg_cls[actiondir]

    def ignore_class(self, labels, feats):
        ignore = np.where(np.array(labels) == self._ignore_class)[0].tolist()
        ignore_labels = np.delete(np.array(labels), ignore).tolist()
        ignore_feats = np.delete(feats, ignore, axis=0)
        return ignore_labels, ignore_feats

    def __len__(self):
        return len(self._datalist)

    def __getitem__(self, idx: int):
        spl, label, label_names, avg_cls = self.read_one_data(idx)
        video = {
            "samples": spl.reshape(spl.shape[0], spl.shape[-1]),
            "labels": np.array(label).astype(int),
            "label_names": label_names,
            "idx": idx,
            "name": self._datalist[idx][0:-4],
            "avg_cls": avg_cls,
        }
        return video
