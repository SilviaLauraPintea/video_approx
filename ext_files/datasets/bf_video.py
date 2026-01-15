### Author Silvia. L. Pintea
# Dataset class for reading "Breakfast" IDT (improved dense trajectory) features using the features provided in:
# Kukleva, Anna, et al. "Unsupervised learning of action classes with continuous temporal embedding."
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
# https://github.com/Annusha/unsup_temp_embed/blob/master/HOWTO_master.md


from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import os
import numpy as np

import struct


class BFVideo:
    def __init__(
        self,
        args: Any,
        set_name: str,
    ):
        self._args = args
        self._set_name = set_name
        self._ignore_class = ""
        args.allavg = False

        if args.allavg:
            self._avg_cls = {
                "cereals": 5,
                "coffee": 5,
                "friedegg": 5,
                "juice": 5,
                "milk": 5,
                "pancake": 5,
                "salat": 5,
                "sandwich": 5,
                "scrambledegg": 5,
                "tea": 5,
            }
        else:
            self._avg_cls = {
                "cereals": 4,
                "coffee": 4,
                "friedegg": 6,
                "juice": 5,
                "milk": 4,
                "pancake": 9,
                "salat": 5,
                "sandwich": 5,
                "scrambledegg": 7,
                "tea": 4,
            }

        # Path to files
        self._datapath = args.datapath
        datalist = self.list_data(args)

        # Pick only the set
        self._datalist = self.pick_set(datalist, args.split, set_name)

    def list_data(self, args: Any):
        files = []
        feature_path = os.path.join(self._datapath, "features")

        print(feature_path)
        dirs = os.listdir(feature_path)

        # Loop over subactions
        for d in dirs:
            curr_path = os.path.join(feature_path, d)
            files += [
                os.path.join(curr_path, f)
                for f in os.listdir(curr_path)
                if os.path.isfile(os.path.join(curr_path, f))
            ]
        return files

    def pick_set(self, datalist: List, split: str, setname: str) -> List:
        if split.startswith("s1"):
            test_id_start = 3
            test_id_end = 16
        elif split.startswith("s2"):
            test_id_start = 16
            test_id_end = 29
        elif split.startswith("s3"):
            test_id_start = 29
            test_id_end = 42
        elif split.startswith("s4"):
            test_id_start = 42
            test_id_end = 55
        else:  # picks all
            if setname.startswith("test"):
                test_id_start = 2
                test_id_end = 56

        files = self.split_files(setname, test_id_start, test_id_end, datalist)
        files.sort()
        return files

    def split_files(self, setname: str, starti: int, endi: int, datalist: List[str]):
        np.random.seed(0)
        files = {"trainval": [], "train": [], "test": [], "val": []}
        for filen in datalist:
            fbase = os.path.basename(filen)
            pid = int(fbase[1:3])

            if pid >= starti and pid < endi:  # test video
                files["test"].append(fbase)
            else:
                files["trainval"].append(fbase)

        # Split train into trainval
        if setname.startswith("test"):
            return files["test"]
        else:
            idx = np.arange(len(files["trainval"]))
            np.random.shuffle(idx)

            # Split into val and train (where it test?)
            if setname.startswith("train"):
                return (
                    np.array(files["trainval"])[idx[0 : 2 * len(idx) // 3]]
                ).tolist()
            elif setname.startswith("val"):
                return (
                    np.array(files["trainval"])[idx[2 * len(idx) // 3 : len(idx)]]
                ).tolist()

    def read_one_data(self, idx: int):
        find_last = (self._datalist[idx]).rfind("_")
        actiondir = self._datalist[idx][find_last + 1 : -4]

        feat_path = os.path.join(self._datapath, "features")
        gt_path = os.path.join(self._datapath, "groundTruth")

        # Read the features
        feat = np.loadtxt(
            os.path.join(os.path.join(feat_path, actiondir), self._datalist[idx])
        )
        gt_str = np.loadtxt(os.path.join(gt_path, self._datalist[idx][0:-4]), dtype="U")
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

        name = self._datalist[idx]
        video = {
            "samples": spl.reshape(spl.shape[0], spl.shape[-1]),
            "labels": np.array(label).astype(int),
            "idx": idx,
            "name": name,
            "label_names": label_names,
            "avg_cls": avg_cls,
        }
        return video
