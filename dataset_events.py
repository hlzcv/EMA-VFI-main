import os
import cv2
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import Dataset

from event_process import event as ev
from event_process.representation import to_voxel_grid


class BSERGBEventDataset(Dataset):
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
    EV_EXTS = {".npz", ".npy"}

    def __init__(self, dataset_name: str, root: str, num_bins: int = 5):
        self.dataset_name = dataset_name.lower()
        self.root = Path(root)
        self.num_bins = int(num_bins)
        self.crop_h = 256
        self.crop_w = 256
        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found for BS-ERGB event dataset.")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _sort_key(path_obj: Path) -> Tuple[int, float, str]:
        stem = path_obj.stem
        try:
            return (0, float(stem), stem)
        except ValueError:
            return (1, 0.0, stem)

    def _resolve_split_roots(self) -> List[Path]:
        split_map = {
            "train": ["3_TRAINING"],
            "validation": ["2_VALIDATION"],
            "val": ["2_VALIDATION"],
            "test": ["1_TEST"],
        }

        candidates = split_map.get(self.dataset_name, ["3_TRAINING"])
        roots = []
        for split_name in candidates:
            split_root = self.root / split_name
            if split_root.is_dir():
                roots.append(split_root)

        if roots:
            return roots
        return [self.root]

    def _list_scene_dirs(self, split_root: Path) -> List[Path]:
        scenes = []
        for d in sorted(split_root.iterdir()):
            if not d.is_dir():
                continue
            if (d / "images").is_dir() and (d / "events").is_dir():
                scenes.append(d)
        return scenes

    def _list_files(self, folder: Path, exts: set) -> List[Path]:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=self._sort_key)
        return files

    @staticmethod
    def _safe_timestamp(path_obj: Path, default_idx: int) -> float:
        try:
            return float(path_obj.stem)
        except ValueError:
            return float(default_idx)

    def _select_event_paths(
        self,
        event_paths: List[Path],
        img_paths: List[Path],
        i0: int,
        i2: int,
    ) -> List[Path]:
        if len(event_paths) == 0:
            return []

        t0 = self._safe_timestamp(img_paths[i0], i0)
        t2 = self._safe_timestamp(img_paths[i2], i2)
        lo, hi = min(t0, t2), max(t0, t2)

        selected = []
        for j, ep in enumerate(event_paths):
            te = self._safe_timestamp(ep, j)
            if lo <= te <= hi:
                selected.append(ep)

        if selected:
            return selected

        # Fallback: nearest by index window when timestamps do not match naming format.
        j0 = min(i0, len(event_paths) - 1)
        j2 = min(i2, len(event_paths) - 1)
        if j0 > j2:
            j0, j2 = j2, j0
        return event_paths[j0 : j2 + 1]

    def _build_samples(self) -> List[dict]:
        samples = []
        for split_root in self._resolve_split_roots():
            for scene_dir in self._list_scene_dirs(split_root):
                image_paths = self._list_files(scene_dir / "images", self.IMG_EXTS)
                event_paths = self._list_files(scene_dir / "events", self.EV_EXTS)
                if len(image_paths) < 3:
                    continue

                for i in range(len(image_paths) - 2):
                    selected_events = self._select_event_paths(event_paths, image_paths, i, i + 2)
                    samples.append(
                        {
                            "img0": str(image_paths[i]),
                            "gt": str(image_paths[i + 1]),
                            "img1": str(image_paths[i + 2]),
                            "events": [str(p) for p in selected_events],
                        }
                    )
        return samples

    def _load_event_voxel(self, event_files: List[str], h: int, w: int) -> torch.Tensor:
        if len(event_files) == 0:
            return torch.zeros(self.num_bins, h, w, dtype=torch.float32)

        features_list = []
        for f in event_files:
            if not os.path.isfile(f):
                continue
            features = ev.load_events(f)
            if features.size == 0:
                continue
            features_list.append(features)

        if len(features_list) == 0:
            return torch.zeros(self.num_bins, h, w, dtype=torch.float32)

        features = np.concatenate(features_list, axis=0)
        if features.shape[0] == 0:
            return torch.zeros(self.num_bins, h, w, dtype=torch.float32)

        start_t = float(features[0, ev.TIMESTAMP_COLUMN])
        end_t = float(features[-1, ev.TIMESTAMP_COLUMN])
        if end_t <= start_t:
            end_t = start_t + 1e-6

        seq = ev.EventSequence(
            features=features,
            image_height=h,
            image_width=w,
            start_time=start_t,
            end_time=end_t,
        )
        voxel = to_voxel_grid(seq, nb_of_time_bins=self.num_bins)
        return voxel.float()

    def _augment_train(
        self,
        img0: np.ndarray,
        gt: np.ndarray,
        img1: np.ndarray,
        event_feat: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        ih, iw, _ = img0.shape
        h = min(self.crop_h, ih)
        w = min(self.crop_w, iw)
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)

        img0 = img0[x : x + h, y : y + w, :]
        gt = gt[x : x + h, y : y + w, :]
        img1 = img1[x : x + h, y : y + w, :]
        event_feat = event_feat[:, x : x + h, y : y + w]

        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, :, ::-1]
            gt = gt[:, :, ::-1]
            img1 = img1[:, :, ::-1]

        if random.uniform(0, 1) < 0.5:
            img0, img1 = img1, img0

        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            gt = gt[::-1]
            img1 = img1[::-1]
            event_feat = torch.flip(event_feat, dims=[1])

        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            gt = gt[:, ::-1]
            img1 = img1[:, ::-1]
            event_feat = torch.flip(event_feat, dims=[2])

        p = random.uniform(0, 1)
        if p < 0.25:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            event_feat = torch.rot90(event_feat, k=-1, dims=[1, 2])
        elif p < 0.5:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt = cv2.rotate(gt, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
            event_feat = torch.rot90(event_feat, k=2, dims=[1, 2])
        elif p < 0.75:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            event_feat = torch.rot90(event_feat, k=1, dims=[1, 2])

        return img0.copy(), gt.copy(), img1.copy(), event_feat.contiguous()

    def __getitem__(self, index: int):
        sample = self.samples[index]

        img0 = cv2.imread(sample["img0"], cv2.IMREAD_COLOR)
        gt = cv2.imread(sample["gt"], cv2.IMREAD_COLOR)
        img1 = cv2.imread(sample["img1"], cv2.IMREAD_COLOR)
        if img0 is None or gt is None or img1 is None:
            raise FileNotFoundError("Failed to read RGB frames in sample index {}".format(index))

        h, w = img0.shape[:2]
        event_feat = self._load_event_voxel(sample["events"], h, w)

        if "train" in self.dataset_name:
            img0, gt, img1, event_feat = self._augment_train(img0, gt, img1, event_feat)

        img0_t = torch.from_numpy(img0).permute(2, 0, 1)
        img1_t = torch.from_numpy(img1).permute(2, 0, 1)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1)

        # Keep original training tensor format + extra event feature branch.
        rgb_triplet = torch.cat((img0_t, img1_t, gt_t), 0)
        return rgb_triplet, event_feat
