import cv2
import os
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from config import *

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = model
        self.h = 256
        self.w = 448
        self.data_root = path
        self.samples = None

        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')

        # Mode A: original Vimeo-90K list files.
        if os.path.exists(train_fn) and os.path.exists(test_fn) and os.path.isdir(self.image_root):
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines()
            with open(test_fn, 'r') as f:
                self.testlist = f.read().splitlines()
        else:
            # Mode B: auto-discover from folder tree, e.g.:
            # G:/val/val/000/val_orig/*.png
            self._init_samples_from_tree()

        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.samples is not None:
            self.meta_data = self.samples
            return

        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    @staticmethod
    def _frame_sort_key(path_obj):
        stem = path_obj.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def _list_frame_files(self, frame_dir):
        frame_dir = Path(frame_dir)
        files = [
            p for p in frame_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        ]
        files.sort(key=self._frame_sort_key)
        return files

    def _find_frame_dirs(self):
        root = Path(self.data_root)
        frame_dirs = []
        seen = set()
        preferred_leaf_names = {"images", "val_orig"}

        for current_root, dirnames, filenames in os.walk(root):
            cur = Path(current_root)

            # Prefer explicit frame folders like ".../images" or ".../val_orig".
            for d in dirnames:
                if d in preferred_leaf_names:
                    candidate = (cur / d).resolve()
                    if candidate not in seen:
                        seen.add(candidate)
                        frame_dirs.append(candidate)

            # Fallback: a folder containing image files directly.
            has_image = any(Path(f).suffix.lower() in self.IMG_EXTS for f in filenames)
            if has_image:
                resolved = cur.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    frame_dirs.append(resolved)

        frame_dirs.sort(key=lambda p: str(p))
        return frame_dirs

    @staticmethod
    def _is_test_like(path_str):
        s = path_str.lower()
        return ("test" in s) or ("val" in s) or ("valid" in s)

    def _init_samples_from_tree(self):
        all_triplets = []
        for frame_dir in self._find_frame_dirs():
            frames = self._list_frame_files(frame_dir)
            if len(frames) < 3:
                continue
            for i in range(len(frames) - 2):
                all_triplets.append((str(frames[i]), str(frames[i + 1]), str(frames[i + 2])))

        if len(all_triplets) == 0:
            raise RuntimeError(
                "No valid triplets found under '{}'. Expected folders like "
                "'.../images/*.png' or '.../val_orig/*.png'.".format(self.data_root)
            )

        # Deterministic split behavior for auto-discovered data.
        test_like = [t for t in all_triplets if self._is_test_like(t[0])]
        train_like = [t for t in all_triplets if not self._is_test_like(t[0])]

        if self.dataset_name == 'test':
            self.samples = test_like if len(test_like) > 0 else all_triplets
        else:
            self.samples = train_like if len(train_like) > 0 else all_triplets

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        if self.samples is not None:
            imgpaths = self.meta_data[index]
        else:
            imgpath = os.path.join(self.image_root, self.meta_data[index])
            imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        if img0 is None or gt is None or img1 is None:
            raise FileNotFoundError("Failed to read frames: {}".format(imgpaths))
        return img0, gt, img1
            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
                
        if 'train' in self.dataset_name:
            img0, gt, img1 = self.aug(img0, gt, img1, 256, 256)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)
