import os
import cv2
import copy
import math
import glob
import json
import torch
import random
import scipy.ndimage
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage import exposure
from collections import Counter
from typing import Tuple, Dict, Optional, List
from torch.utils.data import Dataset  # Base class for PyTorch datasets

from scipy.ndimage import label, distance_transform_edt


#####################################
# Data preprocessing and augmentation routines
#####################################

# In this dataset, vertebra labels correspond to:
#   1–7   : cervical spine (C1–C7)
#   8–19  : thoracic spine (T1–T12)
#   20–25 : lumbar spine (L1–L6)
#   26,27 : sacrum and coccyx (not labeled here)
#   28    : T13 (not present in this dataset)
def compute_n_classes(derivatives_folder: str) -> Tuple[int, Dict[int,int]]:
    """
    Traverse all subdirectories of `derivatives_folder`, find files ending with "_ctd.json",
    and count how many times each vertebra label (1..25) appears.

    Returns:
      - n_classes: maximum vertebra label (1–25) + 1 => e.g., 25 if labels are 1..25.
                   We do not count background (0) here, background is implicit.
      - cls_counts: a dictionary {label_id: occurrence_count}. Missing labels are filled with 0.

    Use case: determine class weights for Dice or cross-entropy loss based on frequency.
    """
    valid_labels = set(range(1, 26))  # Only count labels in the range 1–25
    counts = Counter()

    # Walk through all files under derivatives_folder
    for root, _, files in os.walk(derivatives_folder):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                items = json.load(open(path, "r"))
            except Exception as e:
                print(f"[WARN] Could not load {path}: {e}")
                continue

            # Some JSONs might be a dict or a list of dicts
            if isinstance(items, dict):
                items = [items]
            for it in items:
                lbl = it.get("label", None)
                # Count only if it's an integer in 1–25
                if isinstance(lbl, int) and lbl in valid_labels:
                    counts[lbl] += 1

    # Ensure all labels 1–25 have an entry, even if zero
    for lbl in valid_labels:
        counts.setdefault(lbl, 0)

    n_classes = max(valid_labels)  # This yields 25 (no +1 for background here)
    cls_counts = {lbl: counts[lbl] for lbl in sorted(valid_labels)}
    return n_classes, cls_counts



##########################################################################
# Dataset classes
##########################################################################
class VertebraDataset3D(Dataset):
    """
    3D Dataset for the VerSe_2020 spine segmentation task.
    Each sample returns a dictionary containing:
      - "volume"      : a 3D tensor (or 3D patch) of the CT scan, normalized to [0,1].
      - "mask"        : a 3D tensor of the segmentation mask (0=background, >0=vertebra ID).
      - "volume_ori"  : the original cropped volume before normalization (for debugging).
      - "volume_weak" : (only for unlabeled training) a weakly augmented version of volume.
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 labeled: bool = True,
                 semi_supervised_ratio: float = 0.5,
                 crop_ratio: float = 0.75,
                 num_classes: int = 25,
                 output_size: Tuple[int,int,int] = (64, 64, 64),
                 debug: bool = False,
                 debug_dir: str = './debug_nii'):
        """
        Args:
          root_dir:                path to dataset root (contains subfolders "01_training", "02_validation", etc.).
          split:                   one of "train", "val", or "test".
          labeled:                 whether to load segmentation masks (if False, only volume is used).
          semi_supervised_ratio:   fraction of labeled data to keep if split='train' and labeled=True.
          crop_ratio:              fraction for cropping XY plane.
          output_size:             (H, W, D) output size after cropping & resizing.
          debug:                   if True, save some intermediate NIfTI files for inspection.
          debug_dir:               directory to save debug outputs.
        """
        super(VertebraDataset3D, self).__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.labeled = labeled
        self.semi_supervised_ratio = semi_supervised_ratio
        self.crop_ratio = crop_ratio
        self.output_size = output_size
        self.debug = debug
        self.num_classes = num_classes

        # Determine which folder to look in
        if self.split == 'train':
            self.data_dir = os.path.join(root_dir, "01_training")
        elif self.split == 'val':
            self.data_dir = os.path.join(root_dir, "02_validation")
        elif self.split == 'test':
            self.data_dir = os.path.join(root_dir, "03_test")
        elif self.split == 'all':
            self.data_dir = os.path.join(root_dir, "01+02+03")
        else:
            raise ValueError("split must be one of 'train', 'val', 'test', or 'all(train+val+test)'")

        subjects_path = os.path.join(self.data_dir, "rawdata")
        self.subjects = sorted(os.listdir(subjects_path))

        # If semi-supervised, only keep a subset of labeled subjects
        if self.split == 'train' and self.labeled and semi_supervised_ratio < 1.0:
            n_labeled = int(len(self.subjects) * semi_supervised_ratio)
            self.subjects = self.subjects[:n_labeled]

        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)
        self.debug_dir = debug_dir
        self._debug_written = set()  # Track which subjects have been debug-saved

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single subject by index:
          1. Load the 3D volume (.npy) and reorder slices so that Z-axis is last.
          2. If labeled, load the mask (.npy) similarly.
          3. Compute crop indices: random if train, center if val/test.
          4. Crop, then resize to output_size, then pad in Z if needed.
          5. Normalize intensities to [0,1]: apply window [-500,1500].
          6. Apply strong or weak augmentations (for train).
          7. Return a dictionary with keys "volume", "mask", "volume_ori", and possibly "volume_weak".

        Returns:
          sample: dict where
            - "volume"     : tensor [1, H, W, D] (float32)
            - "mask"       : tensor [H, W, D] (int64) if labeled, else None
            - "volume_ori" : numpy crop before normalization (for debugging)
            - "volume_weak": (if unlabeled train) weakly augmented volume
        """
        sub = self.subjects[idx]

        # 1. Load CT volume (assumes .npy format)
        ct_pattern = os.path.join(self.data_dir, "rawdata", sub, f"*_ct.npy")
        ct_files = glob.glob(ct_pattern)
        if len(ct_files) == 0:
            raise FileNotFoundError(f"No CT file found for subject {sub}, pattern: {ct_pattern}")
        # Some years may have multiple versions; pick the last one
        ct_path = ct_files[-1]
        volume = np.load(ct_path)

        # 2. Load mask if labeled
        mask = None
        if self.labeled:
            mask_pattern = os.path.join(self.data_dir, "derivatives", sub, f"*_msk.npy")
            mask_files = glob.glob(mask_pattern)
            if len(mask_files) > 0:
                mask_path = mask_files[-1]
                mask = np.load(mask_path)
            else:
                # If no mask found, mark as unlabeled
                self.labeled = False

        nx, ny, nz = volume.shape
        ox, oy, oz = self.output_size
        if mask is not None:
            # 1. Compute a tight bounding box around the mask projection
            x0, x1, y0, y1, z0, z1 = get_mask_proj_bbox(mask, margin_xy=0, margin_z=0)

            # 2. Determine random margins for train, fixed margins for val/test
            if self.split == 'train':
                margin_xy = random.randint(16, 48)
                margin_z = random.randint(8, 24)
            else:
                margin_xy = 32
                margin_z = 16

            # 3. Compute square side length in XY and half-side
            len_x = x1 - x0
            len_y = y1 - y0
            base_xy = min(len_x, len_y)
            side_xy = base_xy + 2 * margin_xy
            side_xy = min(side_xy, nx, ny)
            half_xy = side_xy // 2

            # Center of the bounding box
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2

            # 3a. Compute X-axis crop boundaries
            x_min = max(0, cx - half_xy)
            x_max = x_min + side_xy
            if x_max > nx:
                x_max = nx
                x_min = max(0, nx - side_xy)

            # 3b. Compute Y-axis crop boundaries
            y_min = max(0, cy - half_xy)
            y_max = y_min + side_xy
            if y_max > ny:
                y_max = ny
                y_min = max(0, ny - side_xy)

            # 4. Compute target depth proportional to XY size
            target_z = round(side_xy * oz / ox)
            target_z = min(target_z, nz)

            # 5. Determine Z start: random shift for train, center for val/test
            if self.split == 'train':
                z_lo = max(0, z0 - margin_z)
                z_hi = min(nz - target_z, z1 + margin_z - target_z)
                if z_hi < z_lo:
                    mid = (z0 + z1) // 2
                    z_lo = max(0, mid - target_z // 2)
                    z_hi = z_lo
                z_start = random.randint(z_lo, z_hi)
            else:
                mid = (z0 + z1) // 2
                z_start = max(0, min(nz - target_z, mid - target_z // 2))

            z_min = z_start
            z_max = z_start + target_z
        else:
            # Fallback: random crop for train, center crop for val/test
            if self.split == 'train':
                x_min, x_max, y_min, y_max, z_min, z_max = get_random_crop_indices(
                    volume, crop_ratio=self.crop_ratio, output_size=self.output_size)
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = get_center_crop_indices(
                    volume, crop_ratio=self.crop_ratio, output_size=self.output_size)

        # Perform crop
        volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        if mask is not None:
            mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]

        # Pad depth if needed
        h, w, d = volume.shape
        xy = max(h, w)
        desired_d = int(round(xy * self.output_size[-1] / self.output_size[0]))
        if d < desired_d:
            pad_total = desired_d - d
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            min_val = volume.min()
            volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)),
                            mode='constant', constant_values=min_val)
            if mask is not None:
                mask = np.pad(mask, ((0, 0), (0, 0), (pad_before, pad_after)),
                              mode='constant', constant_values=0)

        # Resize to output size
        volume = resize_volume(volume, target_shape=self.output_size, order=1)
        if mask is not None:
            mask = resize_volume(mask, target_shape=self.output_size, order=0)

        # Normalize CT intensities
        volume_normal = normalize_volume(volume)

        # 9. If training, apply augmentations and convert to tensors
        if self.split == 'train':
            if self.labeled and mask is not None:
                # Perform strong augmentation on both volume and mask
                strong_vol_np, strong_mask_np = strong_augment_3d_w_pair(volume_normal, mask, our=True)
                strong_vol = to_tensor_3d(strong_vol_np)
                strong_mask = torch.from_numpy(strong_mask_np).long()
                raw_centers = compute_centroids_via_dt(strong_mask_np, num_classes=self.num_classes)
                strong_centers = [tuple(c) if c[0] >= 0 else (-1, -1, -1) for c in raw_centers]
                sample = {
                    "volume": strong_vol,  # [1, H, W, D]
                    "mask": strong_mask,  # [H, W, D]
                    "centers": strong_centers,  #
                    "volume_ori": volume,  # Original cropped volume (for debug)
                    "subject": sub,
                }
                # sample = {
                #     "volume": strong_vol,         # [1, H, W, D]
                #     "mask": strong_mask,          # [H, W, D]
                #     "volume_ori": volume,         # Original cropped volume (for debug)
                #     "subject": sub,
                # }
            else:
                # For unlabeled data, generate a weak and strong augmented pair
                weak_vol_np = weak_augment_3d_wo_pair(volume_normal)
                strong_vol_np = strong_augment_3d_wo_pair(weak_vol_np)
                weak_vol = to_tensor_3d(weak_vol_np)
                strong_vol = to_tensor_3d(strong_vol_np)
                sample = {
                    "volume": strong_vol,         # Strongly augmented
                    "volume_weak": weak_vol,      # Weakly augmented
                    "volume_ori": volume,         # Original cropped volume
                    "subject": sub,
                }
        else:
            # Validation / test: no augmentations, just normalize and to tensor
            strong_vol = to_tensor_3d(volume_normal)
            strong_mask = torch.from_numpy(mask).long() if mask is not None else None
            raw_centers = compute_centroids_via_dt(mask, num_classes=self.num_classes)
            strong_centers = [tuple(c) if c[0] >= 0 else (-1, -1, -1) for c in raw_centers]
            sample = {
                "volume": strong_vol,  # [1, H, W, D]
                "mask": strong_mask,  # [H, W, D]
                "centers": strong_centers,  #
                "volume_ori": volume,  # Original cropped volume (for debug)
                "subject": sub,
            }
            # sample = {
            #     "volume": strong_vol,          # No augmentation
            #     "mask": strong_mask,
            #     "volume_ori": volume,
            #     "subject": sub,
            # }

        # 8. If debug is on, save a few volumes/masks as NIfTI for inspection
        if self.debug and sub not in self._debug_written:
            # affine = np.eye(4)  # Identity affine (no real-world spacing)
            affine = np.array([
                [1, 0, 0, 0],  # +X  (Right)
                [0, -1, 0, 0],  # -Y  (Posterior)
                [0, 0, -1, 0],  # -Z  (Inferior)
                [0, 0, 0, 1],
            ], dtype=float)
            for key, arr in sample.items():
                # Only save volume/weak/ori/mask
                if key not in ("volume", "mask", "volume_weak", "volume_ori"):
                    continue
                # Convert tensor back to numpy if needed
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                # Reverse normalization or type-cast depending on key
                if key == "volume_ori":
                    arr2 = np.clip(arr, -500, 1500).astype(np.int16)
                elif key in ("volume_weak", "volume"):
                    arr2 = arr.squeeze(0)        # remove channel dimension
                    arr2 = denormalize(arr2)     # map [0,1] back to [-500,1500]
                    arr2 = np.clip(arr2, -500, 1500).astype(np.int16)
                elif key == "mask":
                    arr2 = arr.astype(np.uint8)
                else:
                    continue

                out_path = os.path.join(self.debug_dir, f"{sub}_{key}.nii.gz")
                nib.save(nib.Nifti1Image(arr2, affine), out_path)

            self._debug_written.add(sub)

        return sample



class VertebraDataset3DHigh(Dataset):
    """
    3D Dataset for the VerSe_2020 spine segmentation task.
    Each sample returns a dictionary containing:
      - "volume"      : a 3D tensor (or 3D patch) of the CT scan, normalized to [0,1].
      - "mask"        : a 3D tensor of the segmentation mask (0=background, >0=vertebra ID).
      - "volume_ori"  : the original cropped volume before normalization (for debugging).
      - "volume_weak" : (only for unlabeled training) a weakly augmented version of volume.
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 labeled: bool = True,
                 semi_supervised_ratio: float = 0.5,
                 crop_ratio: float = 0.75,
                 output_size: Tuple[int,int,int] = (64, 64, 128),
                 num_classes: int = 25,
                 debug: bool = False,
                 debug_dir: str = './debug_nii'):
        """
        Args:
          root_dir:                path to dataset root (contains subfolders "01_training", "02_validation", etc.).
          split:                   one of "train", "val", or "test".
          labeled:                 whether to load segmentation masks (if False, only volume is used).
          semi_supervised_ratio:   fraction of labeled data to keep if split='train' and labeled=True.
          crop_ratio:              fraction for cropping XY plane.
          output_size:             (H, W, D) output size after cropping & resizing.
          debug:                   if True, save some intermediate NIfTI files for inspection.
          debug_dir:               directory to save debug outputs.
        """
        super(VertebraDataset3DHigh, self).__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.labeled = labeled
        self.semi_supervised_ratio = semi_supervised_ratio
        self.crop_ratio = crop_ratio
        self.output_size = output_size
        self.debug = debug
        self.num_classes = num_classes

        # Determine which folder to look in
        if self.split == 'train':
            self.data_dir = os.path.join(root_dir, "01_training")
        elif self.split == 'val':
            self.data_dir = os.path.join(root_dir, "02_validation")
        elif self.split == 'test':
            self.data_dir = os.path.join(root_dir, "03_test")
        elif self.split == 'all':
            self.data_dir = os.path.join(root_dir, "01+02+03")
        else:
            raise ValueError("split must be one of 'train', 'val', 'test', or 'all(train+val+test)'")

        subjects_path = os.path.join(self.data_dir, "rawdata")
        self.subjects = sorted(os.listdir(subjects_path))

        # If semi-supervised, only keep a subset of labeled subjects
        if self.split == 'train' and self.labeled and semi_supervised_ratio < 1.0:
            n_labeled = int(len(self.subjects) * semi_supervised_ratio)
            self.subjects = self.subjects[:n_labeled]

        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)
        self.debug_dir = debug_dir
        self._debug_written = set()  # Track which subjects have been debug-saved

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single subject by index:
          1. Load the 3D volume (.npy) and reorder slices so that Z-axis is last.
          2. If labeled, load the mask (.npy) similarly.
          3. Compute crop indices: random if train, center if val/test.
          4. Crop, then resize to output_size, then pad in Z if needed.
          5. Normalize intensities to [0,1]: apply window [-500,1500].
          6. Apply strong or weak augmentations (for train).
          7. Return a dictionary with keys "volume", "mask", "volume_ori", and possibly "volume_weak".

        Returns:
          sample: dict where
            - "volume"     : tensor [1, H, W, D] (float32)
            - "mask"       : tensor [H, W, D] (int64) if labeled, else None
            - "volume_ori" : numpy crop before normalization (for debugging)
            - "volume_weak": (if unlabeled train) weakly augmented volume
        """
        sub = self.subjects[idx]

        ############### load volume/mask/centers
        # 1. Load CT volume (assumes .npy format)
        ct_pattern = os.path.join(self.data_dir, "rawdata", sub, f"*_ct.npy")
        ct_files = glob.glob(ct_pattern)
        if len(ct_files) == 0:
            raise FileNotFoundError(f"No CT file found for subject {sub}, pattern: {ct_pattern}")
        # Some years may have multiple versions; pick the last one
        ct_path = ct_files[-1]
        volume = np.load(ct_path)

        # 2. Load mask if labeled
        mask = None
        if self.labeled:
            mask_pattern = os.path.join(self.data_dir, "derivatives", sub, f"*_msk.npy")
            mask_files = glob.glob(mask_pattern)
            if len(mask_files) > 0:
                mask_path = mask_files[-1]
                mask = np.load(mask_path)
            else:
                # If no mask found, mark as unlabeled
                self.labeled = False

        # # 3. Load centers (assumes .json format)
        # centers = None
        # if self.labeled:
        #     ctd_pattern = os.path.join(self.data_dir, "derivatives", sub, "*-resampled.json")
        #     ctd_files = glob.glob(ctd_pattern)
        #     if len(ctd_files) > 0:
        #         centers = load_centers_from_json(ctd_files[-1])  # (25,3) or None
        #     else:
        #         # If no mask found, mark as unlabeled
        #         self.labeled = False

        ############################# bounding box around the mask projection
        nx, ny, nz = volume.shape
        if mask is not None:
            # 1. Compute a tight bounding box around the mask projection
            x0, x1, y0, y1, z0, z1 = get_mask_proj_bbox(mask, margin_xy=0, margin_z=0)

            # 2. Determine random margins for train, fixed margins for val/test
            if self.split == 'train':
                margin_xy = random.randint(4, 24)
                margin_z = random.randint(4, 12)
            else:
                margin_xy = 12
                margin_z = 6

            # 3. Compute square side length in XY and half-side
            len_x = x1 - x0
            len_y = y1 - y0
            base_xy = min(len_x, len_y)
            side_xy = base_xy + 2 * margin_xy
            side_xy = min(side_xy, nx, ny)
            half_xy = side_xy // 2

            # Center of the bounding box
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2

            # 3a. Compute X-axis crop boundaries
            x_min = max(0, cx - half_xy)
            x_max = x_min + side_xy
            if x_max > nx:
                x_max = nx
                x_min = max(0, nx - side_xy)

            # 3b. Compute Y-axis crop boundaries
            y_min = max(0, cy - half_xy)
            y_max = y_min + side_xy
            if y_max > ny:
                y_max = ny
                y_min = max(0, ny - side_xy)

            # 4. Compute target depth proportional to XY size
            z_min = max(0, z0 - margin_z)
            z_max = min(nz, z1 + margin_z)
        else:
            # Fallback: random crop for train, center crop for val/test
            if self.split == 'train':
                x_min, x_max, y_min, y_max, z_min, z_max = get_random_crop_indices(
                    volume, crop_ratio=self.crop_ratio, output_size=self.output_size)
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = get_center_crop_indices(
                    volume, crop_ratio=self.crop_ratio, output_size=self.output_size)


        # # Perform crop
        # volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        # if mask is not None:
        #     mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]
        #
        # # Pad depth if needed
        # h, w, d = volume.shape
        # xy = max(h, w)
        # desired_d = int(round(xy * self.output_size[-1] / self.output_size[0]))
        # if d < desired_d:
        #     pad_total = desired_d - d
        #     pad_before = pad_total // 2
        #     pad_after = pad_total - pad_before
        #     min_val = volume.min()
        #     volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)),
        #                     mode='constant', constant_values=min_val)
        #     if mask is not None:
        #         mask = np.pad(mask, ((0, 0), (0, 0), (pad_before, pad_after)),
        #                       mode='constant', constant_values=0)
        #
        # # Resize to output size
        # volume = resize_volume(volume, target_shape=self.output_size, order=1)
        # if mask is not None:
        #     mask = resize_volume(mask, target_shape=self.output_size, order=0)


        # ---------------- 裁剪前记录 offset ----------------
        volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        if mask is not None:
            mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]
        # box_off = np.array([x_min, y_min, z_min], dtype=np.float32)  # bounding box offset

        ############################# Pad
        # 计算裁剪范围
        h, w, d = volume.shape
        target_h, target_w, target_d = self.output_size
        pad_h_before, pad_w_before, pad_d_before = None, None, None

        # 确保图像尺寸大于目标尺寸（否则需要填充）
        if h < target_h or w < target_w or d < target_d:
            # 计算需要填充的尺寸
            pad_h = max(target_h - h, 0)
            pad_w = max(target_w - w, 0)
            pad_d = max(target_d - d, 0)

            # 对称填充
            pad_h_before = pad_h // 2
            pad_h_after = pad_h - pad_h_before
            pad_w_before = pad_w // 2
            pad_w_after = pad_w - pad_w_before
            pad_d_before = pad_d // 2
            pad_d_after = pad_d - pad_d_before

            volume = np.pad(volume,
                            ((pad_h_before, pad_h_after),
                             (pad_w_before, pad_w_after),
                             (pad_d_before, pad_d_after)),
                            mode='constant', constant_values=volume.min())
            if mask is not None:
                mask = np.pad(mask,
                              ((pad_h_before, pad_h_after),
                               (pad_w_before, pad_w_after),
                               (pad_d_before, pad_d_after)),
                              mode='constant', constant_values=0)
            # 更新尺寸
            h, w, d = volume.shape

        # # 若有 pad，记录 pad_off --------------
        # pad_off = np.array([pad_h_before, pad_w_before, pad_d_before],
        #                    dtype=np.float32) if pad_h_before is not None else np.zeros(3, np.float32)

        ############################# Crop to output_size
        if self.split == 'train':
            # 随机裁剪
            start_h = random.randint(0, h - target_h)
            start_w = random.randint(0, w - target_w)
            start_d = random.randint(0, d - target_d)
        else:
            start_h = int((h - target_h) / 2)
            start_w = int((w - target_w) / 2)
            start_d = int((d - target_d) / 2)

        volume = volume[start_h:start_h+target_h, start_w:start_w+target_w, start_d:start_d+target_d]
        if mask is not None:
            mask = mask[start_h:start_h+target_h, start_w:start_w+target_w, start_d:start_d+target_d]
        # crop_off = np.array([start_h, start_w, start_d], dtype=np.float32)# 经过最终随机 / 中心裁剪

        # # ---------------- 转换 json 质心 ----------------
        # centers_npy = None
        # if centers is not None:
        #     centers_npy = np.array(centers, dtype=np.float32)  # (25,3)
        #     # 物理坐标 → 裁剪 patch index
        #     centers_npy -= box_off  # 减去初始 crop 原点
        #     centers_npy += pad_off  # 若有 pad
        #     centers_npy -= crop_off  # 最终随机/中心裁剪

        # Normalize CT intensities
        volume_normal = normalize_volume(volume)

        # 9. If training, apply augmentations and convert to tensors
        if self.split == 'train':
            if self.labeled and mask is not None:
                # Perform strong augmentation on both volume and mask
                # strong_vol_np, strong_mask_np, strong_centers_np = \
                #     strong_augment_3d_w_pair(volume_normal, mask, centers_npy, our=True)
                strong_vol_np, strong_mask_np = strong_augment_3d_w_pair(volume_normal, mask, our=True)
                strong_vol = to_tensor_3d(strong_vol_np)
                strong_mask = torch.from_numpy(strong_mask_np).long()
                raw_centers = compute_centroids_via_dt(strong_mask_np, num_classes=self.num_classes)
                strong_centers = [tuple(c) if c[0] >= 0 else (-1,-1,-1) for c in raw_centers]
                # strong_centers = torch.from_numpy(strong_centers_np)
                sample = {
                    "volume": strong_vol,         # [1, H, W, D]
                    "mask": strong_mask,          # [H, W, D]
                    "centers": strong_centers,  #
                    "volume_ori": volume,         # Original cropped volume (for debug)
                    "subject": sub,
                }
            else:
                # For unlabeled data, generate a weak and strong augmented pair
                weak_vol_np = weak_augment_3d_wo_pair(volume_normal)
                strong_vol_np = strong_augment_3d_wo_pair(weak_vol_np)
                weak_vol = to_tensor_3d(weak_vol_np)
                strong_vol = to_tensor_3d(strong_vol_np)
                sample = {
                    "volume": strong_vol,         # Strongly augmented
                    "volume_weak": weak_vol,      # Weakly augmented
                    "volume_ori": volume,         # Original cropped volume
                    "subject": sub,
                }
        else:
            # Validation / test: no augmentations, just normalize and to tensor
            strong_vol = to_tensor_3d(volume_normal)
            strong_mask = torch.from_numpy(mask).long() if mask is not None else None
            # strong_centers = torch.from_numpy(centers_npy) if centers_npy is not None else None
            raw_centers = compute_centroids_via_dt(mask, num_classes=self.num_classes) if mask is not None else None
            strong_centers = [tuple(c) if c[0] >= 0 else (-1,-1,-1) for c in raw_centers] if mask is not None else None
            sample = {
                "volume": strong_vol,          # No augmentation
                "mask": strong_mask,
                "centers": strong_centers,
                "volume_ori": volume,
                "subject": sub,
            }

        # 8. If debug is on, save a few volumes/masks as NIfTI for inspection
        if self.debug and sub not in self._debug_written:
            # affine = np.eye(4)  # Identity affine (no real-world spacing)
            affine = np.array([
                [1, 0, 0, 0],  # +X  (Right)
                [0, -1, 0, 0],  # -Y  (Posterior)
                [0, 0, -1, 0],  # -Z  (Inferior)
                [0, 0, 0, 1],
            ], dtype=float)
            for key, arr in sample.items():
                # Only save volume/weak/ori/mask
                if key not in ("volume", "mask", "volume_weak", "volume_ori"):
                    continue
                # Convert tensor back to numpy if needed
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                # Reverse normalization or type-cast depending on key
                if key == "volume_ori":
                    arr2 = np.clip(arr, -500, 1500).astype(np.int16)
                elif key in ("volume_weak", "volume"):
                    arr2 = arr.squeeze(0)        # remove channel dimension
                    arr2 = denormalize(arr2)     # map [0,1] back to [-500,1500]
                    arr2 = np.clip(arr2, -500, 1500).astype(np.int16)
                elif key == "mask":
                    arr2 = arr.astype(np.uint8)
                else:
                    continue

                out_path = os.path.join(self.debug_dir, f"{sub}_{key}.nii.gz")
                nib.save(nib.Nifti1Image(arr2, affine), out_path)

            self._debug_written.add(sub)

        return sample



class VertebraDataset3DLowHigh(Dataset):
    """
    3D Dataset for the VerSe_2020 spine segmentation task.
    Each sample returns a dictionary containing:
      - "volume"      : a 3D tensor (or 3D patch) of the CT scan, normalized to [0,1].
      - "mask"        : a 3D tensor of the segmentation mask (0=background, >0=vertebra ID).
      - "volume_ori"  : the original cropped volume before normalization (for debugging).
      - "volume_weak" : (only for unlabeled training) a weakly augmented version of volume.
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 labeled: bool = True,
                 semi_supervised_ratio: float = 0.5,
                 crop_ratio: float = 0.75,
                 output_size: Tuple[int,int,int] = (64, 64, 128),
                 num_classes: int = 25,
                 debug: bool = False,
                 debug_dir: str = './debug_nii'):
        """
        Args:
          root_dir:                path to dataset root (contains subfolders "01_training", "02_validation", etc.).
          split:                   one of "train", "val", or "test".
          labeled:                 whether to load segmentation masks (if False, only volume is used).
          semi_supervised_ratio:   fraction of labeled data to keep if split='train' and labeled=True.
          crop_ratio:              fraction for cropping XY plane.
          output_size:             (H, W, D) output size after cropping & resizing.
          debug:                   if True, save some intermediate NIfTI files for inspection.
          debug_dir:               directory to save debug outputs.
        """
        super(VertebraDataset3DLowHigh, self).__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.labeled = labeled
        self.semi_supervised_ratio = semi_supervised_ratio
        self.crop_ratio = crop_ratio
        self.output_size = output_size
        self.debug = debug
        self.num_classes = num_classes

        # Determine which folder to look in
        if self.split == 'train':
            self.data_dir = os.path.join(root_dir, "01_training")
        elif self.split == 'val':
            self.data_dir = os.path.join(root_dir, "02_validation")
        elif self.split == 'test':
            self.data_dir = os.path.join(root_dir, "03_test")
        elif self.split == 'all':
            self.data_dir = os.path.join(root_dir, "01+02+03")
        else:
            raise ValueError("split must be one of 'train', 'val', 'test', or 'all(train+val+test)'")

        subjects_path = os.path.join(self.data_dir, "rawdata")
        self.subjects = sorted(os.listdir(subjects_path))

        # If semi-supervised, only keep a subset of labeled subjects
        if self.split == 'train' and self.labeled and semi_supervised_ratio < 1.0:
            n_labeled = int(len(self.subjects) * semi_supervised_ratio)
            self.subjects = self.subjects[:n_labeled]

        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)
        self.debug_dir = debug_dir
        self._debug_written = set()  # Track which subjects have been debug-saved

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single subject by index:
          1. Load the 3D volume (.npy) and reorder slices so that Z-axis is last.
          2. If labeled, load the mask (.npy) similarly.
          3. Compute crop indices: random if train, center if val/test.
          4. Crop, then resize to output_size, then pad in Z if needed.
          5. Normalize intensities to [0,1]: apply window [-500,1500].
          6. Apply strong or weak augmentations (for train).
          7. Return a dictionary with keys "volume", "mask", "volume_ori", and possibly "volume_weak".

        Returns:
          sample: dict where
            - "volume"     : tensor [1, H, W, D] (float32)
            - "mask"       : tensor [H, W, D] (int64) if labeled, else None
            - "volume_ori" : numpy crop before normalization (for debugging)
            - "volume_weak": (if unlabeled train) weakly augmented volume
        """
        sample = {}
        sub = self.subjects[idx]

        ############### load volume/mask/centers
        # 1. Load CT volume (assumes .npy format)
        ct_pattern = os.path.join(self.data_dir, "rawdata", sub, f"*_ct.npy")
        ct_files = glob.glob(ct_pattern)
        if len(ct_files) == 0:
            raise FileNotFoundError(f"No CT file found for subject {sub}, pattern: {ct_pattern}")
        # Some years may have multiple versions; pick the last one
        ct_path = ct_files[-1]
        volume = np.load(ct_path)

        # 2. Load mask if labeled
        mask = None
        if self.labeled:
            mask_pattern = os.path.join(self.data_dir, "derivatives", sub, f"*_msk.npy")
            mask_files = glob.glob(mask_pattern)
            if len(mask_files) > 0:
                mask_path = mask_files[-1]
                mask = np.load(mask_path)
            else:
                # If no mask found, mark as unlabeled
                self.labeled = False

        ############################# bounding box around the mask projection
        nx, ny, nz = volume.shape
        if mask is not None:
            # 1. Compute a tight bounding box around the mask projection
            x0, x1, y0, y1, z0, z1 = get_mask_proj_bbox(mask, margin_xy=0, margin_z=0)

            # 2. Determine random margins for train, fixed margins for val/test
            if self.split == 'train':
                margin_xy = random.randint(4, 24)
                margin_z = random.randint(4, 12)
            else:
                margin_xy = 12
                margin_z = 6

            # 3. Compute square side length in XY and half-side
            len_x = x1 - x0
            len_y = y1 - y0
            base_xy = min(len_x, len_y)
            side_xy = base_xy + 2 * margin_xy
            side_xy = min(side_xy, nx, ny)
            half_xy = side_xy // 2

            # Center of the bounding box
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2

            # 3a. Compute X-axis crop boundaries
            x_min = max(0, cx - half_xy)
            x_max = x_min + side_xy
            if x_max > nx:
                x_max = nx
                x_min = max(0, nx - side_xy)

            # 3b. Compute Y-axis crop boundaries
            y_min = max(0, cy - half_xy)
            y_max = y_min + side_xy
            if y_max > ny:
                y_max = ny
                y_min = max(0, ny - side_xy)

            # 4. Compute target depth proportional to XY size
            z_min = max(0, z0 - margin_z)
            z_max = min(nz, z1 + margin_z)
        else:
            # Fallback: random crop for train, center crop for val/test
            if self.split == 'train':
                x_min, x_max, y_min, y_max, z_min, z_max = get_random_crop_indices(
                    volume, crop_ratio=self.crop_ratio, output_size=self.output_size)
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = get_center_crop_indices(
                    volume, crop_ratio=self.crop_ratio, output_size=self.output_size)

        # ---------------- 裁剪前记录 offset ----------------
        volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        if mask is not None:
            mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]

        ############################# Pad
        # 计算裁剪范围
        h, w, d = volume.shape
        xy = max(h, w)
        desired_d = int(round(xy * self.output_size[-1] / self.output_size[0]))
        if d < desired_d:
            pad_total = desired_d - d
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            min_val = volume.min()
            volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)),
                            mode='constant', constant_values=min_val)
            if mask is not None:
                mask = np.pad(mask, ((0, 0), (0, 0), (pad_before, pad_after)),
                              mode='constant', constant_values=0)

        # Validation / test: no augmentations, just normalize and to tensor
        vol_high = to_tensor_3d(normalize_volume(volume))
        msk_high = torch.from_numpy(mask).long() if mask is not None else None
        sample["volume_high"] = vol_high
        sample["mask_high"] = msk_high
        sample["subject"] = sub
        sample["volume_ori"] = volume
        sample["mask_ori"] = mask

        # Resize to output size
        volume_resized = resize_volume(volume, target_shape=self.output_size, order=1)
        if mask is not None:
            mask_resized = resize_volume(mask, target_shape=self.output_size, order=0)

        # Validation / test: no augmentations, just normalize and to tensor
        vol_low = to_tensor_3d(normalize_volume(volume_resized))
        msk_low = torch.from_numpy(mask_resized).long() if mask is not None else None
        sample["volume_low"] = vol_low
        sample["mask_low"] = msk_low

        return sample


##########################################################################
# Volume Crop
##########################################################################
def load_centers_from_json(json_path: str,
                           round_to_int: bool = True
                           ) -> List[Tuple[int, int, int]]:
    """
    读取 VerSe 质心 JSON => [(x,y,z), ...]
    第一条通常是 direction 信息，跳过。
    """
    with open(json_path, 'r') as f:
        items = json.load(f)

    centers = []
    for it in items:
        if 'label' not in it:       # 跳过方向/其他 meta
            continue
        x, y, z = it['X'], it['Y'], it['Z']
        if round_to_int:
            centers.append(tuple(int(round(v)) for v in (x, y, z)))
        else:
            centers.append((x, y, z))
    return centers            # 按标签 1…25 的顺序


def get_mask_proj_bbox(mask: np.ndarray,
                       margin_xy: int = 16,
                       margin_z:  int = 8) -> tuple:
    """
    根据 mask 投影沿 Z 和 X 轴，返回裁剪框索引：
      mask: [X, Y, Z] 二值 Mask
      margin_xy: 在 XY 平面边缘外额外扩展像素数
      margin_z:  在 Z 方向边缘外额外扩展切片数

    Returns:
      x_min, x_max, y_min, y_max, z_min, z_max
    """
    X, Y, Z = mask.shape

    # 1) XY 平面投影：沿 Z 轴 any → 得到 [X,Y]
    proj_xy = mask.any(axis=2)  # [X,Y]
    xs, ys = np.where(proj_xy)
    if xs.size == 0 or ys.size == 0:
        # 全空 fallback
        return 0, X, 0, Y, 0, Z

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # 加 margin 并截边界
    x_min = max(0, x0 - margin_xy)
    x_max = min(X, x1 + margin_xy + 1)
    y_min = max(0, y0 - margin_xy)
    y_max = min(Y, y1 + margin_xy + 1)

    # 2) Z 轴投影：沿 X 轴 any → 得到 [Y,Z] 或沿 Y→ [X,Z], 任选其一
    #    这里我沿 X 轴投影
    proj_z = mask.any(axis=0).any(axis=0)  # mask.any(axis=0)→[Y,Z], 再 any→[Z]
    zs = np.where(proj_z)[0]
    if zs.size == 0:
        z_min, z_max = 0, Z
    else:
        z0, z1 = zs.min(), zs.max()
        z_min = max(0, z0 - margin_z)
        z_max = min(Z, z1 + margin_z + 1)

    return x_min, x_max, y_min, y_max, z_min, z_max



def get_center_crop_indices(volume: np.ndarray,
                            crop_ratio: float = 0.9,
                            output_size: Tuple[int,int,int] = (64, 64, 64)) -> Tuple[int,int,int,int,int,int]:
    """
    Compute center-crop indices for a 3D volume based on `crop_ratio`.
    This will crop a cube in the center whose XY side length is min(H,W)*crop_ratio,
    and Z side is scaled to match the output aspect ratio.

    Args:
      volume:      3D numpy array, shape = (H, W, D).
      crop_ratio:  fraction of the smaller XY side to crop (0 < crop_ratio <= 1).
      output_size: desired output size (h, w, d) after cropping/resizing.

    Returns:
      A 6-tuple of indices (x_min, x_max, y_min, y_max, z_min, z_max) for cropping.
    """
    H, W, D = volume.shape
    # Determine square crop in XY plane
    crop_xy = round(min(H, W) * crop_ratio)
    # Compute corresponding Z dimension to maintain output aspect ratio
    crop_z = min(round(crop_xy * output_size[-1] / output_size[0]), D)
    x_margin = int((H - crop_xy) / 2)
    y_margin = int((W - crop_xy) / 2)
    z_margin = int((D - crop_z) / 2)
    return x_margin, x_margin + crop_xy, y_margin, y_margin + crop_xy, z_margin, z_margin + crop_z


def get_random_crop_indices(volume: np.ndarray,
                            crop_ratio: float = 0.9,
                            output_size: Tuple[int,int,int] = (64, 64, 64)) -> Tuple[int,int,int,int,int,int]:
    """
    Compute random crop indices for a 3D volume based on `crop_ratio`.
    Similar to center crop but chooses a random starting point.

    Args:
      volume:      3D numpy array, shape = (H, W, D).
      crop_ratio:  fraction of the smaller XY side to crop.
      output_size: desired output size (h, w, d) after cropping/resizing.

    Returns:
      A 6-tuple of indices (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    H, W, D = volume.shape
    crop_xy = round(min(H, W) * crop_ratio)
    crop_z = min(round(crop_xy * output_size[-1] / output_size[0]), D)
    x_start = random.randint(0, max(0, H - crop_xy))
    y_start = random.randint(0, max(0, W - crop_xy))
    z_start = random.randint(0, max(0, D - crop_z))
    return x_start, x_start + crop_xy, y_start, y_start + crop_xy, z_start, z_start + crop_z


def compute_centroids_via_dt(mask: np.ndarray, num_classes: int) -> list[list[tuple[int,int,int]]]:
    """
    对每个样本（batch 维度外调用），对 mask 中 1..num_classes 类别分别：
      1) 取出该类别的最大连通域
      2) 在连通域上做 distance transform
      3) 找到距离场的 argmax，作为该椎体的中心
    返回形如 List[B][c=(z,y,x)] 的坐标列表。
    """
    centers_per_batch = []
    # 如果是单样本 mask，也就是 shape (H,W,D)
    # 外面循环 batch 之后再调用此函数
    # 下面只做单样本处理
    mask_np = mask
    H, W, D = mask_np.shape
    centers = []
    for c in range(1, num_classes+1):
        mask_c = (mask_np == c).astype(np.uint8)
        if mask_c.sum() == 0:
            # 如果整块里没有这个类别，就给个哑点 (-1,-1,-1)
            centers.append((-1,-1,-1))
            continue

        # label 连通域，取最大那一块
        labeled, n_comp = label(mask_c, structure=np.ones((3,3,3),dtype=int))
        if n_comp == 0:
            centers.append((-1,-1,-1))
            continue

        counts = np.bincount(labeled.flatten())
        counts[0] = 0  # 忽略背景
        largest_label = counts.argmax()
        lcc_mask = (labeled == largest_label)

        # distance transform
        dist = distance_transform_edt(lcc_mask)
        # 拿最大值点
        flat_idx = dist.argmax()
        y, x, z = np.unravel_index(flat_idx, dist.shape)
        centers.append((y, x, z))
    return centers


##########################################################################
# Data Augment
##########################################################################
def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    """
    Resample a 3D volume to `target_shape` using spline interpolation of given `order`.

    Args:
      volume:      3D numpy array, shape = (H, W, D).
      target_shape: desired output shape (h, w, d).
      order:       interpolation order (0=nearest, 1=linear, etc.).

    Returns:
      Resized volume of shape target_shape.
    """
    current_shape = volume.shape
    # Compute zoom factors for each dimension
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    return scipy.ndimage.zoom(volume, zoom=zoom_factors, order=order)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize CT intensities to [0, 1] using a window of [-500, 1500].
    Voxels below -500 are clipped to -500, above 1500 to 1500.

    Then map [-500, 1500] -> [0, 1].
    """
    vol = np.clip(volume, -500, 1500)
    vol = (vol + 500) / 2000.0
    return np.clip(vol, 0, 1)


def denormalize(vol_norm: np.ndarray) -> np.ndarray:
    """
    Inverse of normalize_volume: map [0, 1] -> original window [-500, 1500].
    Useful for debugging or saving back to original scale.
    """
    vol = vol_norm * 2000.0 - 500.0
    return vol


def weak_augment_3d_wo_pair(volume: np.ndarray) -> np.ndarray:
    """
    Apply weak 3D augmentations to a volume:
      1. Random flips along each axis with probability 0.5
      2. Random rotation in the XY plane by an angle in [-22.5°, 22.5°]
    Input `volume` is assumed to be normalized in [0,1].
    """
    vol = copy.deepcopy(volume)
    # Random flips along X, Y, Z
    if random.random() > 0.5:
        vol = np.flip(vol, axis=0)
    if random.random() > 0.5:
        vol = np.flip(vol, axis=1)
    if random.random() > 0.75:
        vol = np.flip(vol, axis=2)

    # Random rotation about Z axis (rotate in XY plane)
    angle = random.uniform(-22.5, 22.5)
    # order=1 for linear interpolation
    vol = scipy.ndimage.rotate(vol, angle, axes=(0, 1), reshape=False, order=1)

    return np.clip(vol, 0, 1)


def strong_augment_3d_wo_pair(volume: np.ndarray,
                      p_scale: float = 0.5,
                      p_gamma: float = 0.5,
                      p_gauss_noise: float = 0.5,
                      p_sp_noise: float = 0.3,
                      p_poisson: float = 0.3,
                      p_speckle: float = 0.3,
                      p_gauss_blur: float = 0.5,
                      p_median: float = 0.5,
                      p_unsharp: float = 0.5,
                      p_clahe: float = 0.5) -> np.ndarray:
    """
    Apply a series of photometric and noise-based augmentations to a 3D volume.
    This is considered "strong" augmentation (for semi-supervised or contrastive tasks).
    It does NOT change spatial coordinates (except blur, rotate, etc.). Steps:
      1. Random intensity scaling and shifting (contrast/brightness)
      2. Random gamma correction
      3. Additive Gaussian noise
      4. Salt-and-pepper noise
      5. Poisson noise (photon noise)
      6. Speckle (multiplicative) noise
      7. Gaussian blur
      8. Median filter
      9. Unsharp masking (sharpen)
      10. CLAHE (adaptive histogram equalization) slice by slice

    Args:
      volume:        input 3D volume (values in [0,1])
      p_*:           probability of applying each augmentation
      cutout_ratio:  fraction of each dimension to mask out for cutout

    Returns:
      Augmented volume clipped to [0,1].
    """
    vol = copy.deepcopy(volume).astype(np.float32)
    H, W, D = vol.shape

    # 1. Random intensity scaling + brightness shift
    if random.random() < p_scale:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-0.1, 0.1)
        vol = np.clip(vol * alpha + beta, 0, 1)

    # 2. Random gamma correction
    if random.random() < p_gamma:
        gamma = random.uniform(0.7, 1.5)
        vol = np.clip(vol ** gamma, 0, 1)

    # 3. Add Gaussian noise
    if random.random() < p_gauss_noise:
        sigma = random.uniform(0.0, 0.025)
        vol = np.clip(vol + np.random.normal(0, sigma, size=vol.shape), 0, 1)

    # 4. Salt-and-pepper noise
    if random.random() < p_sp_noise:
        prob = random.uniform(0.001, 0.01)
        rnd = np.random.rand(H, W, D)
        vol[rnd < (prob / 2)] = 0.0
        vol[rnd > 1 - (prob / 2)] = 1.0

    # 5. Poisson noise
    if random.random() < p_poisson:
        vals = len(np.unique(vol))
        vol = np.clip(np.random.poisson(vol * vals) / float(vals), 0, 1)

    # 6. Speckle (multiplicative) noise
    if random.random() < p_speckle:
        sigma = random.uniform(0.0, 0.05)
        vol = np.clip(vol + vol * np.random.normal(0, sigma, size=vol.shape), 0, 1)

    # 7. Gaussian blur
    if random.random() < p_gauss_blur:
        sigma = random.uniform(0.75, 1.25)
        vol = ndi.gaussian_filter(vol, sigma=sigma)

    # 8. Median filter
    if random.random() < p_median:
        vol = ndi.median_filter(vol, size=3)

    # 9. Unsharp masking (sharpen)
    if random.random() < p_unsharp:
        blurred = ndi.gaussian_filter(vol, sigma=1.0)
        amount = random.uniform(0.5, 1.5)
        vol = np.clip(vol + amount * (vol - blurred), 0, 1)

    # 10. CLAHE (slice-by-slice)
    if random.random() < p_clahe:
        for z in range(D):
            # Apply 2D CLAHE to each XY slice
            vol[..., z] = exposure.equalize_adapthist(vol[..., z], clip_limit=0.03)

    return np.clip(vol, 0, 1)


def strong_augment_3d_w_pair(volume: np.ndarray,
                             mask: np.ndarray,
                             centers: Optional[np.ndarray] = None,
                             our: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply consistent strong augmentation to both volume and mask simultaneously:
      - Random flips along each axis
      - Random rotation in XY plane
      - For volume only: either strong_augment_3d or slight intensity scaling
      - For mask: only spatial transformations (flip, rotate)

    Args:
      volume:  normalized 3D volume (0..1)
      mask:    corresponding 3D mask (integer labels)
      our:     if True, apply full strong_augment_3d to volume; otherwise apply milder adjustments.

    Returns:
      (augmented_volume, augmented_mask), both same shape as input.
    """
    vol = copy.deepcopy(volume)
    msk = copy.deepcopy(mask)
    cen = copy.deepcopy(centers)

    H, W, D = vol.shape
    # 注意：中心坐标使用体素 index，非 mm

    # Random flips along X, Y, Z for both volume & mask
    if random.random() > 0.5:
        vol = np.flip(vol, axis=0)
        msk = np.flip(msk, axis=0)
        if cen is not None:
            valid = cen[:, 0] >= 0
            cen[valid, 0] = (H - 1) - cen[valid, 0]
    if random.random() > 0.5:
        vol = np.flip(vol, axis=1)
        msk = np.flip(msk, axis=1)
        if cen is not None:
            valid = cen[:, 1] >= 0
            cen[valid, 1] = (W - 1) - cen[valid, 1]
    if random.random() > 0.75:
        vol = np.flip(vol, axis=2)
        msk = np.flip(msk, axis=2)
        if cen is not None:
            valid = cen[:, 2] >= 0
            cen[valid, 2] = (D - 1) - cen[valid, 2]

    # Random rotation in XY plane for both volume and mask
    angle = random.uniform(-22.5, 22.5)
    vol = scipy.ndimage.rotate(vol, angle, axes=(0, 1), reshape=False, order=1)
    msk = scipy.ndimage.rotate(msk, angle, axes=(0, 1), reshape=False, order=0)
    if cen is not None:
        # 以图像中心为原点旋转
        rad = math.radians(angle)
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        cx, cy = (H - 1) / 2.0, (W - 1) / 2.0
        valid = cen[:, 0] >= 0
        x = cen[valid, 0] - cx
        y = cen[valid, 1] - cy
        cen[valid, 0] = cos_t * x - sin_t * y + cx
        cen[valid, 1] = sin_t * x + cos_t * y + cy
        # z 坐标不变

    # If `our` flag is True, apply full strong augmentation to volume
    if random.random() > 0.5 and our:
        vol = strong_augment_3d_wo_pair(vol)
    else:
        # Otherwise apply slight intensity scaling only
        if random.random() > 0.5:
            factor = random.uniform(0.95, 1.05)
            vol = vol * factor
        vol = np.clip(vol, 0, 1)

    if cen is None:
        return vol.copy(), msk.copy()
    else:
        return vol.copy(), msk.copy(), cen.copy()


def to_tensor_3d(volume: np.ndarray) -> torch.Tensor:
    """
    Convert a 3D numpy array to a PyTorch tensor of shape (1, H, W, D) and dtype float32.

    Args:
      volume: 3D numpy array, shape = (H, W, D).

    Returns:
      4D torch tensor with channel dimension added.
    """
    tensor = torch.from_numpy(volume.astype(np.float32))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor

