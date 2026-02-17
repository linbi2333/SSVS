#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py — Inference and evaluation pipeline for 3D U-Net vertebra segmentation and classification.

This script loads a trained UNet3D model, runs inference on a test dataset,
outputs NIfTI files for predicted instances and nearest-centroid assignments,
and computes segmentation and classification metrics.

Main functions:
  - save_nii: save NumPy arrays as .nii.gz with affine metadata.
  - predict_and_eval: perform batch inference, save prediction and ground truth,
                     and calculate Dice score and classification metrics.

Usage:
  python test.py

Dependencies:
  torch, numpy, nibabel, networks.UNet3D, dataset.VertebraDataset3D, utils...
"""

import os
import glob
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from networks import UNet3D
from dataset import compute_n_classes, VertebraDataset3D, \
    VertebraDataset3DHigh, VertebraDataset3DLowHigh
from utils import (
    extract_largest_cc_per_class,
    compute_classification_metrics,
    split_stuck_vertebra,
    sliding_window_inference,
    split_vertebra_cc,
    suppress_close_seeds,
    refine_vertebra_inst
)


def save_nii(
    image_array: np.ndarray,
    out_path: str,
    affine: np.ndarray = None
) -> None:
    """
    Save a 3D NumPy array as a compressed NIfTI (.nii.gz) file.

    Parameters
    ----------
    image_array : np.ndarray
        3D array of voxel intensities or labels.
    out_path : str
        Destination file path ending with .nii.gz.
    affine : np.ndarray, optional
        4x4 affine matrix for spatial metadata. If None, uses default axes:
        +X = Right, -Y = Posterior, -Z = Inferior.
    """
    # Use identity-like affine if none provided
    if affine is None:
        affine = np.array([
            [1,  0,  0, 0],  # +X axis maps to Right
            [0, -1,  0, 0],  # -Y axis maps to Posterior
            [0,  0, -1, 0],  # -Z axis maps to Inferior
            [0,  0,  0, 1],
        ], dtype=float)

    # Create and save NIfTI image
    nii_img = nib.Nifti1Image(image_array, affine)
    nib.save(nii_img, out_path)
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# Inference, saving, and metric computation
# -----------------------------------------------------------------------------
def predict_and_eval(
    model: UNet3D,
    test_loader: DataLoader,
    device: torch.device,
    output_dir_pred: str,
    output_dir_gt: str = None,
    output_dir_input: str = None,
    top_k: int = 5
) -> (float, dict):
    """
    Perform inference on test data, save predictions/G T/input as NIfTI,
    and compute segmentation Dice and classification metrics.

    Parameters
    ----------
    model : UNet3D
        Trained 3D U-Net model instance. forward(x) returns
        (cls_hmap, seg_logits, _, _).
    test_loader : DataLoader
        DataLoader yielding batches from VertebraDataset3D.
    device : torch.device
        Computation device (CPU or CUDA).
    output_dir_pred : str
        Directory to save predicted instance and Euclidean maps.
    output_dir_gt : str, optional
        Directory to save ground truth masks as NIfTI.
    output_dir_input : str, optional
        Directory to save original input CT volumes as NIfTI.
    top_k : int
        Number of samples with highest classification F1 to print.

    Returns
    -------
    seg_dice : float
        Average binary Dice score over all test samples.
    cls_metrics : dict
        Classification counts and aggregated Precision, Recall, Accuracy, F1.
    """
    # Ensure output directories exist
    os.makedirs(output_dir_pred, exist_ok=True)
    if output_dir_gt:
        os.makedirs(output_dir_gt, exist_ok=True)
    if output_dir_input:
        os.makedirs(output_dir_input, exist_ok=True)

    model.eval()  # Set model to evaluation mode
    seg_dices: list = []  # Collect per-batch Dice scores
    per_f1: list = []     # Collect per-sample classification F1 scores

    # Initialize classification counters
    TP = FP = FN = TN = 0
    eps = 1e-6  # Small constant to avoid division by zero

    with torch.no_grad():
        for batch in tqdm(test_loader,
                          desc="Inference",
                          total=len(test_loader),
                          unit="batch"):
            # Move inputs to device
            x = batch["volume"].to(device)      # shape [B,1,H,W,D]
            y_t = batch["mask"].to(device).long()  # shape [B,H,W,D]
            subs = batch["subject"]  # List of subject IDs for this batch
            B, H, W, D = y_t.shape

            # Model forward pass
            cls_hmap, seg_logits, _, _ = model(x)
            # seg_logits shape: [B,2,H/?,W/?,D/?]

            # ---------- 1) Compute binary Dice for foreground ----------
            # Convert logits to foreground probability via softmax
            prob_fg = F.softmax(seg_logits, dim=1)[:, 1]  # [B,H',W',D']
            # Binarize ground truth mask (>0 -> foreground)
            gt_bin = (y_t > 0).float()
            # Intersection and union for each sample
            inter = (prob_fg * gt_bin).sum(dim=(1, 2, 3))
            card = prob_fg.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))
            dice_scores = ((2 * inter + eps) / (card + eps)).cpu().tolist()
            seg_dices.extend(dice_scores)

            # ---------- 2) Upsample classification heatmap to full resolution ----------
            cls_up = F.interpolate(
                cls_hmap,
                size=(H, W, D),
                mode='nearest'
            )  # [B,C,H,W,D]
            cls_np = cls_up.cpu().numpy()
            y_np = y_t.cpu().numpy()

            # ---------- 3) Per-sample post-processing ----------
            for b in range(B):
                subj = subs[b]
                gt_vol = y_np[b]    # ground truth mask [H,W,D]
                heat = cls_np[b]    # classification scores [C,H,W,D]
                fg_mask = (prob_fg[b] > 0.5).cpu().numpy()  # binary foreground mask

                # 3a) Initial ID map: voxel-wise argmax class times fg_mask
                id_map = (np.argmax(heat, axis=0).astype(np.uint8) * fg_mask)

                # 3b) Instance segmentation via marker-based watershed
                labels_inst = split_stuck_vertebra(fg_mask, id_map)

                # 3c) Nearest-centroid assignment fallback
                centers, cls_ids = [], []
                C = heat.shape[0]
                for c in range(1, C):  # skip background channel
                    # Multiply class heatmap by foreground mask
                    h_c = heat[c] * fg_mask
                    flat = h_c.ravel()
                    arg = flat.argmax()
                    if flat[arg] <= 1e-3:
                        continue  # no activation for this class
                    x_idx, y_idx, z_idx = np.unravel_index(arg, (H, W, D))
                    centers.append((x_idx, y_idx, z_idx))
                    cls_ids.append(c)

                if centers:
                    # Compute squared distances to each centroid for all voxels
                    ctrs = np.array(centers)
                    ids = np.array(cls_ids)
                    idxs = np.indices((H, W, D))  # shape (3,H,W,D)
                    dist2 = ((idxs[0][None] - ctrs[:, 0, None, None, None])**2 +
                             (idxs[1][None] - ctrs[:, 1, None, None, None])**2 +
                             (idxs[2][None] - ctrs[:, 2, None, None, None])**2)
                    nearest = dist2.argmin(axis=0)  # [H,W,D]
                    labels_euc = (ids[nearest] * fg_mask).astype(np.uint8)

                    labels_euc = np.zeros_like(fg_mask, dtype=np.uint8)
                    block = 3
                    for (x, y, z), cid in zip(ctrs, ids):
                        labels_euc[max(0, x - block):x + block + 1,
                        max(0, y - block):y + block + 1,
                        max(0, z - block):z + block + 1] = cid

                else:
                    # If no centroids, fallback to initial id_map
                    labels_euc = id_map.copy()

                # ---------- 4) Save NIfTI outputs ----------
                # Instance map
                save_nii(
                    labels_inst,
                    os.path.join(output_dir_pred, f"{subj}_inst.nii.gz")
                )
                # Euclidean (nearest-centroid) map
                save_nii(
                    labels_euc,
                    os.path.join(output_dir_pred, f"{subj}_euc.nii.gz")
                )
                # Segmentation binary threshold at 0.9
                seg_bin = (prob_fg[b] >= 0.9).cpu().numpy().astype(np.uint8)
                save_nii(
                    seg_bin,
                    os.path.join(output_dir_pred, f"{subj}_seg.nii.gz")
                )

                # Optionally save ground truth and input volume
                if output_dir_gt:
                    if not os.path.exists(os.path.join(output_dir_gt, f"{subj}_gt.nii.gz")):
                        save_nii(
                            gt_vol.astype(np.uint8),
                            os.path.join(output_dir_gt, f"{subj}_gt.nii.gz")
                        )
                if output_dir_input:
                    # Original CT before normalization
                    if not os.path.exists(os.path.join(output_dir_input, f"{subj}_input.nii.gz")):
                        input_vol = batch["volume_ori"][b].cpu().numpy().astype(np.int16)
                        save_nii(
                            input_vol,
                            os.path.join(output_dir_input, f"{subj}_input.nii.gz")
                        )

                # ---------- 5) Classification metrics per sample ----------
                cc_p = extract_largest_cc_per_class(labels_inst)
                cc_g = extract_largest_cc_per_class(gt_vol)
                m = compute_classification_metrics(cc_p, cc_g, iou_thresh=0.1)
                TP += m['TP']
                FP += m['FP']
                FN += m['FN']
                TN += m['TN']
                prec = m['TP'] / (m['TP'] + m['FP'] + eps)
                rec = m['TP'] / (m['TP'] + m['FN'] + eps)
                f1 = 2 * prec * rec / (prec + rec + eps)
                per_f1.append((subj, f1))

    # ---------- Summarize metrics ----------
    seg_dice = float(np.mean(seg_dices))
    P = TP / (TP + FP + eps)
    R = TP / (TP + FN + eps)
    A = (TP + TN) / (TP + FP + FN + TN + eps)
    F1 = 2 * P * R / (P + R + eps)

    # Print top-K samples by classification F1
    topk = sorted(per_f1, key=lambda x: x[1], reverse=True)[:top_k]
    print(f"\nTop {top_k} classification samples by F1:")
    for i, (s, score) in enumerate(topk, start=1):
        print(f" {i}. {s}, F1={score:.4f}")

    # Print overall metrics
    print(
        f"\nTest Dice (binary) = {seg_dice:.4f}  "
        f"Classification Acc = {A:.4f}  "
        f"Prec = {P:.4f}  "
        f"Rec = {R:.4f}  "
        f"F1 = {F1:.4f}"
    )

    # Return segmentation and classification metrics
    return seg_dice, {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Precision': P, 'Recall': R, 'Accuracy': A, 'F1': F1
    }


def predict_and_eval_lowhigh(
    model: UNet3D,
    test_loader: DataLoader,
    device: torch.device,
    output_dir_pred: str,
    output_dir_gt: str = None,
    output_dir_input: str = None,
    top_k: int = 5
) -> (float, dict):
    """
    Perform inference on test data, save predictions/G T/input as NIfTI,
    and compute segmentation Dice and classification metrics.

    Parameters
    ----------
    model : UNet3D
        Trained 3D U-Net model instance. forward(x) returns
        (cls_hmap, seg_logits, _, _).
    test_loader : DataLoader
        DataLoader yielding batches from VertebraDataset3D.
    device : torch.device
        Computation device (CPU or CUDA).
    output_dir_pred : str
        Directory to save predicted instance and Euclidean maps.
    output_dir_gt : str, optional
        Directory to save ground truth masks as NIfTI.
    output_dir_input : str, optional
        Directory to save original input CT volumes as NIfTI.
    top_k : int
        Number of samples with highest classification F1 to print.

    Returns
    -------
    seg_dice : float
        Average binary Dice score over all test samples.
    cls_metrics : dict
        Classification counts and aggregated Precision, Recall, Accuracy, F1.
    """
    """
        Sliding-window segmentation on the *high-res* volume + global classification
        on the *low-res* volume.  The two predictions are fused into an instance
        map via `split_vertebra_cc_memsave`.
        """
    # ---------------- hyper-params -----------------
    ROI_SIZE = (64, 64, 128)  # sliding window size
    OVERLAP = 0.25  # window overlap ratio
    FG_THR = 0.85  # foreground prob threshold
    SEED_THR = 0.60  # heat threshold for a valid seed
    LAM_Z = 4.0  # axial distance weight in CC split
    CUBE_R = 6  # half-edge of centroid cube (=> 5³ if 2)
    IOU_THR = 0.10  # IoU threshold for CC metrics
    # -----------------------------------------------

    os.makedirs(output_dir_pred, exist_ok=True)
    if output_dir_gt:
        os.makedirs(output_dir_gt, exist_ok=True)
    if output_dir_input:
        os.makedirs(output_dir_input, exist_ok=True)

    model.eval()
    eps = 1e-6
    seg_dices: list[float] = []
    per_f1: list[tuple[str, float]] = []
    TP = FP = FN = TN = 0

    for batch in tqdm(test_loader, desc="Inference", unit="batch"):
        vol_hi: torch.Tensor = batch["volume_high"].to(device)  # for seg
        vol_lo: torch.Tensor = batch["volume_low"].to(device)  # for cls
        y_gt: torch.Tensor | None = batch.get("mask_high")
        if y_gt is not None:
            y_gt = y_gt.to(device)
        subs: list[str] = batch["subject"]

        B, _, Hh, Wh, Dh = vol_hi.shape

        # ---------- 1) Sliding-window segmentation ----------
        cls_full, seg_logits = sliding_window_inference(
            vol=vol_hi,
            model=model,
            win_size=ROI_SIZE,
            batch_size=16,
            overlap=OVERLAP,
            device=device,
        )  # cls_full unused here (kept for completeness)

        prob_fg = torch.softmax(seg_logits, dim=1)[:, 1]  # [B,Hh,Wh,Dh]

        # Binary Dice
        if y_gt is not None:
            gt_bin = (y_gt > 0).float()
            inter = (prob_fg * gt_bin).sum((1, 2, 3))
            card = prob_fg.sum((1, 2, 3)) + gt_bin.sum((1, 2, 3))
            seg_dices += ((2 * inter + eps) / (card + eps)).cpu().tolist()

        # ---------- 2) Low-res classification ----------
        cls_hmap_lo, *_ = model(vol_lo)  # [B,C,Hl,Wl,Dl]
        cls_up = F.interpolate(
            cls_hmap_lo, size=(Hh, Wh, Dh), mode="nearest"
        ).detach().cpu().numpy()  # [B,C,Hh,Wh,Dh]

        fg_np = (prob_fg > FG_THR).cpu().numpy()  # bool mask
        gt_np = None if y_gt is None else y_gt.cpu().numpy()
        C = cls_up.shape[1]

        # ---------- 3) Per-sample fusion ----------
        for b in range(B):
            subj = subs[b]
            heat = cls_up[b]  # (C,Hh,Wh,Dh)
            fg = fg_np[b].astype(bool)
            fg_mask = fg.astype(np.uint8)
            gt_vol = None if gt_np is None else gt_np[b]

            # ---- seeds with axial-NMS ----
            seeds_xyz, seeds_val, seeds_cid = [], [], []
            for cid in range(1, C):  # skip background (0)
                h_c = heat[cid] * fg  # suppress outside FG
                idx = int(h_c.argmax())
                peak = float(h_c.flat[idx])
                if peak < SEED_THR:  # too weak → discard
                    continue
                xi, yi, zi = np.unravel_index(idx, (Hh, Wh, Dh))
                seeds_xyz.append((xi, yi, zi))
                seeds_val.append(peak)
                seeds_cid.append(cid)  # original channel id

            # Z-axis non-max suppression + 连续编号
            kept_xyz, kept_cls = suppress_close_seeds(
                seeds_xyz,  # candidates
                seeds_val,  # confidence
                seeds_cid,  # original class id
                dz=8  # half-window in slices
            )

            # ---------- tiny cubes (Euclidean map) ----------
            # 可视化使用大CUBE_R
            id_map = np.zeros((Hh, Wh, Dh), np.uint8)
            for (x, y, z), cls_id in zip(kept_xyz, kept_cls):
                id_map[
                max(0, x - CUBE_R):min(Hh, x + CUBE_R + 1),
                max(0, y - CUBE_R):min(Wh, y + CUBE_R + 1),
                max(0, z - CUBE_R):min(Dh, z + CUBE_R + 1)
                ] = cls_id
            id_map *= fg_mask  # keep foreground only

            # ---------- 3.c 由外部种子驱动的 CC-split ----------
            inst_map = split_vertebra_cc(
                fg=fg,
                heat=heat,
                lam_z=LAM_Z,
                dist_thr=16.0,  # 依据体素大小可调
                seeds_xyz=kept_xyz,  # <<<<<< 传入
                seeds_cls=kept_cls  # <<<<<< 传入
            )

            volume = batch['volume_high'][b].squeeze().cpu().numpy().astype(np.float32)
            cls_prob = torch.softmax(torch.Tensor(cls_up[b]), dim=0)  # (C,H,W,D), C=26(0=bg, 1-25=v1…v25)
            cls_id = cls_prob.argmax(dim=0).cpu().numpy()  # hard prediction → one-hot
            cls_id *= fg_mask  # keep only foreground
            onehot = np.eye(C, dtype=np.uint8)[cls_id]  # (H,W,D,C)  one-hot
            onehot = onehot.transpose(3, 0, 1, 2)  # → (C,H,W,D)
            inst_map_ref = refine_vertebra_inst(
                volume,  # 简单归一化
                onehot,  # 或保存好的 softmax
                inst_map,  # 你的实例分割
                iterations=96,
            )
            inst_map_ref *= fg.astype(np.uint8)

            # ---- 3.d centroid visualisation ----
            if seeds_xyz:
                centroid_pts = np.zeros_like(fg, dtype=np.uint8)
                centroid_cube = np.zeros_like(fg, dtype=np.uint8)
                for (x, y, z) in seeds_xyz:
                    centroid_pts[x, y, z] = 1
                    centroid_cube[
                    max(0, x - CUBE_R):min(Hh, x + CUBE_R + 1),
                    max(0, y - CUBE_R):min(Wh, y + CUBE_R + 1),
                    max(0, z - CUBE_R):min(Dh, z + CUBE_R + 1),
                    ] = 1

            # ---- 3.e save NIfTI outputs ----
            save_nii(inst_map,
                     os.path.join(output_dir_pred, f"{subj}_inst.nii.gz"))
            save_nii(inst_map_ref,
                     os.path.join(output_dir_pred, f"{subj}_inst_ref.nii.gz"))
            save_nii(id_map,
                     os.path.join(output_dir_pred, f"{subj}_euc.nii.gz"))
            seg_bin = (prob_fg[b] >= 0.9).cpu().numpy().astype(np.uint8)
            save_nii(seg_bin,
                     os.path.join(output_dir_pred, f"{subj}_seg.nii.gz"))

            if output_dir_gt and (gt_vol is not None):
                pth = os.path.join(output_dir_gt, f"{subj}_gt.nii.gz")
                if not os.path.exists(pth):
                    save_nii(gt_vol.astype(np.uint8), pth)

            if output_dir_input:
                pth = os.path.join(output_dir_input, f"{subj}_ct.nii.gz")
                if not os.path.exists(pth):
                    ct_ori = batch["volume_ori"][b].cpu().numpy().astype(np.int16)
                    save_nii(ct_ori, pth)

            # ---- 3.f classification metrics ----
            if gt_vol is not None:
                cc_p = extract_largest_cc_per_class(inst_map_ref)
                cc_g = extract_largest_cc_per_class(gt_vol.squeeze())
                m = compute_classification_metrics(cc_p, cc_g, iou_thresh=IOU_THR)
                TP += m["TP"]
                FP += m["FP"]
                FN += m["FN"]
                TN += m["TN"]
                prec = m["TP"] / (m["TP"] + m["FP"] + eps)
                rec = m["TP"] / (m["TP"] + m["FN"] + eps)
                per_f1.append((subj, 2 * prec * rec / (prec + rec + eps)))

    # ---------------- summary ----------------
    seg_dice = float(np.mean(seg_dices)) if seg_dices else 0.0
    Prec = TP / (TP + FP + eps)
    Rec = TP / (TP + FN + eps)
    Acc = (TP + TN) / (TP + FP + FN + TN + eps)
    F1 = 2 * Prec * Rec / (Prec + Rec + eps)

    topk = sorted(per_f1, key=lambda x: x[1], reverse=True)[:top_k]
    print(f"\nTop-{top_k} classification samples (F1):")
    for rk, (sid, score) in enumerate(topk, 1):
        print(f"{rk:2d}. {sid:20s}  F1={score:.4f}")

    print(
        f"\nTest Dice(bin)={seg_dice:.4f}  "
        f"Cls Acc={Acc:.4f}  Prec={Prec:.4f}  "
        f"Rec={Rec:.4f}  F1={F1:.4f}"
    )

    return seg_dice, {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "Precision": Prec, "Recall": Rec, "Accuracy": Acc, "F1": F1,
    }






# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Select device: CUDA if available else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Build validation dataset and DataLoader
    dataset_root = 'I:/Feng_jiasen/Graduations_Datas/VerSe/VerSe_2020_s_b_d_s_v3'
    ckpt_path = "./checkpoints/best_model_3d_unimatch.pth"
    # dataset_root = 'G:/Projectsss/External_projects/Feng_jiasen/graduations-vertebrae/Datasets/VerSe/VerSe_2020_s_b_d_s_v3'
    sample_size = (64, 64, 128)  # (H, W, D)
    test_dataset = VertebraDataset3DLowHigh(
        root_dir=dataset_root,
        split='all',
        output_size=sample_size,
        labeled=True,
        debug=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    # 3) Load pretrained model checkpoint
    max_label, cls_counts = compute_n_classes(dataset_root)
    n_classes = max_label + 1  # include background class
    model = UNet3D(
        n_channels=1,
        n_seg_classes=2,
        n_vert_classes=n_classes,
        base_filters=32
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        print(f"=> Loaded pretrained model weights from {ckpt_path}")
        if missing:
            print("   Missing keys:", missing)
        if unexpected:
            print("   Unexpected keys:", unexpected)
    else:
        raise KeyError(f"Checkpoint file {ckpt_path} does not contain a 'model' key")
    model.to(device)

    # 4) Define output directories for predictions, ground truth, and inputs
    output_dir_pred = os.path.join(dataset_root, "predictions_nii")
    output_dir_gt   = os.path.join(dataset_root, "gt_nii")
    output_dir_input= os.path.join(dataset_root, "inputs_nii")

    # 5) Run inference, saving outputs and computing metrics
    seg_dice, cls_metrics = predict_and_eval_lowhigh(
        model,
        test_loader,
        device,
        output_dir_pred,
        output_dir_gt,
        output_dir_input
    )
