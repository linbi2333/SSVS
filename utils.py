#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — Utility functions for mask processing, heatmap generation,
connected component extraction, visualization, and evaluation metrics.

Includes:
  - compute_centroids_from_mask: extract centroids of labeled regions in a batch of masks.
  - make_gaussian_heatmap: generate Gaussian heatmaps from centroids.
  - extract_largest_cc_per_class: isolate largest connected component per class in 3D ID map.
  - visualize_batch_heatmap_overlay: overlay heatmaps on CT slices and save PNGs.
  - split_stuck_vertebra: split touching vertebrae via marker-based watershed.
  - compute_classification_metrics: compute TP/FP/FN/TN based on IoU threshold.

Dependencies: torch, numpy, scipy, scikit-image, matplotlib
"""
import os
import torch
import numpy as np
import networkx as netx
import torch.nn.functional as F
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from itertools import product
from typing import Union, Dict, Tuple, List, Optional

from skimage.segmentation import watershed
from skimage.morphology import skeletonize_3d, ball, opening, remove_small_holes
from skimage.segmentation import morphological_geodesic_active_contour as mgac

from scipy.ndimage import label, distance_transform_edt, binary_closing, \
    binary_opening, binary_erosion, binary_dilation, gaussian_gradient_magnitude

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    create_pairwise_gaussian,
    create_pairwise_bilateral,
)


# -----------------------------------------------------------------------------
# 推理预测与定量指标计算
# -----------------------------------------------------------------------------
def compute_classification_metrics(
    cc_pred: Dict[int, np.ndarray],
    cc_gt: Dict[int, np.ndarray],
    iou_thresh: float = 0.1
) -> Dict[str, int]:
    """
    Compute TP/FP/FN/TN counts based on IoU threshold per class.

    Parameters
    ----------
    cc_pred : dict
        Predicted largest component masks {class_id: binary_mask}.
    cc_gt : dict
        Ground-truth largest component masks {class_id: binary_mask}.
    iou_thresh : float, optional
        Minimum IoU to count as true positive.

    Returns
    -------
    metrics : dict
        Dictionary with keys 'TP', 'FP', 'FN', 'TN'.
    """
    TP = FP = FN = TN = 0
    all_ids = set(cc_pred.keys()) | set(cc_gt.keys())
    for c in all_ids:
        pred_mask = cc_pred.get(c)
        gt_mask = cc_gt.get(c)
        # Case: class absent in GT
        if gt_mask is None or gt_mask.sum() == 0:
            if pred_mask is not None and pred_mask.sum() > 0:
                FP += 1
            else:
                TN += 1
            continue
        # Case: present in GT, but missing in prediction
        if pred_mask is None or pred_mask.sum() == 0:
            FN += 1
            continue
        # Compute IoU
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = inter / (union + 1e-6)
        if iou >= iou_thresh:
            TP += 1
        else:
            FN += 1
    return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}



@torch.inference_mode()
def sliding_window_inference(
    vol:   torch.Tensor,          # [1,C_in,H,W,D]  or  [C_in,H,W,D]
    model,
    win_size:   tuple,           # (ph,pw,pd)
    overlap:    float  = 0.5,
    batch_size: int    = 4,
    device:     torch.device = torch.device('cuda'),
    cls_scale:  int    = 4       # ↓stride of classification head (e.g. 4 ⇒ 1/4 res)
):
    """
    Sliding-window inference for a dual-head model:
      cls_hmap, seg_logits, _, _ = model(patch)

    Returns
    -------
    seg_full  : [1,C_seg,H,W,D]    –  stitched segmentation logits
    cls_full  : [1,C_cls,H,W,D]    –  stitched classification heat-map (upsampled)
    """
    # ---------- input shape & window stride ----------
    if vol.dim() == 4:                   # [C,H,W,D] → [1,C,H,W,D]
        vol = vol.unsqueeze(0)
    _, C_in, H, W, D = vol.shape
    ph, pw, pd = win_size
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))
    sd = max(1, int(pd * (1 - overlap)))

    # ------------- pre-compute start indices -------------
    hs = list(range(0, max(H - ph, 0) + 1, sh)) + [max(H - ph, 0)]
    ws = list(range(0, max(W - pw, 0) + 1, sw)) + [max(W - pw, 0)]
    ds = list(range(0, max(D - pd, 0) + 1, sd)) + [max(D - pd, 0)]
    coords = [(h0, w0, d0) for h0, w0, d0 in product(hs, ws, ds)]

    # ---------- allocate stitched canvases ----------
    model = model.to(device).eval()
    with torch.no_grad():
        # quick dry-run to know channel数
        _cls, _seg, *_ = model(vol[:, :, :ph, :pw, :pd].to(device))
        C_cls = _cls.shape[1]
        C_seg = _seg.shape[1]

    seg_full = torch.zeros((1, C_seg, H, W, D), device=device)
    cls_full = torch.zeros((1, C_cls, H, W, D), device=device)
    weight   = torch.zeros((1, 1,  H, W, D), device=device)
    w_patch  = torch.ones ((1, 1,  ph, pw, pd), device=device)   # 可以换成高斯窗

    # ------------ sliding window loop -------------
    for i in range(0, len(coords), batch_size):
        sub_coords = coords[i:i + batch_size]

        patches = torch.cat([
            vol[:, :, h:h+ph, w:w+pw, d:d+pd] for (h, w, d) in sub_coords
        ], 0).to(device)                                           # [B,C_in,ph,pw,pd]

        cls_patch, seg_patch, *_ = model(patches)                  # cls:[B,Cc,ph/4,...]

        # ------ upsample cls_patch 到 seg 分辨率 ------
        cls_patch_up = F.interpolate(
            cls_patch, size=(ph, pw, pd), mode='nearest'
        )

        # ------ 累加到全幅 canvas ------
        for j, (h0, w0, d0) in enumerate(sub_coords):
            seg_full[:, :, h0:h0+ph, w0:w0+pw, d0:d0+pd] += seg_patch[j:j+1]
            cls_full[:, :, h0:h0+ph, w0:w0+pw, d0:d0+pd] += cls_patch_up[j:j+1]
            weight   [:, :, h0:h0+ph, w0:w0+pw, d0:d0+pd] += w_patch

    seg_full = seg_full / weight.clamp_min(1e-6)
    cls_full = cls_full / weight.clamp_min(1e-6)
    return cls_full, seg_full    # same spatial size (H,W,D)



def visualize_batch_heatmap_overlay(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 2.0,
    out_dir: str = "./heatmap_vis",
    alpha: float = 0.5,
    num_samples: int = 4
) -> None:
    """
    Overlay Gaussian heatmap centroids onto CT slices and save as PNG.

    Parameters
    ----------
    x : torch.Tensor
        CT batch tensor [B, 1, H, W, D], values normalized.
    y : torch.Tensor
        Multi-class mask tensor [B, H, W, D], 0=background, 1..C.
    sigma : float, optional
        Gaussian sigma for heatmap generation.
    out_dir : str, optional
        Directory to save output images.
    alpha : float, optional
        Transparency for heatmap overlay.
    num_samples : int, optional
        Number of samples from batch to visualize.
    """
    os.makedirs(out_dir, exist_ok=True)
    x_np = x.cpu().numpy()  # [B,1,H,W,D]
    y_np = y.cpu().numpy()  # [B,H,W,D]
    B, _, H, W, D = x_np.shape
    # Process up to num_samples
    for b in range(min(B, num_samples)):
        ct = x_np[b, 0]  # [H,W,D]
        mask = y_np[b]
        C = int(mask.max())
        # Compute centroids via NumPy
        centers: List[Tuple[int,int,int]] = []
        for c_id in range(1, C+1):
            coords = np.argwhere(mask == c_id)
            if coords.size == 0:
                centers.append(None)
            else:
                # Mean position rounded to int
                cz, cy, cx = coords.mean(axis=0)
                centers.append((int(cz), int(cy), int(cx)))
        # Filter out missing centroids
        valid_centers = [c for c in centers if c is not None]
        if not valid_centers:
            continue
        # Generate heatmap for this sample
        hmap = make_gaussian_heatmap(
            [centers], shape=(H, W, D), sigma=sigma, device=x.device
        )[0]  # [C,H,W,D]
        # Collapse channels by max
        hmap_combined = hmap.max(dim=0)[0].cpu().numpy()
        # Select mid-slice in Z for display
        z_slice = H // 2
        fig, ax = plt.subplots(figsize=(4,4))
        # Display CT slice
        ax.imshow(ct[:,:,z_slice].T, cmap='gray', origin='lower')
        # Apply heatmap overlay
        hm_rgba = plt.get_cmap('jet')(hmap_combined[:,:,z_slice].T)
        ax.imshow(hm_rgba, alpha=alpha, origin='lower')
        ax.axis('off')
        # Save figure
        out_path = os.path.join(out_dir, f"sample{b:02d}_z{z_slice:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    print(f"Saved heatmap overlays to {out_dir}")


# -----------------------------------------------------------------------------
# 形态学操作相关
# -----------------------------------------------------------------------------
def invert_batch_centers(batch_centers):
    """
    batch_centers: list of length C, each item is a tuple (xs, ys, zs)
                   where xs/ys/zs are 1D tensors of length B, giving the
                   centroid coordinate for that class in each sample
                   (-1 表示该样本里该类不存在)

    Returns:
      per_sample_centers: list of length B; per_sample_centers[b] 是一个长度 C 的 list，
                          每个 entry 是 (x,y,z) tuple 或 None。
    """
    C = len(batch_centers)
    if C == 0:
        return []
    # 把 [(xs,ys,zs), …] 拆成三个列表
    xs_list, ys_list, zs_list = zip(*batch_centers)
    B = xs_list[0].shape[0]

    per_sample = []
    for b in range(B):
        centers_b = []
        for c in range(C):
            x = int(xs_list[c][b].item())
            y = int(ys_list[c][b].item())
            z = int(zs_list[c][b].item())
            # 如果任意坐标为 -1，说明这一样本里没有检测到这个类别
            if x < 0 or y < 0 or z < 0:
                centers_b.append(None)
            else:
                centers_b.append((x, y, z))
        per_sample.append(centers_b)
    return per_sample


def make_gaussian_heatmap(
    centers: List[List[Tuple[int,int,int]]],
    shape: Tuple[int, int, int],
    sigma: float = 2.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate Gaussian heatmaps from centroids for a batch of samples.

    Parameters
    ----------
    centers : List[List[Tuple[int,int,int]]]
        Batch of centroid lists, each inner list length = number of channels.
    shape : tuple of int
        Heatmap spatial dimensions (H, W, D).
    sigma : float, optional
        Standard deviation of Gaussian kernel (default=2.0).
    device : torch.device or None
        Device to create heatmap tensor; defaults to CPU.

    Returns
    -------
    heatmaps : torch.Tensor
        Float tensor of shape [B, C, H, W, D] with values in [0,1].
    """
    B = len(centers)
    C = len(centers[0]) if B > 0 else 0
    H, W, D = shape
    if device is None:
        device = torch.device('cpu')
    # Create coordinate grids
    zz = torch.arange(H, device=device).view(H, 1, 1)
    yy = torch.arange(W, device=device).view(1, W, 1)
    xx = torch.arange(D, device=device).view(1, 1, D)
    # Initialize heatmap tensor
    heatmaps = torch.zeros(B, C, H, W, D, device=device, dtype=torch.float32)
    # Fill in Gaussian for each centroid
    for b in range(B):
        for c, coord in enumerate(centers[b]):
            if coord is None:
                continue
            if coord[0] < 0:
                # Invalid centroid, skip
                continue
            zc, yc, xc = coord
            # Squared distance from centroid
            dist2 = (zz - zc)**2 + (yy - yc)**2 + (xx - xc)**2
            heatmaps[b, c] = torch.exp(-dist2 / (2 * sigma * sigma))
    return heatmaps


def extract_largest_cc_per_class(
    id_map: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Extract the largest connected component for each class in a 3D ID map.

    Parameters
    ----------
    id_map : np.ndarray
        3D integer array shape [H, W, D], values 0..C.

    Returns
    -------
    cc_masks : dict
        Mapping {class_id: binary_mask} where binary_mask is the largest
        connected component of that class.
    """
    # squeeze 多余的前导维度，只保留最后 3 维 (H,W,D)
    if id_map.ndim > 3:
        id_map = np.squeeze(id_map)
        if id_map.ndim > 3:  # 仍然 >3 说明 squeeze 不够
            raise ValueError(f"indices map must be 3-D, got shape {id_map.shape}")


    cc_masks: Dict[int, np.ndarray] = {}
    max_id = int(id_map.max())
    # Iterate over each class (skip background 0)
    for c in range(1, max_id + 1):
        mask_c = (id_map == c).astype(np.uint8)
        if mask_c.sum() == 0:
            continue
        # Label connected components
        labeled, num_feats = label(mask_c, structure=np.ones((3,3,3), dtype=int))
        if num_feats == 0:
            continue
        # Count voxels per component, ignore background
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        # Select largest component ID
        largest_id = counts.argmax()
        # Create binary mask for largest CC
        cc_masks[c] = (labeled == largest_id).astype(np.uint8)
    return cc_masks


def split_stuck_vertebra(
    seg_mask: np.ndarray,
    id_map: np.ndarray,
    dilation_iter: int = 2,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Split touching vertebrae instances using marker-based watershed.

    Parameters
    ----------
    seg_mask : np.ndarray
        Binary foreground mask of shape [H,W,D].
    id_map : np.ndarray
        Multi-class ID map [H,W,D], values 0..C.
    dilation_iter : int, optional
        Number of dilation iterations for seed generation.
    structure : ndarray, optional
        Structuring element for morphological ops; default = 3x3x3 ones.

    Returns
    -------
    inst_map : np.ndarray
        Instance map with same shape, each connected component retains its class ID.
    """
    if seg_mask.shape != id_map.shape:
        raise ValueError("seg_mask and id_map must have same shape")
    if structure is None:
        structure = np.ones((3,3,3), dtype=bool)
    # Compute distance transform
    dist = distance_transform_edt(seg_mask)
    H, W, D = seg_mask.shape
    markers = np.zeros((H, W, D), dtype=np.int32)
    # Create markers per class
    for cls_id in np.unique(id_map):
        if cls_id == 0:
            continue
        mask_cls = (id_map == cls_id)
        if not mask_cls.any():
            continue
        labeled, num_feats = ndi.label(mask_cls, structure=structure)
        if num_feats == 0:
            continue
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        seed_label = counts.argmax()
        seed = (labeled == seed_label)
        # Dilate seed for watershed
        seed = ndi.binary_dilation(seed, structure=structure, iterations=dilation_iter)
        markers[seed] = int(cls_id)
    # Apply watershed on negative distance (treat peaks as basins)
    inst_map = watershed(-dist, markers, mask=seg_mask).astype(np.uint8)
    return inst_map



def split_vertebra_cc(
        fg: np.ndarray,                     # (H,W,D) bool
        heat: np.ndarray,                   # (C,H,W,D)
        seeds_xyz: list[tuple[int,int,int]],
        seeds_cls: list[int],
        dist_thr: float = 6.0,              # ≥ 6 voxel away from BG → 视为“主体”
        lam_z: float = 3.0,
) -> np.ndarray:
    """
    Hybrid: Watershed (core) + Nearest-Centroid (rim)

    Returns
    -------
    (H,W,D)  uint8 instance map
    """
    from skimage.segmentation import watershed

    C, H, W, D = heat.shape
    dist = ndi.distance_transform_edt(fg)

    # ------------------------------------------------------------------
    # 1) 准备 marker 体素 (同 split_vertebra_cc 中的 dilated seeds)
    markers = np.zeros_like(fg, np.uint8)
    for (x, y, z), cid in zip(seeds_xyz, seeds_cls):
        markers[x, y, z] = cid
    # 轻微膨胀防止单点种子溺水
    markers = ndi.binary_dilation(markers, iterations=1).astype(np.uint8) * markers

    # ------------------------------------------------------------------
    # 2) Watershed（距离场越大越靠“山顶” → 取负号做 basin）
    ws_lbl = watershed(
        -dist, markers=markers, mask=fg,
        connectivity=np.ones((3, 3, 3), bool)
    ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 3) “主体”区域：距离大于阈值的前景 + ws 标签
    core = (dist >= dist_thr) & fg
    inst_core = ws_lbl * core

    # ------------------------------------------------------------------
    # 4) 为 rim 区域执行 Nearest-Centroid
    rim_mask = fg & (inst_core == 0)
    if rim_mask.any():
        # 预建 xyz 网格（用 ogrid 省显存）
        gx, gy, gz = np.ogrid[:H, :W, :D]
        gx = gx.astype(np.float32); gy = gy.astype(np.float32); gz = gz.astype(np.float32)

        # seed 坐标 → numpy
        ctrs = np.array(seeds_xyz, np.float32)          # (N,3)
        lam   = lam_z

        best_d2 = np.full((H, W, D), np.inf, np.float32)
        rim_lbl = np.zeros((H, W, D), np.uint8)

        for (cx, cy, cz), cid in zip(ctrs, seeds_cls):
            d2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (lam * (gz - cz)) ** 2
            sel = (d2 < best_d2) & rim_mask
            rim_lbl[sel]  = cid
            best_d2[sel] = d2[sel]

        inst_full = inst_core + rim_lbl   # core 覆盖优先
    else:
        inst_full = inst_core.copy()

    return inst_full.astype(np.uint8)


# -----------------------------------------------------------------------------
# Helper -- 1-D Non-Maximum-Suppression along the axial (z) direction
# -----------------------------------------------------------------------------
def suppress_close_seeds(
        seeds_xyz: List[Tuple[int, int, int]],
        seeds_score: List[float],
        seeds_cid:  List[int],
        dz: int = 2
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """
    Z-axis NMS with anatomical ordering constraint.

    Parameters
    ----------
    seeds_xyz   : list[(x,y,z)]      — candidate centroids
    seeds_score : list[float]        — peak value / confidence
    seeds_cid   : list[int]          — predicted channel id (1..25)
    dz          : int                — half window along z

    Returns
    -------
    kept_xyz : list[(x,y,z)]
    kept_cls : list[int]             — strictly 1,2,3… following z-order
    """
    if not seeds_xyz:
        return [], []

    xyz   = np.asarray(seeds_xyz, dtype=int)      # (N,3)
    score = np.asarray(seeds_score, dtype=float)  # (N,)
    cid   = np.asarray(seeds_cid,  dtype=int)     # (N,)

    # -- 按 z 升序排好，方便逐 slab 处理 --
    order_z = np.argsort(xyz[:, 2])
    xyz, score, cid = xyz[order_z], score[order_z], cid[order_z]

    kept_xyz: List[Tuple[int, int, int]] = []
    kept_cls: List[int] = []

    i, N = 0, len(xyz)
    last_cls = 0            # 已选最大类别
    while i < N:
        z_ref = xyz[i, 2]
        j = i
        cand_idx = []
        # 同一 slab: z ∈ [z_ref-dz, z_ref+dz]
        while j < N and xyz[j, 2] - z_ref <= dz:
            if abs(xyz[j, 2] - z_ref) <= dz:
                cand_idx.append(j)
            j += 1

        # 过滤满足 cid > last_cls 的候选
        valid = [k for k in cand_idx if cid[k] > last_cls]
        use_set = valid if valid else cand_idx   # 若无 valid, 全部 fallback

        # 在 use_set 中选 score 最高
        scores_sel = score[use_set]
        k_rel = int(np.argmax(scores_sel))
        k_best = use_set[k_rel]

        xyz_best = tuple(xyz[k_best])
        cls_best = last_cls + 1                  # 强制递增

        kept_xyz.append(xyz_best)
        kept_cls.append(cls_best)
        last_cls = cls_best

        # 跳过本 slab
        i = j

    return kept_xyz, kept_cls


def relabel_by_seeds(inst_map: np.ndarray,
                     id_map: np.ndarray,
                     overwrite: bool = False) -> np.ndarray:
    """
    Relabel `inst_map` according to small centroid cubes in `id_map`.

    Parameters
    ----------
    inst_map : (H,W,D) uint8
        Instance map produced by CC–splitting.
    id_map   : (H,W,D) uint8
        Small cubes around seeds. Value == class-ID, 0 == background.
    overwrite : bool
        True  – seeds直接覆盖任何已有标签
        False – 仅在 inst_map == 0 的位置填补。

    Returns
    -------
    out : (H,W,D) uint8
        Relabelled instance map.
    """
    assert inst_map.shape == id_map.shape, "Shape mismatch between inst_map & id_map"

    out = inst_map.copy()

    if overwrite:
        # seeds (id_map>0) 覆盖全部
        mask = id_map > 0
        out[mask] = id_map[mask]
    else:
        # 只在 inst_map==0 的地方填
        mask = (id_map > 0) & (out == 0)
        out[mask] = id_map[mask]

    return out


def enforce_unique_z(lbl: np.ndarray,
                     per_class: bool = False) -> np.ndarray:
    """
    Ensure that at each axial slice (Z) only a single vertebra ID remains.

    Parameters
    ----------
    lbl : (H,W,D) uint8
        Instance / class label map.
    per_class : bool
        False – 只保证 slice 内实例唯一
        True  – 对每个类别分别唯一

    Returns
    -------
    out : (H,W,D) uint8
    """
    _, _, D = lbl.shape
    out = lbl.copy()

    if not per_class:
        # ----- slice-level唯一性（无类别） -----
        for z in range(D):
            ids, counts = np.unique(out[..., z], return_counts=True)
            # 去掉 0（背景）
            mask = ids > 0
            ids_valid, cnts_valid = ids[mask], counts[mask]
            if ids_valid.size <= 1:
                continue
            best = ids_valid[cnts_valid.argmax()]
            # 其他 id → 置 0
            bad = (out[..., z] != best)
            out[..., z][bad] = 0
    else:
        # ----- 对每个类别分别唯一 -----
        classes = np.unique(out)
        classes = classes[classes > 0]

        for c in classes:
            slc = (out == c)      # bool mask (H,W,D)
            for z in range(D):
                ids, counts = np.unique(slc[..., z], return_counts=True)
                # ids==True (1) 才计入
                if ids.size <= 1:
                    continue
                # 这里 ids 只有 [False, True] 两个元素，取 True 的数量
                keep = counts[ids].argmax()       # True 对应索引=1
                if keep == 0:
                    slc[..., z] = False           # 全部背景
            # 更新输出
            out[~slc & (out == c)] = 0

    return out




'''处理单个椎骨实例'''
def refine_one_id(
    ct: np.ndarray,
    init_lbl: np.ndarray,
    *,
    edge_sigma: Union[float, str] = 'auto',
    seed_erode: int = 1,
    iterations: int = 20,
    alpha: float = 200.0,
    smoothing: int = 2,
) -> np.ndarray:
    """
    Refinement of one vertebra instance using MGAC (morph. geodesic active‐contour).

    Parameters
    ----------
    ct            3-D CT volume (int16 or float32).  Assumed **not** normalised.
    init_lbl      3-D binary mask of the seed instance.
    edge_sigma    Gaussian σ for edge pre-filter.
    seed_erode    How many iterations of 3-D erosion before level-set init.
    iterations    Number of MGAC outer iterations (≈ how far the contour can grow).
    alpha         Balloon force; positive = shrink, negative = grow.  Empirically
                  100-300 让轮廓贴到骨皮质即可。
    smoothing     #iterations of internal morphological smoothing every step.

    Returns
    -------
    refined_mask  same shape binary mask.
    """
    # ---------- 0. normalization ----------
    img_f = ct.astype(np.float32)

    # ---------- 1. dynamic sigma ----------
    if edge_sigma == 'auto':
        L = max(ct.shape)  # extent along largest axis
        sigma = np.clip(0.025 * L, 0.5, 1.5)
    else:
        sigma = float(edge_sigma)
    # --- 1. edge map  ---------------------------------------------------------
    #   a) 先做轻度平滑；b) Sobel 梯度；c) 归一化成 [0,1] → g(I) = 1/(1+|∇I|)
    # img_f = gaussian(ct.astype(np.float32), sigma=sigma, preserve_range=True)
    grad = gaussian_gradient_magnitude(img_f, sigma=sigma)
    # grad = sobel(img_f)                     # |∇I|
    g = 1.0 / (1.0 + grad)                  # edge‐stopping function ∈(0,1]

    # ---------- extra Z-window ----------------------------------
    z_idxs = np.where(init_lbl.any(axis=(0, 1)))[0]
    z_lo, z_hi = z_idxs.min(), z_idxs.max()
    z_margin = round(0.1*abs(z_hi-z_lo))
    z_lo = max(0, z_lo - round(z_margin*2.5))
    z_hi = min(ct.shape[2] - 1, z_hi + round(z_margin*7.5))
    allowed = np.zeros_like(init_lbl, bool)
    allowed[:, :, z_lo:z_hi + 1] = True
    # 把不允许的地方的g设为0 → ∇g ≈ 0 → 前沿无法前进
    g = g * allowed

    # ------- 2. 初始 level-set -------
    # seed = ndi.binary_erosion(init_lbl.astype(bool), iterations=seed_erode)
    struct = np.ones((3, 3, 3), bool)
    seed = binary_opening(init_lbl.astype(bool), structure=struct)  # 0/1
    for i in range(1, seed_erode):
        seed = binary_erosion(seed, structure=struct)  # 0/1
    seed = 2 * seed.astype(np.int8) - 1  # → -1 / +1
    # 若腐蚀后为空，直接返回外扩 1-voxel 的原 mask
    if seed.max() <= 0:
        # return ndi.binary_dilation(init_lbl, iterations=1).astype(np.uint8)
        return init_lbl.astype(np.uint8)

    # --- 3. 运行 MGAC  --------------------------------------------------------
    ls = mgac(
        gimage=g,
        num_iter=iterations,
        init_level_set=seed,
        smoothing=smoothing,
        balloon=alpha,          # 正值 → 膨胀；负值 → 收缩
    )

    # --- 4. 后处理：小洞填充 + 只取最大 CC  ------------------------------
    ls = remove_small_holes(ls, area_threshold=32, connectivity=2)
    lbl_cc, n_cc = ndi.label(ls, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_cc > 0:
        # 取最大连通域
        counts = np.bincount(lbl_cc.ravel())
        lbl_big = counts[1:].argmax() + 1
        ls = (lbl_cc == lbl_big)
        ls = binary_dilation(ls, structure=np.ones((3, 3, 3)))

    return ls.astype(np.uint8)



def crf_refine(ct: np.ndarray,
               prob_fg: np.ndarray,
               compat: int = 4,
               iters: int = 8) -> np.ndarray:
    """
    Refine a single-instance mask via 3D DenseCRF.

    Parameters
    ----------
    ct      : (H,W[,D]) float32, CT intensities normalized to [0,1]
    prob_fg : (H,W[,D]) float32, foreground probability (or init mask as 0/1)
    compat  : int, pairwise compatibility
    iters   : int, number of mean-field iterations

    Returns
    -------
    lbl : (H,W[,D]) uint8, refined hard mask (0 or 1)
    """
    # build 2‐class probability map: bg + fg
    # ensure foreground is float32 in [0,1]
    fg = prob_fg.astype(np.float32)
    bg = 1.0 - fg
    prob = np.stack([bg, fg], axis=0)  # shape (2,H,W[,D])

    C, *sp = prob.shape
    N = int(np.prod(sp))

    # 1) DenseCRF object
    d = dcrf.DenseCRF(N, C)

    # 2) Unary = -log P
    unary = -np.log(prob.reshape(C, -1) + 1e-6)
    unary = np.ascontiguousarray(unary.astype(np.float32))
    d.setUnaryEnergy(unary)

    # 3) pairwise gaussian (spatial smoothing)
    feats_gauss = create_pairwise_gaussian(sdims=(3,3,3), shape=tuple(sp))
    d.addPairwiseEnergy(feats_gauss.astype(np.float32), compat=compat)

    # 4) pairwise bilateral (intensity + spatial)
    # ct[...,None] 保证最后一维是 channel
    feats_bilat = create_pairwise_bilateral(
        sdims=(3,3,3),
        schan=(0.1,),     # intensity sigma (CT 归一化后)
        img=ct[..., None],
        chdim=len(sp)
    )
    d.addPairwiseEnergy(feats_bilat.astype(np.float32), compat=compat)

    # 5) inference
    Q = d.inference(iters)

    # 6) reshape & argmax
    Q = np.array(Q).reshape((C,)+tuple(sp))
    lbl = np.argmax(Q, axis=0).astype(np.uint8)
    return lbl


def bridge_breaks(mask: np.ndarray,
                  z_gap: int = 2) -> np.ndarray:
    """
    mask : binary foreground (单个椎体)
    z_gap: 允许跳过的最大空 slice
    """
    skel = skeletonize_3d(mask)
    # 提取骨架端点
    conn = ndi.convolve(skel.astype(int), np.ones((3,3,3)), mode='constant')
    endpts = np.column_stack(np.where((skel == 1) & (conn == 2)))
    if len(endpts) < 2:
        return mask  # 没有断

    # 按 z 分层，两层端点如果 |Δz|<=z_gap 就连
    g = netx.Graph()
    for i, p in enumerate(endpts):
        g.add_node(i, xyz=p)
    for i, p in enumerate(endpts):
        for j, q in enumerate(endpts):
            if j <= i: continue
            if abs(p[2] - q[2]) <= z_gap:
                g.add_edge(i, j, weight=np.linalg.norm(p - q))

    # 每一连通子图，连最短路径
    mask_filled = mask.copy()
    for comp in netx.connected_components(g):
        comp = list(comp)
        if len(comp) < 2: continue
        # 任选两端最远点
        dmax, pair = 0, None
        for i in comp:
            for j in comp:
                d = np.linalg.norm(endpts[i] - endpts[j])
                if d > dmax:
                    dmax, pair = d, (i, j)
        if pair is None: continue
        p, q = endpts[pair[0]], endpts[pair[1]]
        # 3-D 直线插值
        line = np.linspace(p, q, int(dmax)+1).round().astype(int)
        mask_filled[tuple(line.T)] = 1
    return mask_filled


def refine_vertebra_inst(ct: np.ndarray,
                         pred_prob: np.ndarray,
                         inst_init: np.ndarray,
                         iterations=96,) -> np.ndarray:
    """
    ct         : 归一化 0‒1
    pred_prob  : (C,H,W,D) softmax
    inst_init  : 初始实例分割
    """
    # ---------- (1) CRF 调整像素标签 ----------
    H, W, D = ct.shape
    margin = 48

    refined = np.zeros_like(inst_init, np.uint8)
    inst_init_unique = np.unique(inst_init)[::-1]
    for vid in inst_init_unique:
        if vid == 0:
            continue
        mask0 = (inst_init == vid)
        if mask0.sum() < 16:
            continue

        # 2) CRF 细化，只做前景/背景两类
        prob_fg = mask0.astype(np.float32)  # foreground 概率
        mask_crf = crf_refine(ct, prob_fg,
                              compat=4,
                              iters=16).astype(bool)

        # 3) 形态学开运算：去掉细小裂缝
        mask1 = opening(mask0, ball(2))

        # 4) 交集
        mask2 = mask1 & mask_crf

        # # ---------- (3) 断层自动补桥 ----------
        # mask3 = bridge_breaks(mask2, z_gap=2)

        # ---------- (4) MGAC 小范围精修 ----------
        # ---------- bbox + margin ----------
        # 1. bounding box (fail-safe)
        coords = np.where(mask2)
        if len(coords[0]) < 1:
            refined[mask1 > 0] = vid

            # mid = mask2.shape[1] // 2
            # plt.subplot(3, 3, 5)
            # plt.imshow(mask0[:, mid, :])
            # plt.title('mask0')
            # plt.subplot(3, 3, 6)
            # plt.imshow(mask_crf[:, mid, :])
            # plt.title('mask_crf')
            # plt.subplot(3, 3, 7)
            # plt.imshow(mask1[:, mid, :])
            # plt.title('mask1')
            # plt.subplot(3, 3, 8)
            # plt.imshow(mask2[:, mid, :])
            # plt.title('mask2')
            # plt.show()

            continue

        x0 = max(int(coords[0].min() - margin), 0)
        x1 = min(int(coords[0].max() + margin + 1), H)
        y0 = max(int(coords[1].min() - margin), 0)
        y1 = min(int(coords[1].max() + margin + 1), W)
        z0 = max(int(coords[2].min() - margin), 0)
        z1 = min(int(coords[2].max() + margin + 1), D)
        ct_crop = ct[x0:x1, y0:y1, z0:z1]
        mask_crop = mask2[x0:x1, y0:y1, z0:z1]

        # ---------- single-id refinement ----------
        mask_crop_ref = refine_one_id(
            ct=ct_crop,
            init_lbl=mask_crop,
            seed_erode=0,
            iterations=iterations,
            alpha=0,
            smoothing=1,
        )

        # # --- step2: 可视化一截 sagittal slice ---
        # mid = ct_crop.shape[1] // 2
        # plt.subplot(2, 3, 1)
        # plt.imshow(ct_crop[:, mid, :], cmap='gray')
        # plt.title('CT')
        # plt.subplot(2, 3, 2)
        # plt.imshow(mask_crop[:, mid, :])
        # plt.title('mask')
        # plt.subplot(2, 3, 3)
        # plt.imshow(mask_crop_ref[:, mid, :])
        # plt.title('ref')
        #
        # plt.subplot(2, 3, 4)
        # plt.imshow(mask_crf[x0:x1, y0:y1, z0:z1][:, mid, :])
        # plt.title('mask_crf')
        # plt.subplot(2, 3, 5)
        # plt.imshow(mask2[x0:x1, y0:y1, z0:z1][:, mid, :])
        # plt.title('mask2')

        # ---------- write back ----------
        plt.show()
        refined[x0:x1, y0:y1, z0:z1][mask_crop_ref > 0] = vid

    return refined