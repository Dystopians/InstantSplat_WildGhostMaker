import os
import re
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.dataset_readers import loadCameras
from gaussian_renderer import render, GaussianModel
from utils.pose_utils import get_tensor_from_camera
from utils.general_utils import inverse_sigmoid


# -------------------------
# Utilities
# -------------------------

def parse_images_txt_order(images_txt_path):
    """
    读取 COLMAP images.txt，返回 [(base, ext), ...]，顺序与 images.txt 中一致。
    """
    names_with_ext = []
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        # images.txt: id qvec[4] tvec[3] cam_id name
        if len(parts) >= 9:
            name = parts[-1]
            base = os.path.splitext(os.path.basename(name))[0]
            ext = os.path.splitext(os.path.basename(name))[1]
            names_with_ext.append((base, ext))
            i += 2  # 跳过下一行的 2D 点
        else:
            i += 1
    return names_with_ext


def safe_read_mask(mask_path):
    if not os.path.exists(mask_path):
        return None
    try:
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr > 0
    except Exception:
        return None


def build_per_image_point_ranges(source_path: str, n_views: int):
    """
    基于 init_geo 阶段产生的 overlapping_masks，估算“每张图来源的初始高斯数量”，
    并据此构建一个“按 images.txt 顺序的连续分段映射”。
    返回：
      ranges: {base: (start_offset, count)}
      order_with_ext: [(base, ext), ...] 与 images.txt 顺序一致
      total_init: sum(counts)
    """
    sparse_dir = os.path.join(source_path, f"sparse_{n_views}", "0")
    images_txt = os.path.join(sparse_dir, "images.txt")
    masks_dir = os.path.join(sparse_dir, f"overlapping_masks_{n_views}")

    order_with_ext = parse_images_txt_order(images_txt)
    counts = []
    for base, ext in order_with_ext:
        count = 0
        # 尝试多种扩展名以匹配掩码文件
        tried_exts = [ext, '.png', '.jpg', '.jpeg', '.JPG', '.PNG']
        seen = set()
        mask = None
        for e in tried_exts:
            if e in seen:
                continue
            seen.add(e)
            p = os.path.join(masks_dir, base + e)
            mask = safe_read_mask(p)
            if mask is not None:
                break
        if mask is not None:
            count = int(mask.sum())
        counts.append(count)

    offsets = np.cumsum([0] + counts[:-1]).tolist()
    ranges = {order_with_ext[i][0]: (offsets[i], counts[i]) for i in range(len(order_with_ext))}
    total_init = int(np.sum(counts))
    return ranges, order_with_ext, total_init


def parse_point_and_heading(name_base: str):
    # 期望形如: route_pt001_h180_p00_fov90
    pt_match = re.search(r"pt(\d+)", name_base)
    h_match = re.search(r"h(\d+)", name_base)
    pt = pt_match.group(1) if pt_match else "000"
    h = int(h_match.group(1)) if h_match else -1
    return pt, h


def build_neighbor_ring(order_with_ext):
    """
    以 pt 分组，在组内按 heading 升序构成环。返回：
      groups: {pt: [(base, ext, h), ...]}
      left_neighbor[(base, ext)] = (base_left, ext_left)
      right_neighbor[(base, ext)] = (base_right, ext_right)
    """
    groups = {}
    for base, ext in order_with_ext:
        pt, h = parse_point_and_heading(base)
        groups.setdefault(pt, []).append((base, ext, h))
    for pt in groups:
        groups[pt] = sorted(groups[pt], key=lambda x: x[2])

    left_neighbor = {}
    right_neighbor = {}
    for pt, items in groups.items():
        n = len(items)
        for i, (base, ext, _h) in enumerate(items):
            j = (i - 1) % n
            k = (i + 1) % n
            left_neighbor[(base, ext)] = (items[j][0], items[j][1])
            right_neighbor[(base, ext)] = (items[k][0], items[k][1])
    return groups, left_neighbor, right_neighbor


def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------
# Core
# -------------------------

def render_ghosts_for_all_views(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args):
    """
    关键策略：
      1) 仅根据“来源分段”屏蔽本图贡献（非可见性）；渲染 others-only 画面；
      2) 如存在空洞，再用左右邻居的“来源分段”做一次邻居专渲染，进行空洞填充；
      3) 全程备份并恢复 opacity，不污染后续循环；
    """
    device = torch.device('cuda')

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 获取训练相机（尽量用优化后的位姿）
    base_cams = scene.getTrainCameras()
    pose_path = Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_optimized.npy'
    if pose_path.exists():
        optimized_pose = np.load(pose_path)  # 按 COLMAP id 升序
        cam_ids = [cam.colmap_id for cam in base_cams]
        reordered = np.stack([optimized_pose[i - 1] for i in cam_ids])
        viewpoint_stack = loadCameras(reordered, base_cams)
    else:
        viewpoint_stack = base_cams

    # 读取 images.txt 顺序 与 掩码统计 -> 构造来源分段映射
    ranges, order_with_ext, total_init = build_per_image_point_ranges(dataset.source_path, args.n_views)
    order_set = set([b for b, _ in order_with_ext])

    # 邻居环（用于兜底填充 + json 映射）
    groups, left_neighbor, right_neighbor = build_neighbor_ring(order_with_ext)

    # 输出目录
    base_out_dir = os.path.join(dataset.model_path, 'train', f'ours_{iteration}')
    ghosts_dir = os.path.join(base_out_dir, 'ghosts')
    ensure_dirs(ghosts_dir)

    # JSON 映射
    mapping_entries = []

    # 当前模型中高斯数量（注意：训练后可能 != total_init）
    num_gauss = gaussians._opacity.data.shape[0]

    def render_with_pose(view, cam_pose):
        pkg = render(view, gaussians, pipeline, background, camera_pose=cam_pose)
        return pkg["render"]

    # 渲染主循环
    for view in viewpoint_stack:
        base = view.image_name
        if base not in order_set:
            # 不是 images.txt 中的训练图，跳过（极少见）
            continue

        # -------- 1) others-only：屏蔽“当前原图来源”的高斯 --------
        if base in ranges:
            start, count = ranges[base]
            # 与当前模型高斯数做一次截断，防越界
            lo = max(0, int(start))
            hi = min(num_gauss, int(start + count))
            if hi > lo:
                idxs_src = torch.arange(lo, hi, device=device, dtype=torch.long)
            else:
                idxs_src = torch.empty((0,), device=device, dtype=torch.long)
        else:
            idxs_src = torch.empty((0,), device=device, dtype=torch.long)

        full_backup = None
        if idxs_src.numel() > 0:
            # 备份并屏蔽来源分段
            full_backup = gaussians._opacity.data[idxs_src].clone()
            gaussians._opacity.data[idxs_src] = inverse_sigmoid(
                torch.full_like(full_backup, 1e-6)
            )

        # 渲染 others-only
        cam_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        with torch.no_grad():
            rendering_others = render_with_pose(view, cam_pose)

        # 恢复
        if full_backup is not None:
            gaussians._opacity.data[idxs_src] = full_backup

        # -------- 2) 兜底：若 coverage 不足，用左右邻居来源分段渲染并填洞 --------
        bg = background.view(3, 1, 1)
        # 计算“非背景覆盖率”
        cov_mask = ((rendering_others - bg).abs().max(dim=0, keepdim=True)[0] >= 0.02).float()
        coverage_ratio = cov_mask.mean().item()

        need_neighbor_fill = coverage_ratio < getattr(args, "min_coverage", 0.90)

        if need_neighbor_fill:
            # 找到左右邻居（可多层跨度）
            pt_id, _h = parse_point_and_heading(base)
            ring = groups.get(pt_id, [])
            bases = [x[0] for x in ring]
            try:
                cur_idx = bases.index(base)
            except ValueError:
                cur_idx = None

            keep_indices_list = []
            if cur_idx is not None and len(ring) > 0:
                n = len(ring)
                span = max(1, int(getattr(args, "neighbor_span", 2)))
                for d in range(1, span + 1):
                    l_i = (cur_idx - d) % n
                    r_i = (cur_idx + d) % n
                    for nb_base in (ring[l_i][0], ring[r_i][0]):
                        if nb_base in ranges:
                            s, c = ranges[nb_base]
                            lo2 = max(0, int(s))
                            hi2 = min(num_gauss, int(s + c))
                            if hi2 > lo2:
                                keep_indices_list.append(torch.arange(lo2, hi2, device=device, dtype=torch.long))

            # 若没有拿到邻居来源，则直接保存 others-only；否则做邻居专渲染
            if len(keep_indices_list) > 0:
                idxs_keep = torch.unique(torch.cat(keep_indices_list))
                # 构造“只保留邻居来源”的一次性渲染
                backup_all = gaussians._opacity.data.clone()
                try:
                    keep_mask = torch.zeros_like(backup_all, dtype=torch.bool, device=device)
                    keep_mask[idxs_keep] = True
                    # 先全部关到透明，再把邻居打开为原值
                    gaussians._opacity.data = inverse_sigmoid(torch.full_like(backup_all, 1e-6))
                    gaussians._opacity.data[keep_mask] = backup_all[keep_mask]
                    rendering_neighbors = render_with_pose(view, cam_pose)
                finally:
                    gaussians._opacity.data = backup_all

                # 仅在 others-only 的“近背景处”用邻居结果填补
                cov_holes = ((rendering_others - bg).abs().max(dim=0, keepdim=True)[0] < 0.02).float()
                rendering = rendering_others * (1.0 - cov_holes) + rendering_neighbors * cov_holes
            else:
                rendering = rendering_others
        else:
            rendering = rendering_others

        # -------- 3) 保存输出 & JSON 映射 --------
        out_path = os.path.join(ghosts_dir, f"{base}.png")
        torchvision.utils.save_image(rendering, out_path)

        # 找到扩展名并写入映射
        ext = ""
        for b, e in order_with_ext:
            if b == base:
                ext = e
                break
        gt_filename = f"{base}{ext}"

        # 左邻居（用于参考）
        lbase, lext = left_neighbor.get((base, ext), (base, ext))
        ref_filename = f"{lbase}{lext}"

        mapping_entries.append({
            "ghost_image": f"{base}.png",
            "ground_truth": gt_filename,
            "reference_input": ref_filename,
            "coverage_after_mask": round(coverage_ratio, 4)
        })

    # 写出映射 JSON
    mapping_path = os.path.join(base_out_dir, 'ghosts_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping_entries, f, indent=2, ensure_ascii=False)


# -------------------------
# Entrypoint
# -------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="GET_DATASETS: render per-image ghosts by provenance and neighbor fill")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--get_datasets", action="store_true",
                        help="Enable GET_DATASETS mode (no effect on original training pipeline)")
    parser.add_argument("--neighbor_span", type=int, default=2,
                        help="How many neighbors on each side are allowed to fill holes (default: 2)")
    parser.add_argument("--min_coverage", type=float, default=0.90,
                        help="If others-only coverage (non-background) < this ratio, enable neighbor fill")
    args = get_combined_args(parser)

    if not args.get_datasets:
        raise SystemExit("Run with --get_datasets to avoid altering original behavior.")

    render_ghosts_for_all_views(model.extract(args), args.iterations, pipeline.extract(args), args)
import os
import re
import sys
import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.dataset_readers import loadCameras
from gaussian_renderer import render, GaussianModel
from utils.pose_utils import get_tensor_from_camera
from utils.general_utils import inverse_sigmoid


# -------------------------
# Utilities
# -------------------------

def parse_images_txt_order(images_txt_path):
    """
    读取 COLMAP images.txt，返回 [(base, ext), ...]，顺序与 images.txt 中一致。
    """
    names_with_ext = []
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        # images.txt: id qvec[4] tvec[3] cam_id name
        if len(parts) >= 9:
            name = parts[-1]
            base = os.path.splitext(os.path.basename(name))[0]
            ext = os.path.splitext(os.path.basename(name))[1]
            names_with_ext.append((base, ext))
            i += 2  # 跳过下一行的 2D 点
        else:
            i += 1
    return names_with_ext


def safe_read_mask(mask_path):
    if not os.path.exists(mask_path):
        return None
    try:
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr > 0
    except Exception:
        return None


def build_per_image_point_ranges(source_path: str, n_views: int):
    """
    基于 init_geo 阶段产生的 overlapping_masks，估算“每张图来源的初始高斯数量”，
    并据此构建一个“按 images.txt 顺序的连续分段映射”。
    返回：
      ranges: {base: (start_offset, count)}
      order_with_ext: [(base, ext), ...] 与 images.txt 顺序一致
      total_init: sum(counts)
    """
    sparse_dir = os.path.join(source_path, f"sparse_{n_views}", "0")
    images_txt = os.path.join(sparse_dir, "images.txt")
    masks_dir = os.path.join(sparse_dir, f"overlapping_masks_{n_views}")

    order_with_ext = parse_images_txt_order(images_txt)
    counts = []
    for base, ext in order_with_ext:
        count = 0
        # 尝试多种扩展名以匹配掩码文件
        tried_exts = [ext, '.png', '.jpg', '.jpeg', '.JPG', '.PNG']
        seen = set()
        mask = None
        for e in tried_exts:
            if e in seen:
                continue
            seen.add(e)
            p = os.path.join(masks_dir, base + e)
            mask = safe_read_mask(p)
            if mask is not None:
                break
        if mask is not None:
            count = int(mask.sum())
        counts.append(count)

    offsets = np.cumsum([0] + counts[:-1]).tolist()
    ranges = {order_with_ext[i][0]: (offsets[i], counts[i]) for i in range(len(order_with_ext))}
    total_init = int(np.sum(counts))
    return ranges, order_with_ext, total_init


def parse_point_and_heading(name_base: str):
    # 期望形如: route_pt001_h180_p00_fov90
    pt_match = re.search(r"pt(\d+)", name_base)
    h_match = re.search(r"h(\d+)", name_base)
    pt = pt_match.group(1) if pt_match else "000"
    h = int(h_match.group(1)) if h_match else -1
    return pt, h


def build_neighbor_ring(order_with_ext):
    """
    以 pt 分组，在组内按 heading 升序构成环。返回：
      groups: {pt: [(base, ext, h), ...]}
      left_neighbor[(base, ext)] = (base_left, ext_left)
      right_neighbor[(base, ext)] = (base_right, ext_right)
    """
    groups = {}
    for base, ext in order_with_ext:
        pt, h = parse_point_and_heading(base)
        groups.setdefault(pt, []).append((base, ext, h))
    for pt in groups:
        groups[pt] = sorted(groups[pt], key=lambda x: x[2])

    left_neighbor = {}
    right_neighbor = {}
    for pt, items in groups.items():
        n = len(items)
        for i, (base, ext, _h) in enumerate(items):
            j = (i - 1) % n
            k = (i + 1) % n
            left_neighbor[(base, ext)] = (items[j][0], items[j][1])
            right_neighbor[(base, ext)] = (items[k][0], items[k][1])
    return groups, left_neighbor, right_neighbor


def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------
# Auto-train if missing
# -------------------------

def _model_artifact_exists(model_path: str, iteration: int) -> bool:
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        if not os.path.isdir(point_cloud_dir):
            return False
        # any iteration_* having point_cloud.ply
        try:
            for name in os.listdir(point_cloud_dir):
                if name.startswith("iteration_"):
                    ply_path = os.path.join(point_cloud_dir, name, "point_cloud.ply")
                    if os.path.isfile(ply_path):
                        return True
        except FileNotFoundError:
            return False
        return False
    else:
        ply_path = os.path.join(point_cloud_dir, f"iteration_{iteration}", "point_cloud.ply")
        return os.path.isfile(ply_path)


def _run_subprocess(cmd_args, log_path: str):
    ensure_dirs(os.path.dirname(log_path))
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd_args, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}). See log: {log_path}\nCmd: {' '.join(cmd_args)}")


def _run_init_geo(args):
    root_dir = str(Path(__file__).resolve().parent)
    script_path = os.path.join(root_dir, "init_geo.py")
    cmd = [
        sys.executable,
        script_path,
        "-s", args.source_path,
        "-m", args.model_path,
        "--n_views", str(args.n_views),
    ]
    if getattr(args, "auto_focal_avg", True):
        cmd.append("--focal_avg")
    if getattr(args, "auto_co_vis_dsp", True):
        cmd.append("--co_vis_dsp")
    if getattr(args, "auto_conf_aware_ranking", True):
        cmd.append("--conf_aware_ranking")
    if getattr(args, "auto_infer_video", True):
        cmd.append("--infer_video")
    log_path = os.path.join(args.model_path, "01_init_geo_auto.log")
    _run_subprocess(cmd, log_path)


def _run_train(args):
    root_dir = str(Path(__file__).resolve().parent)
    script_path = os.path.join(root_dir, "train.py")
    # Prefer user-requested iterations when provided (>0), else fall back to auto
    iterations = int(args.iterations) if getattr(args, "iterations", -1) and int(args.iterations) > 0 else int(getattr(args, "auto_iterations", 4000))
    position_lr_max_steps = int(getattr(args, "auto_position_lr_max_steps", iterations))
    cmd = [
        sys.executable,
        script_path,
        "-s", args.source_path,
        "-m", args.model_path,
        "-r", str(getattr(args, "auto_resolution", 1)),
        "--n_views", str(args.n_views),
        "--iterations", str(iterations),
        "--position_lr_init", str(getattr(args, "auto_position_lr_init", 3e-5)),
        "--position_lr_final", str(getattr(args, "auto_position_lr_final", 3e-7)),
        "--position_lr_delay_mult", str(getattr(args, "auto_position_lr_delay_mult", 0.01)),
        "--position_lr_max_steps", str(position_lr_max_steps),
        "--feature_lr", str(getattr(args, "auto_feature_lr", 0.0025)),
        "--opacity_lr", str(getattr(args, "auto_opacity_lr", 0.05)),
        "--scaling_lr", str(getattr(args, "auto_scaling_lr", 0.003)),
        "--rotation_lr", str(getattr(args, "auto_rotation_lr", 3e-4)),
        "--lambda_dssim", str(getattr(args, "auto_lambda_dssim", 0.2)),
    ]
    if getattr(args, "auto_init_scale_from_view_depth", True):
        cmd.append("--init_scale_from_view_depth")
    if getattr(args, "auto_pp_optimizer", True):
        cmd.append("--pp_optimizer")
    if getattr(args, "auto_optim_pose", True):
        cmd.append("--optim_pose")
    log_path = os.path.join(args.model_path, "02_train_auto.log")
    _run_subprocess(cmd, log_path)


def _sparse_exists(source_path: str, n_views: int) -> bool:
    base = os.path.join(source_path, f"sparse_{n_views}")
    return os.path.isfile(os.path.join(base, "0", "images.txt")) and os.path.isfile(os.path.join(base, "0", "cameras.txt"))


def ensure_model_exists_for_render(args, force: bool = False):
    """
    If the expected trained model is missing, run init_geo.py and train.py
    with defaults mirroring scripts/run_infer.sh, then continue.
    """
    if not force and _model_artifact_exists(args.model_path, args.iterations):
        return

    print("[Auto-Train] Preparing geometry and training 3D Gaussians...")
    if not _sparse_exists(args.source_path, args.n_views):
        _run_init_geo(args)
    else:
        print("[Auto-Train] Found existing sparse data. Skipping init_geo.")
    _run_train(args)
    print("[Auto-Train] Training finished. Proceeding to render.")


# Derive default model_path like scripts/run_infer.sh when not provided
def _derive_default_model_path(source_path: str, n_views: int, output_dir: str = "output_infer") -> str:
    sp = Path(source_path).resolve()
    scene = sp.name
    dataset = sp.parent.name
    return str(Path("./") / output_dir / dataset / scene / f"{n_views}_views")


# -------------------------
# Core
# -------------------------

def render_ghosts_for_all_views(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args):
    """
    关键策略：
      1) 仅根据“来源分段”屏蔽本图贡献（非可见性）；渲染 others-only 画面；
      2) 如存在空洞，再用左右邻居的“来源分段”做一次邻居专渲染，进行空洞填充；
      3) 全程备份并恢复 opacity，不污染后续循环；
    """
    device = torch.device('cuda')

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 获取训练相机（尽量用优化后的位姿）
    base_cams = scene.getTrainCameras()
    pose_path = Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_optimized.npy'
    if pose_path.exists():
        optimized_pose = np.load(pose_path)  # 按 COLMAP id 升序
        cam_ids = [cam.colmap_id for cam in base_cams]
        reordered = np.stack([optimized_pose[i - 1] for i in cam_ids])
        viewpoint_stack = loadCameras(reordered, base_cams)
    else:
        viewpoint_stack = base_cams

    # 读取 images.txt 顺序 与 掩码统计 -> 构造来源分段映射
    ranges, order_with_ext, total_init = build_per_image_point_ranges(dataset.source_path, args.n_views)
    order_set = set([b for b, _ in order_with_ext])

    # 邻居环（用于兜底填充 + json 映射）
    groups, left_neighbor, right_neighbor = build_neighbor_ring(order_with_ext)

    # 输出目录
    base_out_dir = os.path.join(dataset.model_path, 'train', f'ours_{iteration}')
    ghosts_dir = os.path.join(base_out_dir, 'ghosts')
    ensure_dirs(ghosts_dir)

    # JSON 映射
    mapping_entries = []

    # 当前模型中高斯数量（注意：训练后可能 != total_init）
    num_gauss = gaussians._opacity.data.shape[0]

    def render_with_pose(view, cam_pose):
        pkg = render(view, gaussians, pipeline, background, camera_pose=cam_pose)
        return pkg["render"]

    # 渲染主循环
    for view in viewpoint_stack:
        base = view.image_name
        if base not in order_set:
            # 不是 images.txt 中的训练图，跳过（极少见）
            continue

        # -------- 1) others-only：屏蔽“当前原图来源”的高斯 --------
        if base in ranges:
            start, count = ranges[base]
            # 与当前模型高斯数做一次截断，防越界
            lo = max(0, int(start))
            hi = min(num_gauss, int(start + count))
            if hi > lo:
                idxs_src = torch.arange(lo, hi, device=device, dtype=torch.long)
            else:
                idxs_src = torch.empty((0,), device=device, dtype=torch.long)
        else:
            idxs_src = torch.empty((0,), device=device, dtype=torch.long)

        full_backup = None
        if idxs_src.numel() > 0:
            # 备份并屏蔽来源分段
            full_backup = gaussians._opacity.data[idxs_src].clone()
            gaussians._opacity.data[idxs_src] = inverse_sigmoid(
                torch.full_like(full_backup, 1e-6)
            )

        # 渲染 others-only
        cam_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        with torch.no_grad():
            rendering_others = render_with_pose(view, cam_pose)

        # 恢复
        if full_backup is not None:
            gaussians._opacity.data[idxs_src] = full_backup

        # -------- 2) 兜底：若 coverage 不足，用左右邻居来源分段渲染并填洞 --------
        bg = background.view(3, 1, 1)
        # 计算“非背景覆盖率”
        cov_mask = ((rendering_others - bg).abs().max(dim=0, keepdim=True)[0] >= 0.02).float()
        coverage_ratio = cov_mask.mean().item()

        need_neighbor_fill = coverage_ratio < getattr(args, "min_coverage", 0.90)

        if need_neighbor_fill:
            # 找到左右邻居（可多层跨度）
            pt_id, _h = parse_point_and_heading(base)
            ring = groups.get(pt_id, [])
            bases = [x[0] for x in ring]
            try:
                cur_idx = bases.index(base)
            except ValueError:
                cur_idx = None

            keep_indices_list = []
            if cur_idx is not None and len(ring) > 0:
                n = len(ring)
                span = max(1, int(getattr(args, "neighbor_span", 2)))
                for d in range(1, span + 1):
                    l_i = (cur_idx - d) % n
                    r_i = (cur_idx + d) % n
                    for nb_base in (ring[l_i][0], ring[r_i][0]):
                        if nb_base in ranges:
                            s, c = ranges[nb_base]
                            lo2 = max(0, int(s))
                            hi2 = min(num_gauss, int(s + c))
                            if hi2 > lo2:
                                keep_indices_list.append(torch.arange(lo2, hi2, device=device, dtype=torch.long))

            # 若没有拿到邻居来源，则直接保存 others-only；否则做邻居专渲染
            if len(keep_indices_list) > 0:
                idxs_keep = torch.unique(torch.cat(keep_indices_list))
                # 构造“只保留邻居来源”的一次性渲染
                backup_all = gaussians._opacity.data.clone()
                try:
                    keep_mask = torch.zeros_like(backup_all, dtype=torch.bool, device=device)
                    keep_mask[idxs_keep] = True
                    # 先全部关到透明，再把邻居打开为原值
                    gaussians._opacity.data = inverse_sigmoid(torch.full_like(backup_all, 1e-6))
                    gaussians._opacity.data[keep_mask] = backup_all[keep_mask]
                    rendering_neighbors = render_with_pose(view, cam_pose)
                finally:
                    gaussians._opacity.data = backup_all

                # 仅在 others-only 的“近背景处”用邻居结果填补
                cov_holes = ((rendering_others - bg).abs().max(dim=0, keepdim=True)[0] < 0.02).float()
                rendering = rendering_others * (1.0 - cov_holes) + rendering_neighbors * cov_holes
            else:
                rendering = rendering_others
        else:
            rendering = rendering_others

        # -------- 3) 保存输出 & JSON 映射 --------
        out_path = os.path.join(ghosts_dir, f"{base}.png")
        torchvision.utils.save_image(rendering, out_path)

        # 找到扩展名并写入映射
        ext = ""
        for b, e in order_with_ext:
            if b == base:
                ext = e
                break
        gt_filename = f"{base}{ext}"

        # 左邻居（用于参考）
        lbase, lext = left_neighbor.get((base, ext), (base, ext))
        ref_filename = f"{lbase}{lext}"

        mapping_entries.append({
            "ghost_image": f"{base}.png",
            "ground_truth": gt_filename,
            "reference_input": ref_filename,
            "coverage_after_mask": round(coverage_ratio, 4)
        })

    # 写出映射 JSON
    mapping_path = os.path.join(base_out_dir, 'ghosts_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping_entries, f, indent=2, ensure_ascii=False)


# -------------------------
# Entrypoint
# -------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="GET_DATASETS: render per-image ghosts by provenance and neighbor fill")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--get_datasets", action="store_true",
                        help="Enable GET_DATASETS mode (no effect on original training pipeline)")
    parser.add_argument("--neighbor_span", type=int, default=2,
                        help="How many neighbors on each side are allowed to fill holes (default: 2)")
    parser.add_argument("--min_coverage", type=float, default=0.90,
                        help="If others-only coverage (non-background) < this ratio, enable neighbor fill")
    # Auto-train defaults (can be changed here or via CLI)
    parser.add_argument("--auto_iterations", type=int, default=4000)
    parser.add_argument("--auto_resolution", type=int, default=1)
    parser.add_argument("--auto_focal_avg", action="store_true", default=True)
    parser.add_argument("--auto_co_vis_dsp", action="store_true", default=True)
    parser.add_argument("--auto_conf_aware_ranking", action="store_true", default=True)
    parser.add_argument("--auto_infer_video", action="store_true", default=True)
    parser.add_argument("--auto_init_scale_from_view_depth", action="store_true", default=True)
    parser.add_argument("--auto_pp_optimizer", action="store_true", default=True)
    parser.add_argument("--auto_optim_pose", action="store_true", default=True)
    parser.add_argument("--auto_position_lr_init", type=float, default=3e-5)
    parser.add_argument("--auto_position_lr_final", type=float, default=3e-7)
    parser.add_argument("--auto_position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--auto_position_lr_max_steps", type=int, default=None)
    parser.add_argument("--auto_feature_lr", type=float, default=0.0025)
    parser.add_argument("--auto_opacity_lr", type=float, default=0.05)
    parser.add_argument("--auto_scaling_lr", type=float, default=0.003)
    parser.add_argument("--auto_rotation_lr", type=float, default=3e-4)
    parser.add_argument("--auto_lambda_dssim", type=float, default=0.2)
    cfg_missing = False
    try:
        args = get_combined_args(parser)
    except FileNotFoundError:
        # No cfg_args => fall back to plain CLI args and auto-train to requested iteration
        cfg_missing = True
        args = parser.parse_args(sys.argv[1:])
        if not getattr(args, "model_path", None) or args.model_path == "":
            if not getattr(args, "source_path", None) or not getattr(args, "n_views", None):
                raise SystemExit("When cfg_args is missing, please provide --model_path or both --source_path and --n_views.")
            args.model_path = _derive_default_model_path(args.source_path, args.n_views)
        print("Config file missing; using derived model_path:", args.model_path)

    if not args.get_datasets:
        raise SystemExit("Run with --get_datasets to avoid altering original behavior.")

    # Ensure model artifacts exist; if cfg missing, force training up to target iteration
    ensure_model_exists_for_render(args, force=cfg_missing)

    render_ghosts_for_all_views(model.extract(args), args.iterations, pipeline.extract(args), args)
