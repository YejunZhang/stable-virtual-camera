"""
Evaluate SEVA on DL3DV-Evaluation dataset.

Each scene: sample 21 frames, use num_input as known views, rest as targets.
One SEVA pass per scene per num_input setting. Computes PSNR/SSIM/LPIPS on targets.

Usage:
    python eval_nvs.py batch \
        --data_dir /mnt/data3/dl3dv__Dataset/DL3DV-Evaluation \
        --num_scenes 5 \
        --total_frames 21 \
        --num_input 3,6,9 \
        --output_dir ./eval_output
"""

import argparse
import json
import math
import os
import os.path as osp
import glob as glob_module
import sys

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from seva.eval import (
    IS_TORCH_NIGHTLY,
    create_transforms_simple,
    get_value_dict,
    infer_prior_inds,
    infer_prior_stats,
    load_img_and_K,
    run_one_scene,
    save_output,
    transform_img_and_K,
)
from seva.geometry import get_camera_dist
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model, seed_everything

device = "cuda:0"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_dl3dv_scene(scene_dir, downscale_factor=4):
    """Load a DL3DV scene from nerfstudio format.

    Args:
        scene_dir: path to scene directory (contains nerfstudio/ subdir)
        downscale_factor: which images_{factor} to use (1, 2, 4, or 8)

    Returns:
        image_paths: list of image file paths
        c2ws: (N, 4, 4) cam-to-world matrices in OpenCV convention
        Ks: (N, 3, 3) intrinsic matrices scaled to actual image resolution
        img_wh: (width, height) of actual images
    """
    ns_dir = osp.join(scene_dir, "nerfstudio")
    with open(osp.join(ns_dir, "transforms.json")) as f:
        meta = json.load(f)

    full_w, full_h = meta["w"], meta["h"]
    scale = 1.0 / downscale_factor

    # Determine image subdirectory
    if downscale_factor == 1:
        img_subdir = "images"
    else:
        img_subdir = f"images_{downscale_factor}"

    image_paths = []
    c2ws = []
    Ks = []

    for frame in meta["frames"]:
        # Image path: replace images/ with images_{factor}/
        orig_path = frame["file_path"]
        fname = osp.basename(orig_path)
        img_path = osp.join(ns_dir, img_subdir, fname)
        image_paths.append(img_path)

        # Camera pose: OpenGL -> OpenCV
        c2w = np.array(frame["transform_matrix"])
        if "applied_transform" in meta:
            applied_transform = np.concatenate(
                [meta["applied_transform"], [[0, 0, 0, 1]]], axis=0
            )
            c2w = np.linalg.inv(applied_transform) @ c2w
        c2ws.append(c2w)

        # Intrinsics: scale from full resolution to actual resolution
        fl_x = meta.get("fl_x", frame.get("fl_x")) * scale
        fl_y = meta.get("fl_y", frame.get("fl_y")) * scale
        cx = meta.get("cx", frame.get("cx")) * scale
        cy = meta.get("cy", frame.get("cy")) * scale
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
        Ks.append(K)

    c2ws = np.array(c2ws)
    # OpenGL -> OpenCV: flip Y and Z
    c2ws[:, :, [1, 2]] *= -1

    Ks = np.array(Ks)
    actual_w = int(full_w * scale)
    actual_h = int(full_h * scale)

    return image_paths, c2ws, Ks, (actual_w, actual_h)


def compute_metrics(pred, gt):
    """Compute PSNR, SSIM, LPIPS between predicted and ground truth images.

    Args:
        pred: (N, 3, H, W) tensor in [-1, 1]
        gt: (N, 3, H, W) tensor in [-1, 1]

    Returns:
        dict with psnr, ssim, lpips values
    """
    # Convert to [0, 1]
    pred_01 = (pred + 1) / 2.0
    gt_01 = (gt + 1) / 2.0

    # PSNR
    mse = ((pred_01 - gt_01) ** 2).mean(dim=[1, 2, 3])
    psnr = (-10 * torch.log10(mse.clamp(min=1e-10))).mean().item()

    # SSIM (simplified structural similarity)
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    ssim = ssim_metric(pred_01, gt_01).item()

    # LPIPS
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(pred.device)
        with torch.no_grad():
            lpips_val = lpips_fn(pred, gt).mean().item()
    except ImportError:
        lpips_val = -1.0

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}


def run_batch(args):
    """Run batch evaluation on DL3DV scenes.

    For each num_input setting, for each scene: sample 21 frames,
    pick num_input as known views, rest as targets, run one SEVA pass,
    then compute PSNR/SSIM/LPIPS on target views.
    """

    num_input_list = [int(x) for x in args.num_input.split(",")]

    # Find all scene directories
    all_scenes = sorted([
        d for d in os.listdir(args.data_dir)
        if osp.isdir(osp.join(args.data_dir, d))
        and osp.exists(osp.join(args.data_dir, d, "nerfstudio", "transforms.json"))
    ])

    if args.num_scenes > 0:
        rng = np.random.RandomState(args.scene_seed)
        scene_indices = rng.choice(len(all_scenes), min(args.num_scenes, len(all_scenes)), replace=False)
        scenes = [all_scenes[i] for i in sorted(scene_indices)]
    else:
        scenes = all_scenes

    print(f"Evaluating {len(scenes)} scenes")
    print(f"Total frames per scene: {args.total_frames}, num_input settings: {num_input_list}")

    # Initialize SEVA model
    print("Loading SEVA model...")
    model = SGMWrapper(
        load_model(
            model_version=args.version,
            pretrained_model_name_or_path="stabilityai/stable-virtual-camera",
            weight_name="model.safetensors",
            device="cpu",
            verbose=True,
        ).eval()
    ).to(device)

    ae = AutoEncoder(chunk_size=1).to(device)
    conditioner = CLIPConditioner().to(device)
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)

    if IS_TORCH_NIGHTLY:
        model = torch.compile(model, dynamic=False)
        conditioner = torch.compile(conditioner, dynamic=False)
        ae = torch.compile(ae, dynamic=False)

    T = args.total_frames  # context window = total frames = 21
    all_results = {}

    for num_input in num_input_list:
        assert num_input < T, f"num_input ({num_input}) must be < total_frames ({T})"
        num_target = T - num_input

        print(f"\n{'='*60}")
        print(f"num_input={num_input}, num_target={num_target}")
        print(f"{'='*60}")

        nv_results = []

        for scene_name in tqdm(scenes, desc=f"input={num_input}"):
            scene_dir = osp.join(args.data_dir, scene_name)

            # Load scene data
            image_paths, c2ws, Ks, (img_w, img_h) = load_dl3dv_scene(
                scene_dir, downscale_factor=args.downscale_factor
            )
            num_frames = len(image_paths)

            if num_frames < T:
                print(f"  Skipping {scene_name}: only {num_frames} frames (need {T})")
                continue

            # Sample T frames from the scene (same for all num_input settings)
            rng = np.random.RandomState(args.scene_seed)
            sampled_indices = sorted(rng.choice(num_frames, T, replace=False).tolist())

            # Pick num_input as input views, rest as target views
            input_rng = np.random.RandomState(args.scene_seed + 1)
            rel_input_indices = sorted(
                input_rng.choice(T, num_input, replace=False).tolist()
            )
            rel_test_indices = [i for i in range(T) if i not in rel_input_indices]

            sampled_paths = [image_paths[i] for i in sampled_indices]
            sampled_c2ws = torch.tensor(c2ws[sampled_indices, :3]).float()
            sampled_Ks = torch.tensor(Ks[sampled_indices]).float()

            # Setup version dict — one pass with T frames
            version_dict = {
                "H": args.H,
                "W": args.W,
                "T": T,
                "C": 4,
                "f": 8,
                "options": {
                    "chunk_strategy": "gt",
                    "video_save_fps": 2.0,
                    "beta_linear_start": 5e-6,
                    "log_snr_shift": 2.4,
                    "guider_types": 1,
                    "cfg": 2.0,
                    "camera_scale": 2.0,
                    "num_steps": args.num_steps,
                    "cfg_min": 1.2,
                    "encoding_t": 1,
                    "decoding_t": 1,
                    "save_input": True,
                },
            }

            # Prepare conditioning
            image_cond = {
                "img": sampled_paths,
                "input_indices": rel_input_indices,
                "prior_indices": [],
            }
            camera_cond = {
                "c2w": sampled_c2ws.clone(),
                "K": sampled_Ks.clone(),
                "input_indices": list(range(T)),
            }

            save_path = osp.join(args.output_dir, f"input{num_input}", scene_name)
            os.makedirs(save_path, exist_ok=True)

            # Run SEVA inference — single pass
            seed_everything(args.scene_seed)
            try:
                video_path_generator = run_one_scene(
                    "img2img",
                    version_dict,
                    model=model,
                    ae=ae,
                    conditioner=conditioner,
                    denoiser=denoiser,
                    image_cond=image_cond,
                    camera_cond=camera_cond,
                    save_path=save_path,
                    use_traj_prior=False,
                    traj_prior_Ks=None,
                    traj_prior_c2ws=None,
                    seed=args.scene_seed,
                )
                for _ in video_path_generator:
                    pass
            except Exception as e:
                print(f"  Error on {scene_name}: {e}")
                continue

            # Load generated images for metrics
            gen_dir = osp.join(save_path, "samples-rgb")
            if not osp.exists(gen_dir):
                print(f"  No output for {scene_name}")
                continue

            gen_paths = sorted(glob_module.glob(osp.join(gen_dir, "*.png")))
            if len(gen_paths) == 0:
                continue

            # Load GT target images using same transform as SEVA (resize+crop)
            H_out, W_out = version_dict["H"], version_dict["W"]
            gt_imgs = []
            for ri in rel_test_indices:
                orig_idx = sampled_indices[ri]
                gt_img, _ = load_img_and_K(
                    image_paths[orig_idx], (W_out, H_out), K=None, device="cpu"
                )
                gt_imgs.append(gt_img[0])  # (3, H, W) in [-1, 1]

            # Load generated target images (000.png, 001.png, ... correspond to test frames in order)
            gen_imgs = []
            for gi in range(len(rel_test_indices)):
                if gi < len(gen_paths):
                    img = Image.open(gen_paths[gi]).convert("RGB")
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_t = torch.from_numpy(img_np).permute(2, 0, 1) * 2.0 - 1.0
                    gen_imgs.append(img_t)

            if len(gen_imgs) == 0 or len(gt_imgs) == 0:
                continue

            n_eval = min(len(gen_imgs), len(gt_imgs))
            gen_tensor = torch.stack(gen_imgs[:n_eval])
            gt_tensor = torch.stack(gt_imgs[:n_eval])

            # Resize GT to match generated if needed
            if gt_tensor.shape[-2:] != gen_tensor.shape[-2:]:
                gt_tensor = F.interpolate(
                    gt_tensor, size=gen_tensor.shape[-2:], mode="bilinear", align_corners=False
                )

            metrics = compute_metrics(gen_tensor, gt_tensor)
            metrics["scene"] = scene_name
            metrics["num_input"] = num_input
            metrics["num_target"] = n_eval
            nv_results.append(metrics)

            print(f"  {scene_name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}")

        all_results[num_input] = nv_results

        # Print summary for this num_input
        if nv_results:
            avg_psnr = np.mean([r["psnr"] for r in nv_results])
            avg_ssim = np.mean([r["ssim"] for r in nv_results])
            avg_lpips = np.mean([r["lpips"] for r in nv_results if r["lpips"] >= 0])
            print(f"\n  Summary (input={num_input}, {len(nv_results)} scenes):")
            print(f"    PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")

    # Save all results
    results_path = osp.join(args.output_dir, "results.json")
    serializable = {}
    for ni, scene_results in all_results.items():
        serializable[f"input{ni}"] = {
            "per_scene": scene_results,
            "avg": {
                "psnr": float(np.mean([r["psnr"] for r in scene_results])),
                "ssim": float(np.mean([r["ssim"] for r in scene_results])),
                "lpips": float(np.mean([r["lpips"] for r in scene_results if r["lpips"] >= 0])),
            } if scene_results else {},
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEVA on DL3DV")
    subparsers = parser.add_subparsers(dest="command")

    # batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Batch evaluate scenes")
    batch_parser.add_argument("--data_dir", type=str, required=True)
    batch_parser.add_argument("--num_scenes", type=int, default=5)
    batch_parser.add_argument("--total_frames", type=int, default=21,
                              help="Total frames sampled per scene (= context window)")
    batch_parser.add_argument("--num_input", type=str, default="3",
                              help="Number of input (known) views, comma-separated for multiple (e.g. 3,6,9)")
    batch_parser.add_argument("--scene_seed", type=int, default=42)
    batch_parser.add_argument("--output_dir", type=str, default="./eval_output")
    batch_parser.add_argument("--downscale_factor", type=int, default=4)
    batch_parser.add_argument("--version", type=float, default=1.1)
    batch_parser.add_argument("--H", type=int, default=576)
    batch_parser.add_argument("--W", type=int, default=576)
    batch_parser.add_argument("--num_steps", type=int, default=50)

    args = parser.parse_args()

    if args.command == "batch":
        run_batch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
