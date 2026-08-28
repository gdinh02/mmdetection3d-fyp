#!/usr/bin/env python3
"""
What this script reads:
  - nuScenes JSON metadata through nuscenes-devkit
  - the six key-frame camera images for one scene

What it does NOT read/copy:
  - the large MMDetection3D nuscenes_infos_train/val.pkl
  - LiDAR .bin files
  - radar files
  - camera sweeps

Output example:
  /mnt/z/dataset/scene-0103/
  ├── samples/
  │   ├── CAM_FRONT/
  │   ├── CAM_FRONT_LEFT/
  │   ├── CAM_FRONT_RIGHT/
  │   ├── CAM_BACK/
  │   ├── CAM_BACK_LEFT/
  │   └── CAM_BACK_RIGHT/
  └── nuscenes_infos_scene-0103.pkl

The generated PKL uses the MMDetection3D v2-style structure:
  {
      "metainfo": ...,
      "data_list": [
          {
              "sample_idx": ...,
              "token": ...,
              "timestamp": ...,
              "ego2global": ...,
              "images": {...},
              "cam_instances": {...}
          },
          ...
      ]
  }

It is intended for the standard FCOS3D nuScenes camera-only setup using:
  load_type='mv_image_based'
  modality=dict(use_lidar=False, use_camera=True)
  box_type_3d='Camera'
  use_valid_flag=True

Typical use:
  python nuscenes_tools/create_fcos3d_scene.py --scene scene-0095

Custom paths:
  python create_fcos3d_scene.py \
      --scene scene-0103 \
      --root /mnt/z/dataset/nuscenes \
      --output-base /mnt/z/dataset

If the output already exists:
  python nuscenes_tools/create_fcos3d_scene.py --scene scene-0064 --overwrite
"""

from __future__ import annotations

import argparse
import math
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

try:
    from shapely.geometry import MultiPoint, box as shapely_box
    from shapely.geometry.polygon import Polygon
except ImportError as exc:
    raise ImportError(
        "This script requires shapely. Install it with:\n"
        "    pip install shapely"
    ) from exc


CAMERAS: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

NUSCENES_CLASSES: Tuple[str, ...] = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

NUSCENES_NAME_MAPPING: Dict[str, str] = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

NUSCENES_ATTRIBUTES: Tuple[str, ...] = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a camera-only, single-scene nuScenes dataset and "
            "MMDetection3D/FCOS3D PKL without loading the full infos PKL."
        )
    )

    parser.add_argument(
        "--scene",
        required=True,
        help="nuScenes scene name, e.g. scene-0103",
    )

    parser.add_argument(
        "--root",
        default="/mnt/z/dataset/nuscenes",
        help="Source nuScenes dataset root",
    )

    parser.add_argument(
        "--output-base",
        default="/mnt/z/dataset",
        help="Parent directory for the extracted scene folder",
    )

    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        choices=("v1.0-trainval", "v1.0-mini", "v1.0-test"),
        help="nuScenes metadata version",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the scene output directory if it exists",
    )

    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Generate the PKL but do not copy images",
    )

    parser.add_argument(
        "--copy-json-metadata",
        action="store_true",
        help=(
            "Also copy the nuScenes version JSON directory into the output. "
            "Not required for ordinary FCOS3D training."
        ),
    )

    return parser.parse_args()


def transform_matrix(
    translation: Sequence[float],
    rotation: Sequence[float],
) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def project_center_with_depth(
    point_cam: Sequence[float],
    camera_intrinsic: np.ndarray,
) -> List[float]:
    point = np.asarray(point_cam, dtype=np.float64).reshape(3, 1)
    depth = float(point[2, 0])

    projected = camera_intrinsic @ point
    projected = projected.reshape(3)

    if abs(projected[2]) < 1e-12:
        return [float("nan"), float("nan"), depth]

    u = float(projected[0] / projected[2])
    v = float(projected[1] / projected[2])
    return [u, v, depth]


def post_process_coords(
    corner_coords: Sequence[Sequence[float]],
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    if len(corner_coords) < 3:
        return None

    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = shapely_box(0, 0, image_size[0], image_size[1])

    if not polygon_from_2d_box.intersects(img_canvas):
        return None

    intersection = polygon_from_2d_box.intersection(img_canvas)

    if intersection.is_empty:
        return None

    if isinstance(intersection, Polygon):
        coords = np.asarray(intersection.exterior.coords, dtype=np.float64)
        return (
            float(np.min(coords[:, 0])),
            float(np.min(coords[:, 1])),
            float(np.max(coords[:, 0])),
            float(np.max(coords[:, 1])),
        )

    bounds = intersection.bounds
    if len(bounds) == 4:
        min_x, min_y, max_x, max_y = bounds
        if max_x > min_x and max_y > min_y:
            return float(min_x), float(min_y), float(max_x), float(max_y)

    return None


def category_to_label(category_name: str) -> Optional[int]:
    mapped = NUSCENES_NAME_MAPPING.get(category_name)
    if mapped is None:
        return None
    return NUSCENES_CLASSES.index(mapped)


def get_attribute_label(nusc: NuScenes, ann_rec: dict) -> int:
    attr_tokens = ann_rec.get("attribute_tokens", [])

    if not attr_tokens:
        attr_name = "None"
    else:
        attr_name = nusc.get("attribute", attr_tokens[0])["name"]

    try:
        return NUSCENES_ATTRIBUTES.index(attr_name)
    except ValueError:
        return NUSCENES_ATTRIBUTES.index("None")


def camera_instances_for_sample_data(
    nusc: NuScenes,
    sample_data_token: str,
) -> List[dict]:
    sd_rec = nusc.get("sample_data", sample_data_token)

    if sd_rec["sensor_modality"] != "camera":
        raise ValueError(f"{sample_data_token} is not camera sample_data.")

    if not sd_rec["is_key_frame"]:
        raise ValueError(
            "FCOS3D camera annotations are generated from nuScenes keyframes."
        )

    sample = nusc.get("sample", sd_rec["sample_token"])
    calibrated_sensor = nusc.get(
        "calibrated_sensor", sd_rec["calibrated_sensor_token"]
    )
    ego_pose = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    intrinsic = np.asarray(
        calibrated_sensor["camera_intrinsic"], dtype=np.float64
    )

    width = int(sd_rec.get("width", 1600))
    height = int(sd_rec.get("height", 900))

    e2g_rot = Quaternion(ego_pose["rotation"]).rotation_matrix
    c2e_rot = Quaternion(calibrated_sensor["rotation"]).rotation_matrix

    result: List[dict] = []

    for ann_token in sample.get("anns", []):
        ann_rec = nusc.get("sample_annotation", ann_token)

        label = category_to_label(ann_rec["category_name"])
        if label is None:
            continue

        box = nusc.get_box(ann_token)

        box.translate(-np.asarray(ego_pose["translation"], dtype=np.float64))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)

        box.translate(
            -np.asarray(calibrated_sensor["translation"], dtype=np.float64)
        )
        box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()

        if len(in_front) == 0:
            continue

        visible_corners = corners_3d[:, in_front]

        corner_coords = view_points(
            visible_corners, intrinsic, normalize=True
        ).T[:, :2]

        bbox_2d = post_process_coords(
            corner_coords.tolist(),
            image_size=(width, height),
        )

        if bbox_2d is None:
            continue

        loc = box.center.astype(np.float64).tolist()

        dim_wlh = np.asarray(box.wlh, dtype=np.float64)
        dims_lhw = [
            float(dim_wlh[1]),
            float(dim_wlh[2]),
            float(dim_wlh[0]),
        ]

        yaw = -float(box.orientation.yaw_pitch_roll[0])

        center_2d_depth = project_center_with_depth(loc, intrinsic)
        depth = float(center_2d_depth[2])

        if not math.isfinite(depth) or depth <= 0:
            continue

        global_velo_2d = np.asarray(
            nusc.box_velocity(ann_token)[:2], dtype=np.float64
        )

        global_velo_3d = np.array(
            [global_velo_2d[0], global_velo_2d[1], 0.0],
            dtype=np.float64,
        )

        cam_velo_3d = (
            global_velo_3d
            @ np.linalg.inv(e2g_rot).T
            @ np.linalg.inv(c2e_rot).T
        )

        velocity = [
            float(cam_velo_3d[0]),
            float(cam_velo_3d[2]),
        ]

        instance = {
            "bbox_label": int(label),
            "bbox_label_3d": int(label),
            "bbox": [
                float(bbox_2d[0]),
                float(bbox_2d[1]),
                float(bbox_2d[2]),
                float(bbox_2d[3]),
            ],
            "bbox_3d_isvalid": True,
            "bbox_3d": (
                [float(v) for v in loc]
                + dims_lhw
                + [float(yaw)]
            ),
            "velocity": velocity,
            "center_2d": [
                float(center_2d_depth[0]),
                float(center_2d_depth[1]),
            ],
            "depth": depth,
            "attr_label": int(get_attribute_label(nusc, ann_rec)),
        }

        result.append(instance)

    return result


def find_scene(nusc: NuScenes, scene_name: str) -> dict:
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            return scene

    raise ValueError(
        f"Scene '{scene_name}' was not found in nuScenes version "
        f"'{nusc.version}'."
    )


def iterate_scene_samples(nusc: NuScenes, scene: dict) -> Iterable[dict]:
    token = scene["first_sample_token"]

    while token:
        sample = nusc.get("sample", token)
        yield sample
        token = sample["next"]


def reference_lidar_metadata(
    nusc: NuScenes,
    sample: dict,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    lidar_token = sample.get("data", {}).get("LIDAR_TOP")

    if not lidar_token:
        return None, None, None

    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get(
        "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
    )
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    return lidar_sd, lidar_cs, lidar_pose


def camera_image_info(
    nusc: NuScenes,
    sample_data_token: str,
    reference_lidar_cs: Optional[dict],
    reference_lidar_pose: Optional[dict],
) -> Tuple[dict, Path]:
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    cam_pose = nusc.get("ego_pose", sd["ego_pose_token"])

    source_relative = Path(sd["filename"])
    source_absolute = Path(nusc.dataroot) / source_relative

    img_info = {
        "img_path": source_relative.name,
        "cam2img": np.asarray(
            cs["camera_intrinsic"], dtype=np.float64
        ).tolist(),
        "sample_data_token": sd["token"],
        "timestamp": float(sd["timestamp"]) / 1e6,
        "cam2ego": transform_matrix(
            cs["translation"], cs["rotation"]
        ).astype(np.float32).tolist(),
        "height": int(sd.get("height", 900)),
        "width": int(sd.get("width", 1600)),
    }

    if reference_lidar_cs is not None and reference_lidar_pose is not None:
        t_lidar_to_ego = transform_matrix(
            reference_lidar_cs["translation"],
            reference_lidar_cs["rotation"],
        )
        t_ego_lidar_to_global = transform_matrix(
            reference_lidar_pose["translation"],
            reference_lidar_pose["rotation"],
        )

        t_cam_to_ego = transform_matrix(
            cs["translation"],
            cs["rotation"],
        )
        t_ego_cam_to_global = transform_matrix(
            cam_pose["translation"],
            cam_pose["rotation"],
        )

        t_lidar_to_global = t_ego_lidar_to_global @ t_lidar_to_ego
        t_cam_to_global = t_ego_cam_to_global @ t_cam_to_ego

        t_lidar_to_cam = np.linalg.inv(t_cam_to_global) @ t_lidar_to_global

        img_info["lidar2cam"] = (
            t_lidar_to_cam.astype(np.float32).tolist()
        )

    return img_info, source_absolute


def copy_camera_image(
    source: Path,
    output_root: Path,
    camera_name: str,
) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Camera image not found: {source}")

    destination_dir = output_root / "samples" / camera_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / source.name
    shutil.copy2(source, destination)

    return destination


def build_data_info(
    nusc: NuScenes,
    sample: dict,
    sample_index: int,
    output_root: Path,
    copy_images: bool,
) -> Tuple[dict, int, int]:
    lidar_sd, lidar_cs, lidar_pose = reference_lidar_metadata(nusc, sample)

    if lidar_sd is not None and lidar_pose is not None:
        timestamp = float(lidar_sd["timestamp"]) / 1e6
        ego2global = transform_matrix(
            lidar_pose["translation"],
            lidar_pose["rotation"],
        ).astype(np.float32).tolist()
    else:
        front_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        front_pose = nusc.get("ego_pose", front_sd["ego_pose_token"])
        timestamp = float(front_sd["timestamp"]) / 1e6
        ego2global = transform_matrix(
            front_pose["translation"],
            front_pose["rotation"],
        ).astype(np.float32).tolist()

    data_info = {
        "sample_idx": int(sample_index),
        "token": sample["token"],
        "timestamp": timestamp,
        "ego2global": ego2global,
        "images": {},
        "cam_instances": {},
    }

    copied = 0
    annotation_count = 0

    for camera_name in CAMERAS:
        if camera_name not in sample["data"]:
            raise KeyError(
                f"Sample {sample['token']} does not contain {camera_name}."
            )

        camera_token = sample["data"][camera_name]

        img_info, source_path = camera_image_info(
            nusc,
            camera_token,
            reference_lidar_cs=lidar_cs,
            reference_lidar_pose=lidar_pose,
        )

        data_info["images"][camera_name] = img_info

        cam_instances = camera_instances_for_sample_data(
            nusc,
            camera_token,
        )

        data_info["cam_instances"][camera_name] = cam_instances
        annotation_count += len(cam_instances)

        if copy_images:
            copy_camera_image(
                source=source_path,
                output_root=output_root,
                camera_name=camera_name,
            )
            copied += 1

    return data_info, copied, annotation_count


def determine_split(scene_name: str, version: str) -> str:
    try:
        from nuscenes.utils import splits
    except Exception:
        return "unknown"

    if version == "v1.0-trainval":
        if scene_name in splits.train:
            return "train"
        if scene_name in splits.val:
            return "val"

    if version == "v1.0-mini":
        if scene_name in splits.mini_train:
            return "mini_train"
        if scene_name in splits.mini_val:
            return "mini_val"

    if version == "v1.0-test":
        if scene_name in splits.test:
            return "test"

    return "unknown"


def make_metainfo(version: str, scene_name: str) -> dict:
    return {
        "categories": {
            class_name: index
            for index, class_name in enumerate(NUSCENES_CLASSES)
        },
        "dataset": "nuscenes",
        "version": version,
        "info_version": "1.1",
        "subset_scene": scene_name,
        "subset_split": determine_split(scene_name, version),
        "camera_only": True,
    }


def copy_version_metadata(
    source_root: Path,
    output_root: Path,
    version: str,
) -> None:
    source = source_root / version
    destination = output_root / version

    if not source.exists():
        raise FileNotFoundError(
            f"nuScenes metadata directory not found: {source}"
        )

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)


def sanity_check_generated_pkl(
    output_pkl: Path,
    expected_samples: int,
) -> None:
    with output_pkl.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise RuntimeError("Generated PKL is not a dictionary.")

    if "metainfo" not in data or "data_list" not in data:
        raise RuntimeError(
            "Generated PKL does not have metainfo/data_list."
        )

    if len(data["data_list"]) != expected_samples:
        raise RuntimeError(
            f"Generated PKL has {len(data['data_list'])} samples, "
            f"expected {expected_samples}."
        )

    for info in data["data_list"]:
        if set(info["images"].keys()) != set(CAMERAS):
            raise RuntimeError(
                f"Sample {info.get('token')} does not contain all 6 cameras."
            )

        if set(info["cam_instances"].keys()) != set(CAMERAS):
            raise RuntimeError(
                f"Sample {info.get('token')} does not contain all "
                "6 cam_instances keys."
            )


def print_config_hint(output_root: Path, output_pkl: Path) -> None:
    print()
    print("FCOS3D dataset config values")
    print("----------------------------")
    print(f"data_root = '{output_root.as_posix()}/'")
    print(f"ann_file = '{output_pkl.name}'")
    print("load_type = 'mv_image_based'")
    print("box_type_3d = 'Camera'")
    print("use_valid_flag = True")
    print("modality = dict(use_lidar=False, use_camera=True)")
    print()
    print("data_prefix = dict(")
    print("    pts='',")
    print("    CAM_FRONT='samples/CAM_FRONT',")
    print("    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',")
    print("    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',")
    print("    CAM_BACK='samples/CAM_BACK',")
    print("    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',")
    print("    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',")
    print(")")


def main() -> int:
    args = parse_args()

    source_root = Path(args.root).expanduser().resolve()
    output_base = Path(args.output_base).expanduser().resolve()
    output_root = output_base / args.scene
    output_pkl = output_root / f"nuscenes_infos_{args.scene}.pkl"

    if not source_root.exists():
        raise FileNotFoundError(
            f"nuScenes source root does not exist: {source_root}"
        )

    version_dir = source_root / args.version
    if not version_dir.exists():
        raise FileNotFoundError(
            f"nuScenes metadata directory does not exist: {version_dir}"
        )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists:\n"
                f"  {output_root}\n"
                f"Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("Camera-only nuScenes scene -> MMDetection3D / FCOS3D")
    print("=" * 76)
    print(f"Source root : {source_root}")
    print(f"Version     : {args.version}")
    print(f"Scene       : {args.scene}")
    print(f"Output      : {output_root}")
    print()
    print("NOTE: The original nuscenes_infos_train/val.pkl is NOT opened.")
    print()

    print("[1/5] Loading nuScenes JSON metadata...")
    nusc = NuScenes(
        version=args.version,
        dataroot=str(source_root),
        verbose=False,
    )

    scene = find_scene(nusc, args.scene)
    split = determine_split(args.scene, args.version)

    samples = list(iterate_scene_samples(nusc, scene))

    print(f"  Scene split      : {split}")
    print(f"  Scene samples    : {len(samples)}")
    print(f"  Scene nbr_samples: {scene.get('nbr_samples')}")

    if scene.get("nbr_samples") != len(samples):
        print(
            "  WARNING: traversed sample count differs from scene nbr_samples."
        )

    print()
    print("[2/5] Building camera infos and FCOS3D annotations...")

    data_list: List[dict] = []
    copied_images = 0
    total_annotations = 0

    for index, sample in enumerate(samples):
        info, copied, annotations = build_data_info(
            nusc=nusc,
            sample=sample,
            sample_index=index,
            output_root=output_root,
            copy_images=not args.no_copy,
        )

        data_list.append(info)
        copied_images += copied
        total_annotations += annotations

        print(
            f"\r  sample {index + 1:>3}/{len(samples)} | "
            f"images {'skipped' if args.no_copy else copied_images} | "
            f"camera GT instances {total_annotations}",
            end="",
            flush=True,
        )

    print()

    print()
    print("[3/5] Writing small MMDetection3D PKL...")

    output_data = {
        "metainfo": make_metainfo(args.version, args.scene),
        "data_list": data_list,
    }

    with output_pkl.open("wb") as f:
        pickle.dump(
            output_data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    pkl_mb = output_pkl.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_pkl}")
    print(f"  Size : {pkl_mb:.2f} MB")

    print()
    print("[4/5] Validating generated PKL...")
    sanity_check_generated_pkl(
        output_pkl=output_pkl,
        expected_samples=len(samples),
    )
    print("  PKL structure looks valid.")
    print(
        f"  FCOS3D will expose approximately "
        f"{len(samples) * len(CAMERAS)} camera records "
        f"({len(samples)} samples x {len(CAMERAS)} cameras)."
    )

    if args.copy_json_metadata:
        print()
        print("[5/5] Copying nuScenes JSON metadata directory...")
        copy_version_metadata(
            source_root=source_root,
            output_root=output_root,
            version=args.version,
        )
        print(f"  Copied {args.version}/")
    else:
        print()
        print("[5/5] nuScenes JSON copy skipped.")
        print(
            "  This is fine for ordinary FCOS3D dataset loading/training."
        )

    print()
    print("=" * 76)
    print("DONE")
    print("=" * 76)
    print(f"Scene                 : {args.scene}")
    print(f"Official split        : {split}")
    print(f"Frame samples         : {len(samples)}")
    print(f"Expected camera items : {len(samples) * len(CAMERAS)}")
    print(
        f"Camera images copied  : "
        f"{'0 (--no-copy)' if args.no_copy else copied_images}"
    )
    print(f"Camera GT instances   : {total_annotations}")
    print(f"Output PKL            : {output_pkl}")
    print(f"Output PKL size       : {pkl_mb:.2f} MB")
    print(f"Output root           : {output_root}")

    print_config_hint(output_root, output_pkl)

    print()
    print("Important:")
    print(
        "  Keep use_valid_flag=True in the FCOS3D NuScenesDataset config. "
        "The camera annotations contain bbox_3d_isvalid and intentionally "
        "do not depend on LiDAR point counts."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
