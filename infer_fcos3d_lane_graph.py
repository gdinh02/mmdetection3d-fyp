from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from mmdet3d.apis import inference_mono_3d_detector, init_model

from lane_graph import (
    LaneBoundaryConfig,
    LaneFitConfig,
    LaneGraphConfig,
    LaneProjectionConfig,
    RoadPlaneConfig,
    TemporalConfig,
    accumulate_temporal_vehicle_evidence,
    build_lane_compatibility_graph_from_vehicles,
    estimate_road_plane,
    extract_vehicles_from_prediction,
    fit_lane_streams,
    get_lane_streams,
    infer_lane_boundaries,
    plot_lane_graph,
    plot_projected_lane_boundaries,
    print_graph_edges,
    project_lane_boundaries_to_image,
)


CONFIG = (
    "configs/fcos3d/"
    "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
)
CHECKPOINT = (
    "checkpoints/"
    "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_"
    "20210717_095645-8d806dc2.pth"
)

NUSCENES_ROOT = Path("/mnt/z/dataset/scene-0064")
NUSCENES_VERSION = None  # auto-detect trainval / mini / test
CAM_TYPE = "CAM_FRONT"
DEVICE = "cuda:0"
WANTED_VEHICLE_CLASSES = {"car", "truck", "bus"}

# Selection priority: SCENE_TOKEN -> SCENE_NAME -> SCENE_INDEX.
SCENE_TOKEN = None
SCENE_NAME = None
SCENE_INDEX = 64

# Keyframe within the selected scene. -1 = final keyframe.
REFERENCE_SAMPLE_INDEX = -1

GRAPH_CONFIG = LaneGraphConfig(
    score_thresh=0.30,
    max_depth=50.0,
    max_cross_track=1.5,
    max_yaw_diff_deg=15.0,
    max_along_track=25.0,
    sigma_cross_track=0.8,
    sigma_yaw_deg=8.0,
)

TEMPORAL_CONFIG = TemporalConfig(
    history_frames=5,
    max_track_distance=12.0,
    max_track_yaw_diff_deg=30.0,
    max_track_frame_gap=1,
    temporal_decay=0.90,
    min_track_observations=1,
)

FIT_CONFIG = LaneFitConfig(
    degree=2,
    residual_threshold=0.75,
    max_trials=200,
    random_seed=0,
)

BOUNDARY_CONFIG = LaneBoundaryConfig(
    min_overlap=2.0,
    min_lane_width=2.5,
    max_lane_width=5.0,
    sample_count=50,
)

ROAD_PLANE_CONFIG = RoadPlaneConfig(
    residual_threshold=0.35,
    max_trials=200,
    random_seed=0,
)

PROJECTION_CONFIG = LaneProjectionConfig(
    sample_count=200,
    min_depth=1.0,
    clip_to_image=True,
)

NUSCENES_CLASSES = (
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


def detect_nuscenes_version(root):
    if NUSCENES_VERSION is not None:
        version_dir = root / NUSCENES_VERSION
        if not version_dir.is_dir():
            raise FileNotFoundError(f"Missing nuScenes metadata: {version_dir}")
        return NUSCENES_VERSION

    for version in ("v1.0-trainval", "v1.0-mini", "v1.0-test"):
        if (root / version).is_dir():
            return version

    raise FileNotFoundError(
        f"No nuScenes metadata folder found under {root}. Expected "
        "v1.0-trainval, v1.0-mini, or v1.0-test."
    )


def make_transform(translation, rotation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def choose_scene(nusc):
    if SCENE_TOKEN is not None:
        return nusc.get("scene", SCENE_TOKEN)

    if SCENE_NAME is not None:
        matches = [scene for scene in nusc.scene if scene["name"] == SCENE_NAME]
        if not matches:
            raise ValueError(f"No scene named {SCENE_NAME!r}")
        return matches[0]

    if not 0 <= SCENE_INDEX < len(nusc.scene):
        raise IndexError(f"SCENE_INDEX must be in 0..{len(nusc.scene) - 1}")
    return nusc.scene[SCENE_INDEX]


def scene_samples(nusc, scene):
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    return samples


def select_temporal_window(samples):
    reference_idx = (
        REFERENCE_SAMPLE_INDEX
        if REFERENCE_SAMPLE_INDEX >= 0
        else len(samples) + REFERENCE_SAMPLE_INDEX
    )
    if reference_idx < 0 or reference_idx >= len(samples):
        raise IndexError(
            f"REFERENCE_SAMPLE_INDEX={REFERENCE_SAMPLE_INDEX} is outside "
            f"0..{len(samples) - 1}"
        )
    start = max(0, reference_idx - TEMPORAL_CONFIG.history_frames + 1)
    return list(enumerate(samples[start : reference_idx + 1], start=start))


def build_mmdet_sample_info(nusc, sample, sample_idx):
    cam_token = sample["data"][CAM_TYPE]
    sample_data = nusc.get("sample_data", cam_token)
    calibrated_sensor = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

    image_path = Path(nusc.get_sample_data_path(cam_token)).resolve()
    cam2ego = make_transform(
        calibrated_sensor["translation"], calibrated_sensor["rotation"]
    )
    ego2global = make_transform(ego_pose["translation"], ego_pose["rotation"])
    cam2img = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64)

    info = {
        "sample_idx": sample_idx,
        "token": sample["token"],
        "timestamp": sample_data["timestamp"],
        "ego2global": ego2global.tolist(),
        "images": {
            CAM_TYPE: {
                "img_path": str(image_path),
                "cam2img": cam2img.tolist(),
                "sample_data_token": cam_token,
                "timestamp": sample_data["timestamp"],
                "cam2ego": cam2ego.tolist(),
            }
        },
        "instances": [],
    }
    return info, str(image_path)


def write_single_sample_info(info, path):
    mmengine.dump(
        {
            "metainfo": {
                "classes": NUSCENES_CLASSES,
                "dataset": "NuScenesDataset",
                "info_version": "1.1",
            },
            "data_list": [info],
        },
        path,
    )


def camera_to_global(info):
    ego2global = np.asarray(info["ego2global"], dtype=np.float64)
    cam2ego = np.asarray(info["images"][CAM_TYPE]["cam2ego"], dtype=np.float64)
    return ego2global @ cam2ego


def run_temporal_inference(model, nusc, selected_samples, vehicle_label_ids):
    frame_records = []

    with tempfile.TemporaryDirectory(prefix="fcos3d_nuscenes_") as temp_dir:
        for order, (scene_sample_idx, sample) in enumerate(selected_samples):
            info, image_path = build_mmdet_sample_info(
                nusc, sample, sample_idx=scene_sample_idx
            )
            one_frame_info = Path(temp_dir) / f"frame_{scene_sample_idx:04d}.pkl"
            write_single_sample_info(info, one_frame_info)

            print(
                f"Frame {order + 1}/{len(selected_samples)} "
                f"(scene sample {scene_sample_idx}):\n  {image_path}"
            )
            result = inference_mono_3d_detector(
                model,
                image_path,
                str(one_frame_info),
                cam_type=CAM_TYPE,
            )
            pred = result.pred_instances_3d
            vehicles = extract_vehicles_from_prediction(
                pred, vehicle_label_ids, cfg=GRAPH_CONFIG
            )

            frame_records.append(
                {
                    "frame_index": scene_sample_idx,
                    "sample": info,
                    "image_path": image_path,
                    "pred": pred,
                    "vehicles": vehicles,
                    "cam_to_global": camera_to_global(info),
                }
            )
            print(
                f"  raw detections={len(pred.bboxes_3d)} | "
                f"usable vehicles={len(vehicles)}"
            )

    return frame_records


def main():
    if not NUSCENES_ROOT.is_dir():
        raise FileNotFoundError(
            f"nuScenes root does not exist: {NUSCENES_ROOT}\n"
            "Mount the dataset there or change NUSCENES_ROOT."
        )

    version = detect_nuscenes_version(NUSCENES_ROOT)
    nusc = NuScenes(version=version, dataroot=str(NUSCENES_ROOT), verbose=True)
    scene = choose_scene(nusc)
    samples = scene_samples(nusc, scene)
    selected_samples = select_temporal_window(samples)

    print("\n========================================")
    print("SCENE")
    print("========================================")
    print(f"root:        {NUSCENES_ROOT}")
    print(f"version:     {version}")
    print(f"name:        {scene['name']}")
    print(f"token:       {scene['token']}")
    print(f"description: {scene.get('description', '')}")
    print(f"keyframes:   {len(samples)}")
    print(
        f"window:      {selected_samples[0][0]}..{selected_samples[-1][0]} "
        f"({len(selected_samples)} frames)"
    )

    print("\nLoading FCOS3D...")
    model = init_model(CONFIG, CHECKPOINT, device=DEVICE)
    class_names = model.dataset_meta["classes"]
    vehicle_label_ids = {
        class_id
        for class_id, name in enumerate(class_names)
        if name in WANTED_VEHICLE_CLASSES
    }

    frame_records = run_temporal_inference(
        model, nusc, selected_samples, vehicle_label_ids
    )

    temporal_vehicles, tracks = accumulate_temporal_vehicle_evidence(
        frame_records, reference_record_index=-1, cfg=TEMPORAL_CONFIG
    )
    temporal_vehicles = [
        v for v in temporal_vehicles if 0.0 < v["z"] <= GRAPH_CONFIG.max_depth
    ]

    graph = build_lane_compatibility_graph_from_vehicles(
        temporal_vehicles, cfg=GRAPH_CONFIG
    )
    streams = get_lane_streams(graph, min_vehicles=2)
    lane_fits = fit_lane_streams(graph, streams, cfg=FIT_CONFIG)
    lane_boundaries = infer_lane_boundaries(lane_fits, cfg=BOUNDARY_CONFIG)

    print("\n========================================")
    print("TEMPORAL VEHICLE EVIDENCE")
    print("========================================")
    print(f"observations: {len(temporal_vehicles)}")
    print(f"tracks:       {len(tracks)}")
    print(f"graph nodes:  {graph.number_of_nodes()}")
    print(f"graph edges:  {graph.number_of_edges()}")
    print("\nCompatible temporal pairs:")
    print_graph_edges(graph)

    print("\n========================================")
    print("LANE FITS / BOUNDARIES")
    print("========================================")
    print(f"lane streams: {len(streams)}")
    print(f"lane fits:    {len(lane_fits)}")
    print(f"boundaries:   {len(lane_boundaries)}")

    reference_record = frame_records[-1]
    road_plane = estimate_road_plane(
        reference_record["pred"],
        reference_record["vehicles"],
        cfg=ROAD_PLANE_CONFIG,
    )

    projected_boundaries = []
    if road_plane is not None:
        cam2img = np.asarray(
            reference_record["sample"]["images"][CAM_TYPE]["cam2img"],
            dtype=np.float64,
        )
        image = plt.imread(reference_record["image_path"])
        projected_boundaries = project_lane_boundaries_to_image(
            lane_boundaries,
            road_plane,
            cam2img,
            image_shape=image.shape,
            cfg=PROJECTION_CONFIG,
        )
        plot_projected_lane_boundaries(
            reference_record["image_path"],
            projected_boundaries,
            save_path="temporal_lane_boundaries_image.png",
        )

    plot_lane_graph(
        graph,
        streams=streams,
        lane_fits=lane_fits,
        lane_boundaries=lane_boundaries,
        max_depth=GRAPH_CONFIG.max_depth,
        x_range=(-12.0, 12.0),
        save_path="temporal_lane_graph_bev.png",
    )


if __name__ == "__main__":
    main()
