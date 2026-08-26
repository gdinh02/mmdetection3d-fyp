import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mmengine
import numpy as np

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

# This can be the existing one-frame demo info or a normal nuScenes info file
# containing many consecutive samples. With one sample, the script naturally
# falls back to single-frame behaviour.
SEQUENCE_INFO_FILE = (
    "demo/data/nuscenes/"
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.pkl"
)
DATA_ROOT = "data/nuscenes"
REFERENCE_FRAME_INDEX = -1

DEVICE = "cuda:0"
CAM_TYPE = "CAM_FRONT"
WANTED_VEHICLE_CLASSES = {"car", "truck", "bus"}

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
    max_reference_distance=60.0,
    max_time_gap_s=3.0,
)

FIT_CONFIG = LaneFitConfig(
    degree=2,
    residual_threshold=0.75,
    max_trials=200,
    random_seed=0,
)

BOUNDARY_CONFIG = LaneBoundaryConfig(
    min_overlap=10.0,
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


def _resolve_reference_index(length, index):
    resolved = index if index >= 0 else length + index
    if resolved < 0 or resolved >= length:
        raise IndexError(f"REFERENCE_FRAME_INDEX {index} is outside 0..{length - 1}")
    return resolved


def _camera_to_global(sample, cam_type):
    """Return the 4x4 camera-to-global pose for one nuScenes sample."""
    ego2global = np.asarray(sample["ego2global"], dtype=np.float64)
    cam2ego = np.asarray(sample["images"][cam_type]["cam2ego"], dtype=np.float64)

    if ego2global.shape != (4, 4) or cam2ego.shape != (4, 4):
        raise ValueError("ego2global and cam2ego must both be 4x4 matrices")

    return ego2global @ cam2ego


def _timestamp_seconds(sample):
    timestamp = sample.get("timestamp")
    if timestamp is None:
        return None

    # nuScenes timestamps are normally in microseconds.
    timestamp = float(timestamp)
    return timestamp / 1e6 if abs(timestamp) > 1e9 else timestamp


def _select_temporal_samples(sequence_info, reference_index, cfg):
    """Select recent frames that plausibly belong to the reference scene."""
    data_list = sequence_info["data_list"]
    ref_idx = _resolve_reference_index(len(data_list), reference_index)
    reference = data_list[ref_idx]
    reference_pose = np.asarray(reference["ego2global"], dtype=np.float64)
    reference_position = reference_pose[:3, 3]
    reference_time = _timestamp_seconds(reference)
    reference_scene = reference.get("scene_token")

    start = max(0, ref_idx - cfg.history_frames + 1)
    selected = []

    for frame_idx in range(start, ref_idx + 1):
        sample = data_list[frame_idx]

        if CAM_TYPE not in sample.get("images", {}):
            continue

        sample_scene = sample.get("scene_token")
        if reference_scene is not None and sample_scene is not None:
            if sample_scene != reference_scene:
                continue

        pose = np.asarray(sample["ego2global"], dtype=np.float64)
        ego_distance = float(np.linalg.norm(pose[:3, 3] - reference_position))
        if ego_distance > cfg.max_reference_distance:
            continue

        sample_time = _timestamp_seconds(sample)
        if reference_time is not None and sample_time is not None:
            if abs(reference_time - sample_time) > cfg.max_time_gap_s:
                continue

        selected.append((frame_idx, sample))

    if not selected or selected[-1][0] != ref_idx:
        selected.append((ref_idx, reference))

    selected.sort(key=lambda item: item[0])
    return selected, ref_idx


def _resolve_image_path(sample, cam_type, info_file, data_root):
    image_path = Path(sample["images"][cam_type]["img_path"])
    candidates = [
        image_path,
        Path(data_root) / image_path,
        Path(info_file).resolve().parent / image_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve image path from info entry. Tried:\n  "
        + "\n  ".join(str(candidate) for candidate in candidates)
    )


def _write_single_sample_info(sequence_info, sample, path):
    """Create the one-sample annotation file expected by mono-3D inference."""
    single = {
        key: value
        for key, value in sequence_info.items()
        if key != "data_list"
    }
    single["data_list"] = [sample]
    mmengine.dump(single, path)


def _run_temporal_inference(model, sequence_info, selected_frames, vehicle_label_ids):
    frame_records = []

    with tempfile.TemporaryDirectory(prefix="fcos3d_temporal_") as temp_dir:
        for order, (frame_idx, sample) in enumerate(selected_frames):
            image_path = _resolve_image_path(
                sample,
                CAM_TYPE,
                SEQUENCE_INFO_FILE,
                DATA_ROOT,
            )
            one_frame_info = Path(temp_dir) / f"frame_{frame_idx}.pkl"
            _write_single_sample_info(sequence_info, sample, one_frame_info)

            print(
                f"Frame {order + 1}/{len(selected_frames)} "
                f"(data_list[{frame_idx}]): {image_path}"
            )
            result = inference_mono_3d_detector(
                model,
                image_path,
                str(one_frame_info),
                cam_type=CAM_TYPE,
            )
            pred = result.pred_instances_3d
            vehicles = extract_vehicles_from_prediction(
                pred,
                vehicle_label_ids,
                cfg=GRAPH_CONFIG,
            )

            frame_records.append({
                "frame_index": frame_idx,
                "sample": sample,
                "image_path": image_path,
                "pred": pred,
                "vehicles": vehicles,
                "cam_to_global": _camera_to_global(sample, CAM_TYPE),
            })
            print(
                f"  raw detections={len(pred.bboxes_3d)} | "
                f"usable vehicles={len(vehicles)}"
            )

    return frame_records


def main():
    print("Loading FCOS3D...")
    model = init_model(CONFIG, CHECKPOINT, device=DEVICE)
    print("Model loaded.")

    class_names = model.dataset_meta["classes"]
    vehicle_label_ids = {
        class_id
        for class_id, name in enumerate(class_names)
        if name in WANTED_VEHICLE_CLASSES
    }
    print("Vehicle class IDs:", vehicle_label_ids)

    sequence_info = mmengine.load(SEQUENCE_INFO_FILE)
    selected_frames, reference_index = _select_temporal_samples(
        sequence_info,
        REFERENCE_FRAME_INDEX,
        TEMPORAL_CONFIG,
    )

    print("\n========================================")
    print("TEMPORAL WINDOW")
    print("========================================")
    print(f"Selected frames: {len(selected_frames)}")
    print(f"Reference index: data_list[{reference_index}]")

    frame_records = _run_temporal_inference(
        model,
        sequence_info,
        selected_frames,
        vehicle_label_ids,
    )

    temporal_vehicles, tracks = accumulate_temporal_vehicle_evidence(
        frame_records,
        reference_record_index=-1,
        cfg=TEMPORAL_CONFIG,
    )

    # Only evidence visible in front of the reference camera contributes to
    # the reference-frame lane model.
    temporal_vehicles = [
        vehicle
        for vehicle in temporal_vehicles
        if 0.0 < vehicle["z"] <= GRAPH_CONFIG.max_depth
    ]

    graph = build_lane_compatibility_graph_from_vehicles(
        temporal_vehicles,
        cfg=GRAPH_CONFIG,
    )

    print("\n========================================")
    print("TEMPORAL VEHICLE EVIDENCE")
    print("========================================")
    print(f"Accumulated observations: {len(temporal_vehicles)}")
    print(f"Tracks created:           {len(tracks)}")
    print(f"Graph nodes:              {graph.number_of_nodes()}")
    print(f"Graph edges:              {graph.number_of_edges()}")

    for track_id, track in sorted(tracks.items()):
        print(
            f"track {track_id:2d}: "
            f"observations={track['observations']} | "
            f"class={class_names[track['label']]}"
        )

    print("\nCompatible temporal pairs:")
    print_graph_edges(graph)

    streams = get_lane_streams(graph, min_vehicles=2)
    lane_fits = fit_lane_streams(graph, streams, cfg=FIT_CONFIG)
    lane_boundaries = infer_lane_boundaries(lane_fits, cfg=BOUNDARY_CONFIG)

    print("\n========================================")
    print("TEMPORAL LANE FITS")
    print("========================================")
    if not lane_fits:
        print("No fitted lane streams found.")
    else:
        for fit in lane_fits:
            coeff_text = ", ".join(f"{value:.5f}" for value in fit["coefficients"])
            print(
                f"Stream {fit['stream_id']}: "
                f"degree={fit['degree']} | "
                f"coefficients=[{coeff_text}] | "
                f"RMSE={fit['rmse']:.3f} m | "
                f"inliers={len(fit['inliers'])} | "
                f"outliers={len(fit['outliers'])}"
            )

    print("\n========================================")
    print("INFERRED LANE BOUNDARIES")
    print("========================================")
    if not lane_boundaries:
        print("No adjacent fitted streams passed the boundary checks.")
    else:
        for boundary_id, boundary in enumerate(lane_boundaries):
            print(
                f"Boundary {boundary_id}: "
                f"S{boundary['left_stream_id']}|S{boundary['right_stream_id']} | "
                f"width={boundary['lane_width']:.2f} m | "
                f"overlap={boundary['overlap']:.1f} m"
            )

    # Everything above is expressed in the newest/reference camera frame, so
    # estimate the road plane and project onto that same reference image.
    reference_record = frame_records[-1]
    road_plane = estimate_road_plane(
        pred_instances_3d=reference_record["pred"],
        vehicles=reference_record["vehicles"],
        cfg=ROAD_PLANE_CONFIG,
    )

    projected_boundaries = []
    if road_plane is None:
        print("\nCould not estimate the reference-frame road plane.")
    else:
        sample = reference_record["sample"]
        cam2img = np.asarray(sample["images"][CAM_TYPE]["cam2img"])
        image = plt.imread(reference_record["image_path"])
        projected_boundaries = project_lane_boundaries_to_image(
            lane_boundaries=lane_boundaries,
            road_plane=road_plane,
            cam2img=cam2img,
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

    print("\nSaved:")
    print("  temporal_lane_graph_bev.png")
    if projected_boundaries:
        print("  temporal_lane_boundaries_image.png")


if __name__ == "__main__":
    main()
