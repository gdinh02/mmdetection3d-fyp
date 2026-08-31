import copy
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import mmengine
import numpy as np
from mmdet3d.apis import inference_mono_3d_detector, init_model

from lane_graph import (
    BoundaryTrackingConfig,
    LaneBoundaryConfig,
    LaneFitConfig,
    LaneGraphConfig,
    LaneMergeConfig,
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
    merge_compatible_lane_streams,
    plot_lane_graph,
    plot_projected_lane_boundaries,
    project_lane_boundaries_to_image,
    update_temporal_lane_tracks,
)


# -----------------------------------------------------------------------------
# SELF-CONTAINED SCENE INPUT
# -----------------------------------------------------------------------------
# The scene folder only needs:
#   scene-0095/
#       samples/
#       nuscenes_infos_scene-0095.pkl
#
# No nuScenes JSON tables and no maps/ folder are used by this script.
SCENE_ROOT = Path("/mnt/z/dataset/scene-0095")
SCENE_INFO_FILE = SCENE_ROOT / "nuscenes_infos_scene-0095.pkl"
OUTPUT_DIR = SCENE_ROOT / "lane_inference_output"

CONFIG = Path(
    "configs/fcos3d/"
    "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
)
CHECKPOINT = Path(
    "checkpoints/"
    "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_"
    "20210717_095645-8d806dc2.pth"
)

DEVICE = "cuda:0"
CAM_TYPE = "CAM_FRONT"
WANTED_VEHICLE_CLASSES = {"car", "truck", "bus"}


# -----------------------------------------------------------------------------
# OUTPUT / CACHE
# -----------------------------------------------------------------------------
DETECTION_CACHE_FILE = OUTPUT_DIR / "fcos3d_detection_cache.pkl"
LANE_RESULTS_FILE = OUTPUT_DIR / "scene_lane_results.pkl"
IMAGE_OVERLAY_DIR = OUTPUT_DIR / "image_overlays"
BEV_OVERLAY_DIR = OUTPUT_DIR / "bev_overlays"

# Set True only when the detector/checkpoint/cache extraction settings change.
FORCE_REBUILD_DETECTION_CACHE = False
CACHE_VERSION = 2

# Cache more detections than the graph currently uses so that graph score/depth
# thresholds can be tuned later without rerunning FCOS3D.
CACHE_SCORE_THRESH = 0.05
CACHE_MAX_DEPTH = 80.0

SAVE_IMAGE_OVERLAYS = True
SAVE_BEV_OVERLAYS = True
BEV_SHOW_LABELS = False


# -----------------------------------------------------------------------------
# LANE PIPELINE CONFIGURATION
# -----------------------------------------------------------------------------
GRAPH_CONFIG = LaneGraphConfig(
    score_thresh=0.30,
    max_depth=50.0,
    max_cross_track=1.5,
    max_yaw_diff_deg=15.0,
    max_along_track=25.0,
    sigma_cross_track=0.8,
    sigma_yaw_deg=3.0,
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

MERGE_CONFIG = LaneMergeConfig(
    max_longitudinal_gap=8.0,
    max_lateral_disagreement=1.0,
    max_tangent_diff_deg=10.0,
    sample_count=15,
    max_iterations=50,
)

BOUNDARY_CONFIG = LaneBoundaryConfig(
    min_overlap=3.0,
    min_lane_width=2.5,
    max_lane_width=5.0,
    sample_count=50,
    default_lane_width=3.5,
    single_stream_min_inliers=3,
    single_stream_max_rmse=0.75,
    single_stream_confidence=0.45,
    provisional_dedup_distance=0.75,
)

BOUNDARY_TRACKING_CONFIG = BoundaryTrackingConfig(
    min_overlap=3.0,
    max_lateral_distance=1.0,
    max_tangent_diff_deg=10.0,
    smoothing_alpha=0.30,
    max_missed_frames=2,
    missing_confidence_decay=0.75,
    sample_count=60,
    min_depth=1.0,
    min_points_after_transform=6,
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


# -----------------------------------------------------------------------------
# SCENE / PATH HELPERS
# -----------------------------------------------------------------------------
def _validate_scene_folder():
    if not SCENE_ROOT.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {SCENE_ROOT}")
    if not SCENE_INFO_FILE.is_file():
        raise FileNotFoundError(f"Scene info file not found: {SCENE_INFO_FILE}")
    if not (SCENE_ROOT / "samples").is_dir():
        raise FileNotFoundError(
            f"samples folder not found: {SCENE_ROOT / 'samples'}"
        )


def _normalise_relative_path(path_value):
    """Normalise Windows/POSIX paths stored inside a nuScenes info pickle."""
    return PurePosixPath(str(path_value).replace("\\", "/"))


def _resolve_scene_image(sample, cam_type):
    """Resolve an image using only the isolated SCENE_ROOT."""
    image_entry = sample["images"][cam_type]
    stored_path = _normalise_relative_path(image_entry["img_path"])

    candidate = SCENE_ROOT.joinpath(*stored_path.parts)
    if candidate.is_file():
        return candidate

    lower_parts = [part.lower() for part in stored_path.parts]
    if "samples" in lower_parts:
        samples_idx = lower_parts.index("samples")
        scene_relative = stored_path.parts[samples_idx:]
        candidate = SCENE_ROOT.joinpath(*scene_relative)
        if candidate.is_file():
            return candidate

    candidate = SCENE_ROOT / "samples" / cam_type / stored_path.name
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "Could not resolve the image inside the isolated scene folder.\n"
        f"Stored img_path: {image_entry['img_path']}\n"
        f"Scene root:      {SCENE_ROOT}\n"
        f"Expected under:  {SCENE_ROOT / 'samples' / cam_type}"
    )


def _camera_to_global(sample, cam_type):
    """Camera -> ego -> global transform from the scene-specific pickle."""
    ego2global = np.asarray(sample["ego2global"], dtype=np.float64)
    cam2ego = np.asarray(sample["images"][cam_type]["cam2ego"], dtype=np.float64)

    if ego2global.shape != (4, 4) or cam2ego.shape != (4, 4):
        raise ValueError("ego2global and cam2ego must both be 4x4 matrices")

    return ego2global @ cam2ego


def _available_scene_frames(scene_info):
    """Return all CAM_FRONT frames in data_list order."""
    return [
        (frame_idx, sample)
        for frame_idx, sample in enumerate(scene_info["data_list"])
        if CAM_TYPE in sample.get("images", {})
    ]


def _write_single_sample_info(scene_info, sample, image_path, destination):
    """
    Build the one-frame pkl required by inference_mono_3d_detector.

    The sample is copied and its image path is rewritten to the actual isolated
    scene path. No nuScenes JSON table or map file is accessed.
    """
    sample_copy = copy.deepcopy(sample)
    sample_copy["images"][CAM_TYPE]["img_path"] = str(image_path)

    one_frame_info = {
        key: value for key, value in scene_info.items() if key != "data_list"
    }
    one_frame_info["data_list"] = [sample_copy]
    mmengine.dump(one_frame_info, destination)


# -----------------------------------------------------------------------------
# FCOS3D DETECTION CACHE
# -----------------------------------------------------------------------------
def _expected_cache_metadata(frame_indices):
    return {
        "cache_version": CACHE_VERSION,
        "scene_root": str(SCENE_ROOT),
        "scene_info_file": str(SCENE_INFO_FILE),
        "cam_type": CAM_TYPE,
        "config": str(CONFIG),
        "checkpoint": str(CHECKPOINT),
        "wanted_vehicle_classes": sorted(WANTED_VEHICLE_CLASSES),
        "cache_score_thresh": float(CACHE_SCORE_THRESH),
        "cache_max_depth": float(CACHE_MAX_DEPTH),
        "frame_indices": list(frame_indices),
    }


def _cache_is_compatible(cache, frame_indices):
    if not isinstance(cache, dict) or "metadata" not in cache or "frames" not in cache:
        return False

    expected = _expected_cache_metadata(frame_indices)
    metadata = cache["metadata"]
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            return False

    cached_indices = [int(record["frame_index"]) for record in cache["frames"]]
    return cached_indices == list(frame_indices)


def _load_detection_cache(frame_indices):
    if FORCE_REBUILD_DETECTION_CACHE or not DETECTION_CACHE_FILE.is_file():
        return None

    print(f"Loading FCOS3D detection cache: {DETECTION_CACHE_FILE}")
    cache = mmengine.load(DETECTION_CACHE_FILE)
    if not _cache_is_compatible(cache, frame_indices):
        print("Existing cache metadata does not match this run; rebuilding it.")
        return None

    print(f"Loaded cached detections for {len(cache['frames'])} frames.")
    return cache


def _build_detection_cache(scene_info, scene_frames):
    print("No compatible cache found. Loading FCOS3D...")
    model = init_model(str(CONFIG), str(CHECKPOINT), device=DEVICE)
    print("FCOS3D loaded.")

    class_names = model.dataset_meta["classes"]
    vehicle_label_ids = {
        class_id
        for class_id, name in enumerate(class_names)
        if name in WANTED_VEHICLE_CLASSES
    }
    print(
        "Caching classes: "
        + ", ".join(class_names[class_id] for class_id in sorted(vehicle_label_ids))
    )

    cache_extract_cfg = LaneGraphConfig(
        score_thresh=CACHE_SCORE_THRESH,
        max_depth=CACHE_MAX_DEPTH,
        max_cross_track=GRAPH_CONFIG.max_cross_track,
        max_yaw_diff_deg=GRAPH_CONFIG.max_yaw_diff_deg,
        max_along_track=GRAPH_CONFIG.max_along_track,
        sigma_cross_track=GRAPH_CONFIG.sigma_cross_track,
        sigma_yaw_deg=GRAPH_CONFIG.sigma_yaw_deg,
    )

    records = []
    with tempfile.TemporaryDirectory(prefix="fcos3d_scene_cache_") as temp_dir:
        for order, (frame_idx, sample) in enumerate(scene_frames, start=1):
            image_path = _resolve_scene_image(sample, CAM_TYPE)
            temporary_info = Path(temp_dir) / f"frame_{frame_idx}.pkl"
            _write_single_sample_info(
                scene_info,
                sample,
                image_path,
                temporary_info,
            )

            print(
                f"FCOS3D {order}/{len(scene_frames)} "
                f"(data_list[{frame_idx}]): {image_path.name}"
            )

            result = inference_mono_3d_detector(
                model,
                str(image_path),
                str(temporary_info),
                cam_type=CAM_TYPE,
            )
            pred = result.pred_instances_3d

            cached_vehicles = extract_vehicles_from_prediction(
                pred,
                vehicle_label_ids,
                cfg=cache_extract_cfg,
            )

            # Estimate the road plane from the subset that would currently be
            # accepted by the lane graph rather than very weak cached boxes.
            road_plane_vehicles = [
                vehicle
                for vehicle in cached_vehicles
                if (
                    vehicle["score"] >= GRAPH_CONFIG.score_thresh
                    and 0.0 < vehicle["z"] <= GRAPH_CONFIG.max_depth
                )
            ]
            road_plane = estimate_road_plane(
                pred,
                road_plane_vehicles,
                cfg=ROAD_PLANE_CONFIG,
            )

            records.append(
                {
                    "frame_index": int(frame_idx),
                    "image_path": str(image_path),
                    "cam_to_global": _camera_to_global(sample, CAM_TYPE),
                    "cam2img": np.asarray(
                        sample["images"][CAM_TYPE]["cam2img"], dtype=np.float64
                    ),
                    "road_plane": road_plane,
                    "vehicles": cached_vehicles,
                    "raw_detection_count": int(len(pred.bboxes_3d)),
                }
            )

            print(
                f"  raw={len(pred.bboxes_3d)} | "
                f"cached vehicle detections={len(cached_vehicles)} | "
                f"road_plane={'yes' if road_plane is not None else 'no'}"
            )

    frame_indices = [frame_idx for frame_idx, _ in scene_frames]
    cache = {
        "metadata": _expected_cache_metadata(frame_indices),
        "class_names": list(class_names),
        "frames": records,
    }
    mmengine.dump(cache, DETECTION_CACHE_FILE)
    print(f"Saved FCOS3D cache: {DETECTION_CACHE_FILE}")
    return cache


def _get_or_build_detection_cache(scene_info, scene_frames):
    frame_indices = [frame_idx for frame_idx, _ in scene_frames]
    cache = _load_detection_cache(frame_indices)
    if cache is not None:
        return cache
    return _build_detection_cache(scene_info, scene_frames)


def _filter_cached_vehicles(vehicles):
    """Apply current graph score/depth settings to cached detections."""
    filtered = []
    for vehicle in vehicles:
        if vehicle["score"] < GRAPH_CONFIG.score_thresh:
            continue
        if not (0.0 < vehicle["z"] <= GRAPH_CONFIG.max_depth):
            continue
        updated = dict(vehicle)
        updated["evidence_weight"] = 1.0
        filtered.append(updated)
    return filtered


def _runtime_record(cache_record):
    """Minimal frame record expected by temporal vehicle accumulation."""
    return {
        "frame_index": int(cache_record["frame_index"]),
        "vehicles": _filter_cached_vehicles(cache_record["vehicles"]),
        "cam_to_global": np.asarray(cache_record["cam_to_global"], dtype=np.float64),
    }


# -----------------------------------------------------------------------------
# ROLLING LANE INFERENCE
# -----------------------------------------------------------------------------
def _infer_lane_window(window_records):
    temporal_vehicles, vehicle_tracks = accumulate_temporal_vehicle_evidence(
        window_records,
        reference_record_index=-1,
        cfg=TEMPORAL_CONFIG,
    )
    temporal_vehicles = [
        vehicle
        for vehicle in temporal_vehicles
        if 0.0 < vehicle["z"] <= GRAPH_CONFIG.max_depth
    ]

    graph = build_lane_compatibility_graph_from_vehicles(
        temporal_vehicles,
        cfg=GRAPH_CONFIG,
    )
    initial_streams = get_lane_streams(graph, min_vehicles=2)
    initial_lane_fits = fit_lane_streams(
        graph,
        initial_streams,
        cfg=FIT_CONFIG,
    )

    streams, lane_fits, merge_events = merge_compatible_lane_streams(
        graph,
        initial_streams,
        lane_fits=initial_lane_fits,
        merge_cfg=MERGE_CONFIG,
        fit_cfg=FIT_CONFIG,
    )

    raw_boundaries = infer_lane_boundaries(
        lane_fits,
        cfg=BOUNDARY_CONFIG,
    )

    return {
        "temporal_vehicles": temporal_vehicles,
        "vehicle_tracks": vehicle_tracks,
        "graph": graph,
        "initial_streams": initial_streams,
        "initial_lane_fits": initial_lane_fits,
        "streams": streams,
        "lane_fits": lane_fits,
        "merge_events": merge_events,
        "raw_boundaries": raw_boundaries,
    }


def _projection_road_plane(current_road_plane, boundary_state):
    """Prefer the measured plane; otherwise use a tracked plane in this frame."""
    if current_road_plane is not None:
        return current_road_plane

    if not boundary_state:
        return None

    tracks = boundary_state.get("tracks", {})
    if not tracks:
        return None

    # Prefer the plane associated with the strongest currently active boundary.
    candidates = [
        track
        for track in tracks.values()
        if track.get("road_plane") is not None
    ]
    if not candidates:
        return None

    candidates.sort(
        key=lambda track: float(track["boundary"].get("confidence", 0.0)),
        reverse=True,
    )
    return candidates[0]["road_plane"]


def _save_frame_visualisations(
    cache_record,
    inference,
    tracked_boundaries,
    projection_road_plane,
):
    frame_index = int(cache_record["frame_index"])
    image_path = Path(cache_record["image_path"])
    image_output = IMAGE_OVERLAY_DIR / f"frame_{frame_index:06d}.png"
    bev_output = BEV_OVERLAY_DIR / f"frame_{frame_index:06d}.png"

    projected_boundaries = []
    if projection_road_plane is not None and tracked_boundaries:
        image = plt.imread(image_path)
        projected_boundaries = project_lane_boundaries_to_image(
            tracked_boundaries,
            projection_road_plane,
            np.asarray(cache_record["cam2img"], dtype=np.float64),
            image_shape=image.shape,
            cfg=PROJECTION_CONFIG,
        )

    if SAVE_IMAGE_OVERLAYS:
        plot_projected_lane_boundaries(
            image_path,
            projected_boundaries,
            save_path=image_output,
            show=False,
            title=f"Frame {frame_index}: temporally tracked lane boundaries",
        )

    if SAVE_BEV_OVERLAYS:
        plot_lane_graph(
            inference["graph"],
            streams=inference["streams"],
            lane_fits=inference["lane_fits"],
            lane_boundaries=tracked_boundaries,
            max_depth=GRAPH_CONFIG.max_depth,
            x_range=(-12.0, 12.0),
            show_labels=BEV_SHOW_LABELS,
            save_path=bev_output,
            show=False,
        )

    return projected_boundaries, image_output, bev_output


def _compact_frame_result(
    cache_record,
    window_records,
    inference,
    tracked_boundaries,
    boundary_events,
    projection_road_plane,
    projected_boundaries,
):
    return {
        "frame_index": int(cache_record["frame_index"]),
        "image_path": str(cache_record["image_path"]),
        "window_frame_indices": [int(record["frame_index"]) for record in window_records],
        "num_temporal_vehicle_observations": len(inference["temporal_vehicles"]),
        "num_vehicle_tracks": len(inference["vehicle_tracks"]),
        "num_graph_nodes": inference["graph"].number_of_nodes(),
        "num_graph_edges": inference["graph"].number_of_edges(),
        "initial_lane_fits": inference["initial_lane_fits"],
        "lane_fits": inference["lane_fits"],
        "merge_events": inference["merge_events"],
        "raw_boundaries": inference["raw_boundaries"],
        "tracked_boundaries": tracked_boundaries,
        "boundary_events": boundary_events,
        "road_plane": projection_road_plane,
        "projected_boundary_count": len(projected_boundaries),
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    _validate_scene_folder()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    BEV_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    print("Scene-wide FCOS3D lane inference")
    print(f"Scene root: {SCENE_ROOT}")
    print(f"Info file:  {SCENE_INFO_FILE}")
    print("No nuScenes JSON tables or map files will be loaded.\n")

    scene_info = mmengine.load(SCENE_INFO_FILE)
    if "data_list" not in scene_info or not scene_info["data_list"]:
        raise ValueError("Scene info .pkl contains no data_list entries")

    scene_frames = _available_scene_frames(scene_info)
    if not scene_frames:
        raise RuntimeError(f"No {CAM_TYPE} frames found in scene info")

    print(f"Scene samples in .pkl: {len(scene_info['data_list'])}")
    print(f"{CAM_TYPE} frames:       {len(scene_frames)}")
    print(f"Temporal history:       {TEMPORAL_CONFIG.history_frames} frames\n")

    cache = _get_or_build_detection_cache(scene_info, scene_frames)
    cache_frames = cache["frames"]

    boundary_state = None
    scene_results = []
    last_image_output = None
    last_bev_output = None

    print("\n========================================")
    print("ROLLING SCENE LANE INFERENCE")
    print("========================================")

    for position, cache_record in enumerate(cache_frames):
        current_frame_index = int(cache_record["frame_index"])
        start = max(0, position - TEMPORAL_CONFIG.history_frames + 1)
        cache_window = cache_frames[start : position + 1]
        window_records = [_runtime_record(record) for record in cache_window]

        print(
            f"\nFrame {position + 1}/{len(cache_frames)} "
            f"data_list[{current_frame_index}] | "
            f"window={[record['frame_index'] for record in window_records]}"
        )

        inference = _infer_lane_window(window_records)

        tracked_boundaries, boundary_state, boundary_events = (
            update_temporal_lane_tracks(
                boundary_state,
                inference["raw_boundaries"],
                np.asarray(cache_record["cam_to_global"], dtype=np.float64),
                cache_record.get("road_plane"),
                current_frame_index,
                cfg=BOUNDARY_TRACKING_CONFIG,
            )
        )

        projection_road_plane = _projection_road_plane(
            cache_record.get("road_plane"),
            boundary_state,
        )

        projected_boundaries, image_output, bev_output = _save_frame_visualisations(
            cache_record,
            inference,
            tracked_boundaries,
            projection_road_plane,
        )
        last_image_output = image_output
        last_bev_output = bev_output

        matched = sum(event["type"] == "matched" for event in boundary_events)
        new = sum(event["type"] == "new" for event in boundary_events)
        predicted = sum(event["type"] == "predicted" for event in boundary_events)

        print(
            f"  observations={len(inference['temporal_vehicles'])} | "
            f"vehicle_tracks={len(inference['vehicle_tracks'])} | "
            f"fits={len(inference['lane_fits'])} | "
            f"raw_boundaries={len(inference['raw_boundaries'])} | "
            f"tracked={len(tracked_boundaries)} "
            f"(matched={matched}, new={new}, predicted={predicted})"
        )

        scene_results.append(
            _compact_frame_result(
                cache_record,
                window_records,
                inference,
                tracked_boundaries,
                boundary_events,
                projection_road_plane,
                projected_boundaries,
            )
        )

    result_payload = {
        "metadata": {
            "scene_root": str(SCENE_ROOT),
            "scene_info_file": str(SCENE_INFO_FILE),
            "cam_type": CAM_TYPE,
            "history_frames": TEMPORAL_CONFIG.history_frames,
            "boundary_smoothing_alpha": BOUNDARY_TRACKING_CONFIG.smoothing_alpha,
            "max_boundary_missed_frames": BOUNDARY_TRACKING_CONFIG.max_missed_frames,
        },
        "frames": scene_results,
    }
    mmengine.dump(result_payload, LANE_RESULTS_FILE)

    # Preserve the old single-result filenames as aliases for the final frame.
    if SAVE_IMAGE_OVERLAYS and last_image_output and last_image_output.is_file():
        shutil.copy2(
            last_image_output,
            OUTPUT_DIR / "temporal_lane_boundaries_image.png",
        )
    if SAVE_BEV_OVERLAYS and last_bev_output and last_bev_output.is_file():
        shutil.copy2(
            last_bev_output,
            OUTPUT_DIR / "temporal_lane_graph_bev.png",
        )

    total_tracked = sum(len(frame["tracked_boundaries"]) for frame in scene_results)
    frames_with_boundaries = sum(bool(frame["tracked_boundaries"]) for frame in scene_results)

    print("\n========================================")
    print("SCENE COMPLETE")
    print("========================================")
    print(f"Processed frames:       {len(scene_results)}")
    print(f"Frames with boundaries: {frames_with_boundaries}")
    print(f"Boundary instances:     {total_tracked}")
    print(f"Detection cache:        {DETECTION_CACHE_FILE}")
    print(f"Lane results:           {LANE_RESULTS_FILE}")
    print(f"Image overlays:         {IMAGE_OVERLAY_DIR}")
    print(f"BEV overlays:           {BEV_OVERLAY_DIR}")


if __name__ == "__main__":
    main()
