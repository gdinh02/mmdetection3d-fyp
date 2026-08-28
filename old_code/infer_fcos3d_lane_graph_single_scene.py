import tempfile
from pathlib import Path, PurePosixPath

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


# -----------------------------------------------------------------------------
# SELF-CONTAINED SCENE INPUT
# -----------------------------------------------------------------------------
# The scene folder only needs:
#   scene-0064/
#       samples/
#       nuscenes_infos_scene-0064.pkl
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
REFERENCE_FRAME_INDEX = -1
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
)

FIT_CONFIG = LaneFitConfig(
    degree=2,
    residual_threshold=0.75,
    max_trials=200,
    random_seed=0,
)

BOUNDARY_CONFIG = LaneBoundaryConfig(
    min_overlap=3.0,
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


def _validate_scene_folder():
    if not SCENE_ROOT.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {SCENE_ROOT}")
    if not SCENE_INFO_FILE.is_file():
        raise FileNotFoundError(f"Scene info file not found: {SCENE_INFO_FILE}")
    if not (SCENE_ROOT / "samples").is_dir():
        raise FileNotFoundError(f"samples folder not found: {SCENE_ROOT / 'samples'}")


def _resolve_reference_index(length, index):
    resolved = index if index >= 0 else length + index
    if resolved < 0 or resolved >= length:
        raise IndexError(f"REFERENCE_FRAME_INDEX {index} is outside 0..{length - 1}")
    return resolved


def _normalise_relative_path(path_value):
    """Normalise Windows/POSIX paths stored inside a nuScenes info pickle."""
    text = str(path_value).replace("\\", "/")
    return PurePosixPath(text)


def _resolve_scene_image(sample, cam_type):
    """
    Resolve an image using only SCENE_ROOT.

    The scene-specific pickle may still contain paths from the original full
    nuScenes dataset. If so, this function discards everything before the
    'samples/' component and remaps the suffix into SCENE_ROOT/samples/.
    """
    image_entry = sample["images"][cam_type]
    stored_path = _normalise_relative_path(image_entry["img_path"])

    # 1. Already relative to the isolated scene root.
    candidate = SCENE_ROOT.joinpath(*stored_path.parts)
    if candidate.is_file():
        return candidate

    # 2. Strip any old full-dataset prefix before 'samples/'.
    lower_parts = [part.lower() for part in stored_path.parts]
    if "samples" in lower_parts:
        samples_idx = lower_parts.index("samples")
        scene_relative = stored_path.parts[samples_idx:]
        candidate = SCENE_ROOT.joinpath(*scene_relative)
        if candidate.is_file():
            return candidate

    # 3. Final scene-local fallback: CAM folder + filename.
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


def _select_temporal_samples(scene_info, reference_index, history_frames):
    """Take the reference frame and its preceding frames from this one scene."""
    data_list = scene_info["data_list"]
    ref_idx = _resolve_reference_index(len(data_list), reference_index)
    start = max(0, ref_idx - history_frames + 1)

    selected = [
        (frame_idx, data_list[frame_idx])
        for frame_idx in range(start, ref_idx + 1)
        if CAM_TYPE in data_list[frame_idx].get("images", {})
    ]

    if not selected:
        raise RuntimeError(f"No {CAM_TYPE} frames were found in the selected window")

    return selected, ref_idx


def _write_single_sample_info(scene_info, sample, destination):
    """
    Build the tiny one-frame .pkl required by inference_mono_3d_detector.

    This is generated from the existing scene .pkl; it does not access any
    nuScenes JSON table or map file.
    """
    one_frame_info = {
        key: value for key, value in scene_info.items() if key != "data_list"
    }
    one_frame_info["data_list"] = [sample]
    mmengine.dump(one_frame_info, destination)


def _run_scene_inference(model, scene_info, selected_frames, vehicle_label_ids):
    frame_records = []

    with tempfile.TemporaryDirectory(prefix="fcos3d_scene_") as temp_dir:
        for order, (frame_idx, sample) in enumerate(selected_frames, start=1):
            image_path = _resolve_scene_image(sample, CAM_TYPE)
            temporary_info = Path(temp_dir) / f"frame_{frame_idx}.pkl"
            _write_single_sample_info(scene_info, sample, temporary_info)

            print(
                f"Frame {order}/{len(selected_frames)} "
                f"(data_list[{frame_idx}]): {image_path.name}"
            )

            result = inference_mono_3d_detector(
                model,
                str(image_path),
                str(temporary_info),
                cam_type=CAM_TYPE,
            )
            pred = result.pred_instances_3d
            vehicles = extract_vehicles_from_prediction(
                pred,
                vehicle_label_ids,
                cfg=GRAPH_CONFIG,
            )

            frame_records.append(
                {
                    "frame_index": frame_idx,
                    "sample": sample,
                    "image_path": image_path,
                    "pred": pred,
                    "vehicles": vehicles,
                    "cam_to_global": _camera_to_global(sample, CAM_TYPE),
                }
            )

            print(
                f"  raw detections={len(pred.bboxes_3d)} | "
                f"usable vehicles={len(vehicles)}"
            )

    return frame_records


def main():
    _validate_scene_folder()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scene-only FCOS3D lane inference")
    print(f"Scene root: {SCENE_ROOT}")
    print(f"Info file:  {SCENE_INFO_FILE}")
    print("No nuScenes JSON tables or map files will be loaded.\n")

    print("Loading FCOS3D...")
    model = init_model(str(CONFIG), str(CHECKPOINT), device=DEVICE)
    print("Model loaded.")

    class_names = model.dataset_meta["classes"]
    vehicle_label_ids = {
        class_id
        for class_id, name in enumerate(class_names)
        if name in WANTED_VEHICLE_CLASSES
    }

    scene_info = mmengine.load(SCENE_INFO_FILE)
    if "data_list" not in scene_info or not scene_info["data_list"]:
        raise ValueError("Scene info .pkl contains no data_list entries")

    selected_frames, reference_index = _select_temporal_samples(
        scene_info,
        REFERENCE_FRAME_INDEX,
        TEMPORAL_CONFIG.history_frames,
    )

    print("========================================")
    print("TEMPORAL WINDOW")
    print("========================================")
    print(f"Scene samples in .pkl: {len(scene_info['data_list'])}")
    print(f"Selected frames:       {len(selected_frames)}")
    print(f"Reference frame:       data_list[{reference_index}]")

    frame_records = _run_scene_inference(
        model,
        scene_info,
        selected_frames,
        vehicle_label_ids,
    )

    temporal_vehicles, tracks = accumulate_temporal_vehicle_evidence(
        frame_records,
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
    streams = get_lane_streams(graph, min_vehicles=2)
    lane_fits = fit_lane_streams(graph, streams, cfg=FIT_CONFIG)
    lane_boundaries = infer_lane_boundaries(lane_fits, cfg=BOUNDARY_CONFIG)

    print("\n========================================")
    print("TEMPORAL VEHICLE EVIDENCE")
    print("========================================")
    print(f"Observations: {len(temporal_vehicles)}")
    print(f"Tracks:       {len(tracks)}")
    print(f"Graph nodes:  {graph.number_of_nodes()}")
    print(f"Graph edges:  {graph.number_of_edges()}")

    print("\nCompatible temporal pairs:")
    print_graph_edges(graph)

    print("\n========================================")
    print("LANE FITS")
    print("========================================")
    for fit in lane_fits:
        print(
            f"Stream {fit['stream_id']}: "
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

    reference_record = frame_records[-1]
    road_plane = estimate_road_plane(
        reference_record["pred"],
        reference_record["vehicles"],
        cfg=ROAD_PLANE_CONFIG,
    )

    projected_boundaries = []
    if road_plane is not None and lane_boundaries:
        cam2img = np.asarray(
            reference_record["sample"]["images"][CAM_TYPE]["cam2img"]
        )
        image = plt.imread(reference_record["image_path"])
        projected_boundaries = project_lane_boundaries_to_image(
            lane_boundaries,
            road_plane,
            cam2img,
            image_shape=image.shape,
            cfg=PROJECTION_CONFIG,
        )

        if projected_boundaries:
            plot_projected_lane_boundaries(
                reference_record["image_path"],
                projected_boundaries,
                save_path=OUTPUT_DIR / "temporal_lane_boundaries_image.png",
            )

    plot_lane_graph(
        graph,
        streams=streams,
        lane_fits=lane_fits,
        lane_boundaries=lane_boundaries,
        max_depth=GRAPH_CONFIG.max_depth,
        x_range=(-12.0, 12.0),
        save_path=OUTPUT_DIR / "temporal_lane_graph_bev.png",
    )

    print("\nOutput directory:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
