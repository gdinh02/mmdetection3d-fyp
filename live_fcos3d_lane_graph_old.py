import copy
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

import cv2
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
    project_lane_boundaries_to_image,
    update_temporal_lane_tracks,
)


# =============================================================================
# INPUT MODE
# =============================================================================
# "scene_replay": replay a locally stored nuScenes scene in sequence.
# "camera":       webcam / RTSP / video source.
INPUT_MODE = "scene_replay"


# =============================================================================
# FCOS3D MODEL
# =============================================================================
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


# =============================================================================
# SCENE REPLAY SETTINGS
# =============================================================================
# The scene folder only needs samples/ plus the scene-specific info pickle.
SCENE_ROOT = Path("/mnt/z/dataset/scene-0095")
SCENE_INFO_FILE = SCENE_ROOT / "nuscenes_infos_scene-0095.pkl"

# Playback modes:
#   "realtime"  -> preserve timing from scene timestamps
#   "fixed_fps" -> replay at REPLAY_FPS
#   "max_speed" -> immediately feed the next frame after inference finishes
REPLAY_MODE = "realtime"
REPLAY_FPS = 10.0
REPLAY_SPEED = 1.0       # 2.0 = 2x timestamp speed in realtime mode
LOOP_SCENE = False
REPLAY_START_INDEX = 0
REPLAY_END_INDEX = None  # inclusive data_list index; None = end of scene


# =============================================================================
# REAL CAMERA / RTSP SETTINGS
# =============================================================================
# Examples: 0, 1, "rtsp://...", "/path/to/video.mp4"
VIDEO_SOURCE = 0
PROCESS_EVERY_NTH_FRAME = 1
DROP_STALE_FRAMES = True

# Used only for INPUT_MODE == "camera". Create this once with the supplied
# make_live_calibration.py script.
CALIBRATION_FILE = Path("live_camera_calibration.npz")

# For a fixed camera identity pose is valid. For a moving physical camera,
# implement get_custom_cam_to_global() and set POSE_MODE = "custom".
# With POSE_MODE = "none", temporal history is reduced to one frame.
POSE_MODE = "none"  # "static", "none", or "custom"


def get_custom_cam_to_global(frame_id, timestamp_s):
    raise NotImplementedError(
        "Implement get_custom_cam_to_global() from live odometry/SLAM/GNSS/INS."
    )


def get_camera_cam_to_global(frame_id, timestamp_s):
    if POSE_MODE in {"static", "none"}:
        return np.eye(4, dtype=np.float64)
    if POSE_MODE == "custom":
        pose = np.asarray(
            get_custom_cam_to_global(frame_id, timestamp_s), dtype=np.float64
        )
        if pose.shape != (4, 4):
            raise ValueError("Custom camera pose must have shape (4, 4)")
        return pose
    raise ValueError("POSE_MODE must be 'static', 'none', or 'custom'")


# =============================================================================
# DISPLAY / OUTPUT
# =============================================================================
DISPLAY_WINDOW = True
WINDOW_NAME = "Live / replay FCOS3D lane inference"
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = SCENE_ROOT/"live_lane_inference.mp4"
TEMP_JPEG_QUALITY = 95


# =============================================================================
# LANE PIPELINE CONFIGURATION
# =============================================================================
GRAPH_CONFIG = LaneGraphConfig(
    score_thresh=0.30,
    max_depth=50.0,
    max_cross_track=1.0,
    max_yaw_diff_deg=10.0,
    max_along_track=25.0,
    sigma_cross_track=0.8,
    sigma_yaw_deg=5.0,
)

TEMPORAL_CONFIG = TemporalConfig(
    history_frames=5,
    max_track_distance=12.0,
    max_track_yaw_diff_deg=30.0,
    max_track_frame_gap=1,
    temporal_decay=0.85,
    min_track_observations=2,
)

FIT_CONFIG = LaneFitConfig(
    degree=2,
    residual_threshold=0.50,
    max_trials=200,
    random_seed=0,
)

MERGE_CONFIG = LaneMergeConfig(
    max_longitudinal_gap=12.0,
    max_lateral_disagreement=0.75,
    max_tangent_diff_deg=8.0,
    sample_count=15,
    max_iterations=50,
)

BOUNDARY_CONFIG = LaneBoundaryConfig(
    min_overlap=6.0,
    min_lane_width=2.8,
    max_lane_width=4.2,
    sample_count=30,
    enable_single_stream_boundaries=True,
)

ROAD_PLANE_CONFIG = RoadPlaneConfig(
    residual_threshold=0.35,
    max_trials=200,
    random_seed=0,
)

PROJECTION_CONFIG = LaneProjectionConfig(
    sample_count=80,
    min_depth=1.0,
    clip_to_image=True,
)

BOUNDARY_TRACKING_CONFIG = BoundaryTrackingConfig(
    min_overlap=1.5,
    max_lateral_distance=1.25,
    max_tangent_diff_deg=15.0,
    smoothing_alpha=0.30,
    min_confirmed_hits=2,
    emit_unconfirmed=False,
    emit_predicted=False,
    max_missed_frames=1,
    missing_confidence_decay=0.50,
    sample_count=30,
    min_depth=1.0,
    min_points_after_transform=6,
)


@dataclass
class FramePacket:
    frame_id: int
    source_index: int
    timestamp_s: float
    image_bgr: np.ndarray
    image_path: Optional[Path]
    cam2img: np.ndarray
    cam2ego: np.ndarray
    cam_to_global: np.ndarray
    sample: Optional[dict] = None
    source_name: str = ""


# =============================================================================
# SHARED CALIBRATION / PATH HELPERS
# =============================================================================
def _normalise_relative_path(path_value):
    return PurePosixPath(str(path_value).replace("\\", "/"))


def _resolve_scene_image(scene_root, sample, cam_type):
    image_entry = sample["images"][cam_type]
    stored_path = _normalise_relative_path(image_entry["img_path"])

    candidate = scene_root.joinpath(*stored_path.parts)
    if candidate.is_file():
        return candidate

    lower_parts = [part.lower() for part in stored_path.parts]
    if "samples" in lower_parts:
        samples_idx = lower_parts.index("samples")
        candidate = scene_root.joinpath(*stored_path.parts[samples_idx:])
        if candidate.is_file():
            return candidate

    candidate = scene_root / "samples" / cam_type / stored_path.name
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "Could not resolve scene image.\n"
        f"Stored path: {image_entry['img_path']}\n"
        f"Scene root:  {scene_root}"
    )


def _camera_to_global(sample, cam_type):
    ego2global = np.asarray(sample["ego2global"], dtype=np.float64)
    cam2ego = np.asarray(sample["images"][cam_type]["cam2ego"], dtype=np.float64)
    if ego2global.shape != (4, 4) or cam2ego.shape != (4, 4):
        raise ValueError("ego2global and cam2ego must both be 4x4")
    return ego2global @ cam2ego


def _timestamp_to_seconds(raw_timestamp):
    """Convert common nuScenes / epoch timestamp scales to seconds."""
    value = float(raw_timestamp)
    magnitude = abs(value)
    if magnitude >= 1e14:      # microseconds since epoch (nuScenes)
        return value / 1e6
    if magnitude >= 1e11:      # milliseconds since epoch
        return value / 1e3
    return value


def load_live_calibration(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. "
            "Create it with make_live_calibration.py."
        )

    calibration = np.load(path)
    if "cam2img" not in calibration:
        raise ValueError("Calibration .npz must contain cam2img")

    cam2img = np.asarray(calibration["cam2img"], dtype=np.float64)
    cam2ego = (
        np.asarray(calibration["cam2ego"], dtype=np.float64)
        if "cam2ego" in calibration
        else np.eye(4, dtype=np.float64)
    )

    if cam2img.shape not in ((3, 3), (3, 4), (4, 4)):
        raise ValueError("cam2img must be 3x3, 3x4, or 4x4")
    if cam2ego.shape != (4, 4):
        raise ValueError("cam2ego must be 4x4")

    return cam2img, cam2ego


# =============================================================================
# SCENE REPLAY SOURCE
# =============================================================================
class SceneReplaySource:
    """Replay a stored nuScenes scene through the same interface as live input."""

    def __init__(self, scene_root, info_file, cam_type):
        self.scene_root = Path(scene_root)
        self.info_file = Path(info_file)
        self.cam_type = cam_type

        if REPLAY_MODE not in {"realtime", "fixed_fps", "max_speed"}:
            raise ValueError(
                "REPLAY_MODE must be 'realtime', 'fixed_fps', or 'max_speed'"
            )
        if REPLAY_SPEED <= 0:
            raise ValueError("REPLAY_SPEED must be > 0")
        if REPLAY_FPS <= 0:
            raise ValueError("REPLAY_FPS must be > 0")

        if not self.scene_root.is_dir():
            raise FileNotFoundError(f"Scene folder not found: {self.scene_root}")
        if not self.info_file.is_file():
            raise FileNotFoundError(f"Scene info file not found: {self.info_file}")

        self.scene_info = mmengine.load(self.info_file)
        data_list = self.scene_info.get("data_list", [])
        if not data_list:
            raise ValueError("Scene info contains no data_list entries")

        end = len(data_list) - 1 if REPLAY_END_INDEX is None else REPLAY_END_INDEX
        if REPLAY_START_INDEX < 0 or end >= len(data_list) or end < REPLAY_START_INDEX:
            raise IndexError(
                f"Replay range {REPLAY_START_INDEX}..{end} is invalid for "
                f"{len(data_list)} samples"
            )

        self.entries = [
            (idx, data_list[idx])
            for idx in range(REPLAY_START_INDEX, end + 1)
            if cam_type in data_list[idx].get("images", {})
        ]
        if not self.entries:
            raise RuntimeError(f"No {cam_type} frames in selected replay range")

        self.position = 0
        self.frame_id = 0
        self.stopped = False
        self._wall_anchor = None
        self._timestamp_anchor = None
        self._fixed_next_wall = None

    @property
    def metadata_template(self):
        return {
            key: value
            for key, value in self.scene_info.items()
            if key != "data_list"
        }

    def _pace(self, timestamp_s):
        if REPLAY_MODE == "max_speed":
            return

        now = time.perf_counter()
        if REPLAY_MODE == "fixed_fps":
            period = 1.0 / REPLAY_FPS
            if self._fixed_next_wall is None:
                self._fixed_next_wall = now
            else:
                self._fixed_next_wall += period
                delay = self._fixed_next_wall - now
                if delay > 0:
                    time.sleep(delay)
            return

        # realtime: use original scene timestamp deltas. If inference is slower
        # than recorded timing, delay becomes <= 0 and replay naturally runs as
        # fast as the pipeline can manage without accumulating a backlog.
        if self._wall_anchor is None:
            self._wall_anchor = now
            self._timestamp_anchor = timestamp_s
            return

        target = self._wall_anchor + (
            (timestamp_s - self._timestamp_anchor) / REPLAY_SPEED
        )
        delay = target - now
        if delay > 0:
            time.sleep(delay)

    def read(self):
        if self.stopped:
            return None

        if self.position >= len(self.entries):
            if not LOOP_SCENE:
                self.stopped = True
                return None
            self.position = 0
            self._wall_anchor = None
            self._timestamp_anchor = None
            self._fixed_next_wall = None

        source_index, sample = self.entries[self.position]
        self.position += 1

        image_path = _resolve_scene_image(self.scene_root, sample, self.cam_type)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read replay image: {image_path}")

        raw_timestamp = sample.get("timestamp", self.frame_id)
        timestamp_s = _timestamp_to_seconds(raw_timestamp)
        self._pace(timestamp_s)

        cam_info = sample["images"][self.cam_type]
        packet = FramePacket(
            frame_id=self.frame_id,
            source_index=int(source_index),
            timestamp_s=timestamp_s,
            image_bgr=image_bgr,
            image_path=image_path,
            cam2img=np.asarray(cam_info["cam2img"], dtype=np.float64),
            cam2ego=np.asarray(cam_info["cam2ego"], dtype=np.float64),
            cam_to_global=_camera_to_global(sample, self.cam_type),
            sample=sample,
            source_name=f"scene[{source_index}]",
        )
        self.frame_id += 1
        return packet

    def release(self):
        self.stopped = True


# =============================================================================
# CAMERA / RTSP SOURCE
# =============================================================================
class LatestFrameGrabber:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_sequence = -1
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        sequence = 0
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.stopped = True
                break
            with self.lock:
                self.latest_frame = frame
                self.latest_sequence = sequence
            sequence += 1

    def read_latest(self, last_sequence):
        with self.lock:
            if self.latest_frame is None or self.latest_sequence == last_sequence:
                return None, last_sequence
            return self.latest_frame.copy(), self.latest_sequence

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class SequentialFrameReader:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.sequence = -1
        self.stopped = False

    def read_latest(self, _last_sequence):
        ok, frame = self.cap.read()
        if not ok:
            self.stopped = True
            return None, self.sequence
        self.sequence += 1
        return frame, self.sequence

    def release(self):
        self.stopped = True
        self.cap.release()


class CameraSource:
    def __init__(self, source):
        self.cam2img, self.cam2ego = load_live_calibration(CALIBRATION_FILE)
        self.reader = (
            LatestFrameGrabber(source).start()
            if DROP_STALE_FRAMES
            else SequentialFrameReader(source)
        )
        self.last_sequence = -1
        self.frame_id = 0
        self.stopped = False

    @property
    def metadata_template(self):
        return {}

    def read(self):
        while not self.reader.stopped:
            frame, sequence = self.reader.read_latest(self.last_sequence)
            if frame is None:
                time.sleep(0.002)
                continue
            self.last_sequence = sequence

            if sequence % PROCESS_EVERY_NTH_FRAME != 0:
                continue

            timestamp_s = time.time()
            packet = FramePacket(
                frame_id=self.frame_id,
                source_index=int(sequence),
                timestamp_s=timestamp_s,
                image_bgr=frame,
                image_path=None,
                cam2img=self.cam2img.copy(),
                cam2ego=self.cam2ego.copy(),
                cam_to_global=get_camera_cam_to_global(
                    self.frame_id, timestamp_s
                ),
                sample=None,
                source_name=f"camera[{sequence}]",
            )
            self.frame_id += 1
            return packet

        self.stopped = True
        return None

    def release(self):
        self.stopped = True
        self.reader.release()


# =============================================================================
# MMDETECTION3D ADAPTER
# =============================================================================
def write_inference_annotation(
    destination,
    packet,
    actual_image_path,
    metadata_template,
):
    """Build the one-frame metadata record needed by FCOS3D."""
    if packet.sample is not None:
        sample = copy.deepcopy(packet.sample)
        sample["images"][CAM_TYPE]["img_path"] = str(actual_image_path)
    else:
        ego2global = packet.cam_to_global @ np.linalg.inv(packet.cam2ego)
        sample = {
            "sample_idx": int(packet.frame_id),
            "timestamp": float(packet.timestamp_s),
            "ego2global": ego2global,
            "images": {
                CAM_TYPE: {
                    "img_path": str(actual_image_path),
                    "cam2img": packet.cam2img,
                    "cam2ego": packet.cam2ego,
                }
            },
        }

    info = dict(metadata_template)
    info["data_list"] = [sample]
    mmengine.dump(info, destination)


# =============================================================================
# LANE INFERENCE
# =============================================================================
def build_lane_result(history_records):
    temporal_vehicles, tracks = accumulate_temporal_vehicle_evidence(
        list(history_records),
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
    initial_fits = fit_lane_streams(graph, initial_streams, cfg=FIT_CONFIG)
    streams, lane_fits, merge_events = merge_compatible_lane_streams(
        graph,
        initial_streams,
        lane_fits=initial_fits,
        merge_cfg=MERGE_CONFIG,
        fit_cfg=FIT_CONFIG,
    )
    boundaries = infer_lane_boundaries(lane_fits, cfg=BOUNDARY_CONFIG)

    return (
        temporal_vehicles,
        tracks,
        graph,
        streams,
        lane_fits,
        boundaries,
        merge_events,
    )


def draw_projected_boundaries(frame_bgr, projected_boundaries):
    output = frame_bgr.copy()

    for boundary_index, boundary in enumerate(projected_boundaries):
        pixels = np.asarray(boundary["pixels"], dtype=np.float64)
        if len(pixels) < 2:
            continue

        points = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
        is_predicted = bool(boundary.get("is_predicted", False))
        color = (0, 165, 255) if is_predicted else (255, 180, 0)
        cv2.polylines(
            output,
            [points],
            isClosed=False,
            color=color,
            thickness=0 if is_predicted else 3,
            lineType=cv2.LINE_AA,
        )

        u, v = points[len(points) // 2, 0]
        track_id = boundary.get("boundary_track_id")
        status = boundary.get("temporal_status", "measurement")
        label = (
            f"lane B{track_id} {status}"
            if track_id is not None
            else f"lane {boundary_index}"
        )
        cv2.putText(
            output,
            label,
            (int(u) + 5, int(v) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_status(
    frame,
    packet,
    inference_ms,
    pipeline_fps,
    vehicle_count,
    boundary_count,
    history_size,
):
    lines = [
        f"input: {INPUT_MODE} | {packet.source_name}",
        f"inference: {inference_ms:.1f} ms",
        f"pipeline FPS: {pipeline_fps:.1f}",
        f"vehicles: {vehicle_count}",
        f"boundaries: {boundary_count}",
        f"history: {history_size}",
    ]
    if INPUT_MODE == "scene_replay":
        lines.append(f"replay: {REPLAY_MODE} | speed={REPLAY_SPEED:g}x")
    else:
        lines.append(f"pose mode: {POSE_MODE}")
    lines.append("q / ESC: quit")

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 25


def create_input_source():
    if INPUT_MODE == "scene_replay":
        print(f"Replay scene: {SCENE_ROOT}")
        print(f"Replay info:  {SCENE_INFO_FILE}")
        print(f"Replay mode:  {REPLAY_MODE}")
        return SceneReplaySource(SCENE_ROOT, SCENE_INFO_FILE, CAM_TYPE)

    if INPUT_MODE == "camera":
        print(f"Camera/video source: {VIDEO_SOURCE}")
        return CameraSource(VIDEO_SOURCE)

    raise ValueError("INPUT_MODE must be 'scene_replay' or 'camera'")


def effective_history_length():
    # Scene replay has recorded camera poses, so full temporal accumulation is
    # valid. A moving physical camera without pose data must not mix frames.
    if INPUT_MODE == "camera" and POSE_MODE == "none":
        if TEMPORAL_CONFIG.history_frames > 1:
            print(
                "WARNING: camera POSE_MODE='none'; using one-frame history "
                "to avoid misaligned temporal evidence."
            )
        return 1
    return TEMPORAL_CONFIG.history_frames


def boundary_tracking_enabled():
    # A moving camera needs a pose for boundary geometry to be transformed
    # into the next camera frame. Scene replay provides recorded poses, while
    # live camera mode requires either a static mount or a custom pose source.
    return not (INPUT_MODE == "camera" and POSE_MODE == "none")


# =============================================================================
# MAIN STREAMING LOOP
# =============================================================================
def main():
    source = create_input_source()
    history = deque(maxlen=effective_history_length())
    boundary_tracker_state = None
    previous_source_index = None

    print("Loading FCOS3D...")
    model = init_model(str(CONFIG), str(CHECKPOINT), device=DEVICE)
    class_names = model.dataset_meta["classes"]
    vehicle_label_ids = {
        class_id
        for class_id, name in enumerate(class_names)
        if name in WANTED_VEHICLE_CLASSES
    }
    print("FCOS3D loaded.")
    print(
        "Vehicle classes: "
        + ", ".join(class_names[i] for i in sorted(vehicle_label_ids))
    )

    video_writer = None
    previous_pipeline_time = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="live_fcos3d_") as temp_dir:
        temp_dir = Path(temp_dir)
        camera_image_path = temp_dir / "camera_frame.jpg"
        annotation_path = temp_dir / "frame_info.pkl"

        try:
            while not source.stopped:
                packet = source.read()
                if packet is None:
                    break

                # A replay loop jumps from the final scene sample back to the
                # first. Do not carry vehicle or boundary state across that
                # discontinuity.
                if (
                    INPUT_MODE == "scene_replay"
                    and previous_source_index is not None
                    and packet.source_index <= previous_source_index
                ):
                    history.clear()
                    boundary_tracker_state = None
                previous_source_index = packet.source_index

                # Scene replay can use the original JPEG directly. Camera frames
                # are written to one reusable temporary file for MMDetection3D.
                if packet.image_path is not None:
                    inference_image_path = packet.image_path
                else:
                    ok = cv2.imwrite(
                        str(camera_image_path),
                        packet.image_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, TEMP_JPEG_QUALITY],
                    )
                    if not ok:
                        raise RuntimeError("Failed to write temporary camera frame")
                    inference_image_path = camera_image_path

                write_inference_annotation(
                    annotation_path,
                    packet,
                    inference_image_path,
                    source.metadata_template,
                )

                inference_start = time.perf_counter()
                result = inference_mono_3d_detector(
                    model,
                    str(inference_image_path),
                    str(annotation_path),
                    cam_type=CAM_TYPE,
                )
                inference_ms = (time.perf_counter() - inference_start) * 1000.0

                pred = result.pred_instances_3d
                current_vehicles = extract_vehicles_from_prediction(
                    pred,
                    vehicle_label_ids,
                    cfg=GRAPH_CONFIG,
                )
                road_plane = estimate_road_plane(
                    pred,
                    current_vehicles,
                    cfg=ROAD_PLANE_CONFIG,
                )

                history.append(
                    {
                        "frame_index": packet.frame_id,
                        "vehicles": current_vehicles,
                        "cam_to_global": packet.cam_to_global,
                    }
                )

                (
                    temporal_vehicles,
                    tracks,
                    graph,
                    streams,
                    lane_fits,
                    lane_boundaries,
                    merge_events,
                ) = build_lane_result(history)

                boundary_events = []
                if boundary_tracking_enabled():
                    (
                        lane_boundaries,
                        boundary_tracker_state,
                        boundary_events,
                    ) = update_temporal_lane_tracks(
                        boundary_tracker_state,
                        lane_boundaries,
                        packet.cam_to_global,
                        road_plane,
                        packet.frame_id,
                        cfg=BOUNDARY_TRACKING_CONFIG,
                    )
                else:
                    boundary_tracker_state = None

                projected_boundaries = []
                if road_plane is not None and lane_boundaries:
                    projected_boundaries = project_lane_boundaries_to_image(
                        lane_boundaries,
                        road_plane,
                        packet.cam2img,
                        image_shape=packet.image_bgr.shape,
                        cfg=PROJECTION_CONFIG,
                    )

                output = draw_projected_boundaries(
                    packet.image_bgr, projected_boundaries
                )

                now = time.perf_counter()
                dt = max(now - previous_pipeline_time, 1e-6)
                pipeline_fps = 1.0 / dt
                previous_pipeline_time = now

                draw_status(
                    output,
                    packet,
                    inference_ms,
                    pipeline_fps,
                    len(current_vehicles),
                    len(projected_boundaries),
                    len(history),
                )

                if SAVE_OUTPUT_VIDEO:
                    if video_writer is None:
                        height, width = output.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(OUTPUT_VIDEO_PATH),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            max(1.0, pipeline_fps),
                            (width, height),
                        )
                    video_writer.write(output)

                if DISPLAY_WINDOW:
                    cv2.imshow(WINDOW_NAME, output)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                print(
                    f"{packet.source_name}: "
                    f"raw={len(pred.bboxes_3d)} | "
                    f"vehicles={len(current_vehicles)} | "
                    f"temporal_obs={len(temporal_vehicles)} | "
                    f"tracks={len(tracks)} | "
                    f"streams={len(streams)} | "
                    f"merges={len(merge_events)} | "
                    f"boundaries={len(projected_boundaries)} | "
                    f"boundary_events={len(boundary_events)} | "
                    f"FCOS3D={inference_ms:.1f} ms"
                )

        finally:
            source.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
