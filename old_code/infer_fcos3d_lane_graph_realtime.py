#!/usr/bin/env python3

import argparse
import csv
import json
import queue
import threading
import time

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis import init_model

'''
python infer_fcos3d_lane_graph_realtime.py \
    --config configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py \
    --checkpoint checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth \
    --nuscenes-root /mnt/z/nuscenes \
    --scene-csv nuscenes_tools/nuscenes_scene_reviews.csv \
    --scene scene-0011 \
    --camera CAM_FRONT
'''


# ============================================================
# JSON utilities
# ============================================================

def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_large_json_array(path):
    """
    Stream a large JSON array if ijson is installed.

    Falls back to json.load() otherwise.

    Installing ijson is recommended because sample_data.json
    can be large:

        pip install ijson
    """
    path = Path(path)

    try:
        import ijson

        with path.open("rb") as f:
            for item in ijson.items(f, "item"):
                yield item

    except ImportError:
        print(
            "[WARNING] ijson is not installed. "
            "Loading the entire JSON file into RAM."
        )

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            yield item


# ============================================================
# Kept-scene CSV handling
# ============================================================

def load_kept_scenes(csv_path):
    """
    Expected CSV layout from your file:

        col 0  scene name
        col 1  scene token
        col 2  keep / skip
        col 3  unused
        col 4  unused
        col 5  CAM_FRONT sample_data token
        col 6  CAM_FRONT image path
        col 7  location
        col 8  description
        col 9  review timestamp

    Only rows marked 'keep' are returned.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    kept = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(f, dialect)

        for csv_row_number, row in enumerate(reader):

            if not row:
                continue

            row = [x.strip() for x in row]

            if len(row) < 3:
                continue

            scene_name = row[0]
            decision = row[2].lower()

            # Ignore header rows or malformed rows.
            if not scene_name.startswith("scene-"):
                continue

            if decision != "keep":
                continue

            scene = {
                "kept_index": len(kept),
                "csv_row": csv_row_number,
                "scene_name": scene_name,
                "scene_token":
                    row[1] if len(row) > 1 else "",
                "decision": decision,
                "front_sample_data_token":
                    row[5] if len(row) > 5 else "",
                "front_image":
                    row[6] if len(row) > 6 else "",
                "location":
                    row[7] if len(row) > 7 else "",
                "description":
                    row[8] if len(row) > 8 else "",
                "review_time":
                    row[9] if len(row) > 9 else "",
            }

            kept.append(scene)

    return kept


def print_kept_scenes(csv_path):
    scenes = load_kept_scenes(csv_path)

    if not scenes:
        print("No scenes marked 'keep'.")
        return

    print()
    print(
        f"{'IDX':<5}"
        f"{'SCENE':<14}"
        f"{'LOCATION':<24}"
        f"DESCRIPTION"
    )

    print("-" * 110)

    for scene in scenes:

        description = scene["description"]

        if len(description) > 65:
            description = description[:62] + "..."

        print(
            f"{scene['kept_index']:<5}"
            f"{scene['scene_name']:<14}"
            f"{scene['location']:<24}"
            f"{description}"
        )

    print()
    print(f"Total kept scenes: {len(scenes)}")
    print()


def select_kept_scene(csv_path, selector):
    """
    selector can be:

        0
        1
        2

    meaning kept-scene index,

    OR:

        scene-0011

    OR:

        c075fbdd97124beaba95bc5c25149f30
    """

    kept = load_kept_scenes(csv_path)

    if not kept:
        raise RuntimeError(
            f"No scenes marked 'keep' in {csv_path}"
        )

    selector = str(selector).strip()

    # --------------------------------------------
    # Kept-scene index
    # --------------------------------------------

    if selector.isdigit():

        idx = int(selector)

        if idx < 0 or idx >= len(kept):
            raise IndexError(
                f"Kept index {idx} is invalid. "
                f"There are {len(kept)} kept scenes."
            )

        return kept[idx]

    # --------------------------------------------
    # Scene name
    # --------------------------------------------

    for scene in kept:
        if scene["scene_name"] == selector:
            return scene

    # --------------------------------------------
    # Scene token
    # --------------------------------------------

    for scene in kept:
        if scene["scene_token"] == selector:
            return scene

    raise ValueError(
        f"'{selector}' was not found among the kept scenes."
    )


# ============================================================
# nuScenes scene lookup
# ============================================================

def get_scene_record(
    nuscenes_root,
    metadata_version,
    selected_scene
):
    root = Path(nuscenes_root)

    scene_path = (
        root
        / metadata_version
        / "scene.json"
    )

    scenes = load_json(scene_path)

    wanted_token = selected_scene["scene_token"]
    wanted_name = selected_scene["scene_name"]

    # Prefer exact token.
    for scene in scenes:
        if scene["token"] == wanted_token:
            return scene

    # Fallback to name.
    for scene in scenes:
        if scene["name"] == wanted_name:
            return scene

    raise ValueError(
        f"{wanted_name} / {wanted_token} "
        "was not found in scene.json"
    )


# ============================================================
# Build one camera stream from one nuScenes scene
# ============================================================

def get_scene_camera_frames(
    nuscenes_root,
    metadata_version,
    scene_record,
    camera="CAM_FRONT",
    keyframes_only=False,
):
    """
    Get all available frames for one camera in one nuScenes scene.

    Reads raw nuScenes JSON directly.
    Does NOT require:
        - NuScenes()
        - LiDAR
        - .pkl files
    """

    root = Path(nuscenes_root)
    meta = root / metadata_version

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------
    samples = load_json(meta / "sample.json")

    sample_by_token = {
        sample["token"]: sample
        for sample in samples
    }

    first_sample_token = scene_record["first_sample_token"]
    last_sample_token = scene_record["last_sample_token"]

    first_sample = sample_by_token[first_sample_token]
    last_sample = sample_by_token[last_sample_token]

    first_ts = int(first_sample["timestamp"])
    last_ts = int(last_sample["timestamp"])

    # --------------------------------------------------------
    # Sensor/calibration information
    # --------------------------------------------------------
    sensors = load_json(meta / "sensor.json")
    calibrated_sensors = load_json(
        meta / "calibrated_sensor.json"
    )

    sensor_by_token = {
        x["token"]: x
        for x in sensors
    }

    calib_by_token = {
        x["token"]: x
        for x in calibrated_sensors
    }

    # Find calibration token(s) belonging to requested camera.
    camera_calibration_tokens = set()

    for calib in calibrated_sensors:
        sensor = sensor_by_token[calib["sensor_token"]]

        if sensor["channel"] == camera:
            camera_calibration_tokens.add(calib["token"])

    if not camera_calibration_tokens:
        raise RuntimeError(
            f"No calibration found for {camera}"
        )

    # --------------------------------------------------------
    # Read relevant sample_data records
    # --------------------------------------------------------
    sd_by_token = {}

    first_sd_token = None
    last_sd_token = None

    # A little margin around scene timestamps.
    margin_us = 2_000_000

    for sd in iter_large_json_array(
        meta / "sample_data.json"
    ):
        # Only requested camera.
        if (
            sd["calibrated_sensor_token"]
            not in camera_calibration_tokens
        ):
            continue

        timestamp = int(sd["timestamp"])

        # Ignore camera data far outside this scene.
        if timestamp < first_ts - margin_us:
            continue

        if timestamp > last_ts + margin_us:
            continue

        sd_by_token[sd["token"]] = sd

        # Raw sample.json has no `data` dictionary,
        # so derive these associations from sample_data.json.
        if (
            sd["sample_token"] == first_sample_token
            and sd["is_key_frame"]
        ):
            first_sd_token = sd["token"]

        if (
            sd["sample_token"] == last_sample_token
            and sd["is_key_frame"]
        ):
            last_sd_token = sd["token"]

    if first_sd_token is None:
        raise RuntimeError(
            f"Could not find first {camera} frame for "
            f"{scene_record['name']}"
        )

    if last_sd_token is None:
        raise RuntimeError(
            f"Could not find last {camera} frame for "
            f"{scene_record['name']}"
        )

    # --------------------------------------------------------
    # Follow camera sample_data chain
    # --------------------------------------------------------
    frames = []
    missing_files = 0

    token = first_sd_token
    visited = set()

    while token:
        if token in visited:
            raise RuntimeError(
                "Loop detected in sample_data chain."
            )

        visited.add(token)

        if token not in sd_by_token:
            raise RuntimeError(
                f"sample_data token {token} "
                f"not available for {camera}"
            )

        sd = sd_by_token[token]

        calib = calib_by_token[
            sd["calibrated_sensor_token"]
        ]

        K = np.asarray(
            calib["camera_intrinsic"],
            dtype=np.float32
        )

        image_path = root / sd["filename"]

        use_frame = (
            not keyframes_only
            or sd["is_key_frame"]
        )

        if use_frame:
            if image_path.exists():
                frames.append({
                    "timestamp": int(sd["timestamp"]),
                    "path": image_path,
                    "K": K,
                    "sample_data_token": sd["token"],
                    "sample_token": sd["sample_token"],
                    "is_key_frame": bool(
                        sd["is_key_frame"]
                    ),
                })
            else:
                missing_files += 1

        # We've reached the final keyframe of this scene.
        if token == last_sd_token:
            break

        token = sd.get("next", "")

    frames.sort(
        key=lambda x: x["timestamp"]
    )

    print(
        f"{scene_record['name']} | {camera}: "
        f"{len(frames)} available frames"
    )

    if missing_files:
        print(
            f"Missing camera files skipped: "
            f"{missing_files}"
        )

    return frames

# ============================================================
# FCOS3D model wrapper
# ============================================================

class FCOS3DRealtime:

    def __init__(
        self,
        config,
        checkpoint,
        device="cuda:0",
        score_thr=0.3,
        amp=False,
    ):
        self.device = device
        self.score_thr = score_thr

        self.amp = (
            amp
            and device.startswith("cuda")
            and torch.cuda.is_available()
        )

        print()
        print("Loading FCOS3D...")
        print(f"Config:     {config}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Device:     {device}")

        self.model = init_model(
            config,
            checkpoint,
            device=device,
        )

        self.model.eval()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # ----------------------------------------------------
        # FCOS3D's normal test pipeline begins with
        # LoadImageFromFileMono3D.
        #
        # MMDetection3D's MonoDet3DInferencerLoader supports
        # BOTH:
        #
        #   img = filename
        #
        # and
        #
        #   img = np.ndarray
        #
        # together with cam2img.
        # ----------------------------------------------------

        pipeline_cfg = deepcopy(
            self.model.cfg
            .test_dataloader
            .dataset
            .pipeline
        )

        found_loader = False

        for transform in pipeline_cfg:

            transform_type = transform.get(
                "type",
                ""
            )

            if (
                transform_type
                == "LoadImageFromFileMono3D"
            ):
                transform["type"] = (
                    "MonoDet3DInferencerLoader"
                )

                found_loader = True
                break

        if not found_loader:
            raise RuntimeError(
                "Could not find "
                "LoadImageFromFileMono3D "
                "in the test pipeline."
            )

        self.pipeline = Compose(
            pipeline_cfg
        )

        self.classes = (
            self.model.dataset_meta
            .get("classes", [])
        )

        print(
            f"Classes:    {len(self.classes)}"
        )

        print(
            f"AMP:        {self.amp}"
        )

        print("FCOS3D ready.")
        print()

    def infer(self, frame, K):
        """
        frame:
            OpenCV BGR image as ndarray.

        K:
            3x3 camera intrinsic matrix.
        """

        data = {
            "img": frame,
            "cam2img": np.asarray(
                K,
                dtype=np.float32
            ),
        }

        processed = self.pipeline(data)

        batch = pseudo_collate(
            [processed]
        )

        with torch.inference_mode():

            if self.amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    result = (
                        self.model
                        .test_step(batch)[0]
                    )

            else:

                result = (
                    self.model
                    .test_step(batch)[0]
                )

        return result


# ============================================================
# 3D box visualization
# ============================================================

# This vertex ordering follows MMDetection3D's projected
# 3D box visualizer.
BOX_LINE_SEQUENCE = [
    0, 1, 2, 3,
    7, 6, 5, 4,
    0, 3, 7, 4,
    5, 1, 2, 6
]


def project_camera_points(points_3d, K):
    """
    Project Nx3 camera-coordinate points to pixels.
    """

    points_3d = np.asarray(
        points_3d,
        dtype=np.float32
    )

    K = np.asarray(
        K,
        dtype=np.float32
    )

    depths = points_3d[:, 2]

    valid = depths > 0.1

    pixels = np.full(
        (len(points_3d), 2),
        np.nan,
        dtype=np.float32
    )

    if valid.any():

        projected = (
            points_3d[valid]
            @ K.T
        )

        pixels[valid] = (
            projected[:, :2]
            / projected[:, 2:3]
        )

    return pixels, valid


def draw_predictions(
    frame,
    result,
    K,
    classes,
    score_thr
):
    """
    Draw FCOS3D camera-coordinate 3D boxes onto image.
    """

    output = frame.copy()

    if not hasattr(
        result,
        "pred_instances_3d"
    ):
        return output, 0

    pred = result.pred_instances_3d

    if (
        not hasattr(pred, "scores_3d")
        or pred.scores_3d.numel() == 0
    ):
        return output, 0

    keep = (
        pred.scores_3d
        >= score_thr
    )

    if keep.sum().item() == 0:
        return output, 0

    boxes = pred.bboxes_3d[keep]

    scores = (
        pred.scores_3d[keep]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        pred.labels_3d[keep]
        .detach()
        .cpu()
        .numpy()
    )

    corners = (
        boxes.corners
        .detach()
        .cpu()
        .numpy()
    )

    box_tensor = (
        boxes.tensor
        .detach()
        .cpu()
        .numpy()
    )

    height, width = output.shape[:2]

    drawn = 0

    for i in range(len(corners)):

        uv, valid = project_camera_points(
            corners[i],
            K
        )

        # Don't attempt to draw boxes mostly behind camera.
        if valid.sum() < 4:
            continue

        points = []

        for vertex_index in BOX_LINE_SEQUENCE:

            if not valid[vertex_index]:
                points.append(None)
                continue

            x, y = uv[vertex_index]

            points.append(
                (
                    int(round(x)),
                    int(round(y))
                )
            )

        for j in range(
            len(BOX_LINE_SEQUENCE) - 1
        ):

            p1 = points[j]
            p2 = points[j + 1]

            if p1 is None or p2 is None:
                continue

            cv2.line(
                output,
                p1,
                p2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        valid_uv = uv[valid]

        x = int(
            np.nanmin(
                valid_uv[:, 0]
            )
        )

        y = int(
            np.nanmin(
                valid_uv[:, 1]
            )
        )

        x = max(
            0,
            min(width - 1, x)
        )

        y = max(
            20,
            min(height - 1, y)
        )

        label = int(labels[i])

        if (
            0 <= label
            < len(classes)
        ):
            class_name = classes[label]
        else:
            class_name = str(label)

        # Camera-coordinate Z is depth.
        depth = float(
            box_tensor[i, 2]
        )

        text = (
            f"{class_name} "
            f"{scores[i]:.2f} "
            f"{depth:.1f}m"
        )

        cv2.putText(
            output,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        drawn += 1

    return output, drawn


# ============================================================
# Latest-frame queue
# ============================================================

def put_latest(q, item, stats):
    """
    Keep queue length at 1.

    If inference is slower than the source, discard the old
    unprocessed frame and replace it with the latest frame.
    """

    try:
        q.put_nowait(item)

    except queue.Full:

        try:
            q.get_nowait()
            stats["dropped"] += 1
        except queue.Empty:
            pass

        q.put_nowait(item)


# ============================================================
# nuScenes producer
# ============================================================

def nuscenes_scene_producer(
    frames,
    output_queue,
    stop_event,
    speed,
    stats
):
    if not frames:
        return

    first_timestamp = (
        frames[0]["timestamp"]
    )

    wall_start = time.perf_counter()

    for index, info in enumerate(frames):

        if stop_event.is_set():
            break

        relative_seconds = (
            info["timestamp"]
            - first_timestamp
        ) / 1_000_000.0

        target_wall_time = (
            wall_start
            + relative_seconds / speed
        )

        while not stop_event.is_set():

            remaining = (
                target_wall_time
                - time.perf_counter()
            )

            if remaining <= 0:
                break

            time.sleep(
                min(remaining, 0.005)
            )

        if stop_event.is_set():
            break

        frame = cv2.imread(
            str(info["path"])
        )

        if frame is None:
            continue

        item = {
            "frame": frame,
            "K": info["K"],
            "frame_index": index,
            "timestamp": info["timestamp"],
            "path": str(info["path"]),
            "is_key_frame":
                info["is_key_frame"],
        }

        put_latest(
            output_queue,
            item,
            stats
        )

    # Wait for consumer to consume final queued image.
    while (
        not stop_event.is_set()
        and not output_queue.empty()
    ):
        time.sleep(0.01)

    if not stop_event.is_set():
        output_queue.put(None)


# ============================================================
# Webcam / video producer
# ============================================================

def video_producer(
    source,
    K,
    output_queue,
    stop_event,
    stats
):
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    # Helps reduce camera latency where supported.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    frame_index = 0

    while not stop_event.is_set():

        ok, frame = cap.read()

        if not ok:
            break

        item = {
            "frame": frame,
            "K": K,
            "frame_index": frame_index,
            "timestamp": None,
            "path": str(source),
            "is_key_frame": True,
        }

        put_latest(
            output_queue,
            item,
            stats
        )

        frame_index += 1

    cap.release()

    while (
        not stop_event.is_set()
        and not output_queue.empty()
    ):
        time.sleep(0.01)

    if not stop_event.is_set():
        output_queue.put(None)


# ============================================================
# Inference consumer
# ============================================================

def inference_consumer(
    detector,
    input_queue,
    stop_event,
    stats,
    window_title,
):
    processed = 0

    while not stop_event.is_set():

        try:
            item = input_queue.get(
                timeout=0.25
            )

        except queue.Empty:
            continue

        if item is None:
            break

        frame = item["frame"]
        K = item["K"]

        inference_start = (
            time.perf_counter()
        )

        result = detector.infer(
            frame,
            K
        )

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        inference_fps = (
            1.0 / inference_seconds
            if inference_seconds > 0
            else 0.0
        )

        vis, num_objects = (
            draw_predictions(
                frame,
                result,
                K,
                detector.classes,
                detector.score_thr,
            )
        )

        frame_index = (
            item["frame_index"]
        )

        # ----------------------------------------------------
        # Status overlay
        # ----------------------------------------------------

        cv2.putText(
            vis,
            (
                f"Inference: "
                f"{inference_fps:.1f} FPS"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            (
                f"Frame: {frame_index}  "
                f"Objects: {num_objects}  "
                f"Dropped: {stats['dropped']}"
            ),
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window_title,
            vis
        )

        processed += 1

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("q"),
            27,
        ):
            stop_event.set()
            break

    cv2.destroyAllWindows()

    print()
    print(
        f"Processed frames: {processed}"
    )

    print(
        f"Dropped frames:   "
        f"{stats['dropped']}"
    )


# ============================================================
# Run selected nuScenes scene
# ============================================================

def run_kept_nuscenes_scene(
    detector,
    nuscenes_root,
    metadata_version,
    csv_path,
    selector,
    camera,
    playback_speed,
    keyframes_only,
):
    selected = select_kept_scene(
        csv_path,
        selector
    )

    scene = get_scene_record(
        nuscenes_root,
        metadata_version,
        selected,
    )

    print()
    print("=" * 80)
    print("SELECTED SCENE")
    print("=" * 80)

    print(
        f"Kept index:  "
        f"{selected['kept_index']}"
    )

    print(
        f"Scene:       "
        f"{selected['scene_name']}"
    )

    print(
        f"Token:       "
        f"{selected['scene_token']}"
    )

    print(
        f"Location:    "
        f"{selected['location']}"
    )

    print(
        f"Camera:      {camera}"
    )

    print(
        f"Description: "
        f"{selected['description']}"
    )

    print("=" * 80)
    print()

    frames = get_scene_camera_frames(
        nuscenes_root,
        metadata_version,
        scene,
        camera=camera,
        keyframes_only=keyframes_only,
    )

    if not frames:
        raise RuntimeError(
            "No available camera images "
            "were found for this scene."
        )

    if len(frames) >= 2:

        duration = (
            frames[-1]["timestamp"]
            - frames[0]["timestamp"]
        ) / 1_000_000.0

        rate = (
            (len(frames) - 1)
            / duration
            if duration > 0
            else 0
        )

        print(
            f"Frames:       {len(frames)}"
        )

        print(
            f"Duration:     "
            f"{duration:.2f} s"
        )

        print(
            f"Available FPS:"
            f" {rate:.2f}"
        )

        print(
            f"Playback:     "
            f"{playback_speed:.2f}x"
        )

        print()

    q = queue.Queue(
        maxsize=1
    )

    stop_event = (
        threading.Event()
    )

    stats = {
        "dropped": 0
    }

    producer = threading.Thread(
        target=nuscenes_scene_producer,
        args=(
            frames,
            q,
            stop_event,
            playback_speed,
            stats,
        ),
        daemon=True,
    )

    producer.start()

    inference_consumer(
        detector,
        q,
        stop_event,
        stats,
        window_title=(
            f"FCOS3D - "
            f"{selected['scene_name']} "
            f"- {camera}"
        ),
    )

    stop_event.set()

    producer.join(
        timeout=2
    )


# ============================================================
# Run webcam / video / RTSP
# ============================================================

def run_video_source(
    detector,
    source,
    K,
):
    q = queue.Queue(
        maxsize=1
    )

    stop_event = (
        threading.Event()
    )

    stats = {
        "dropped": 0
    }

    producer = threading.Thread(
        target=video_producer,
        args=(
            source,
            K,
            q,
            stop_event,
            stats,
        ),
        daemon=True,
    )

    producer.start()

    inference_consumer(
        detector,
        q,
        stop_event,
        stats,
        window_title="FCOS3D realtime",
    )

    stop_event.set()

    producer.join(
        timeout=2
    )


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Realtime FCOS3D inference for "
            "nuScenes kept scenes, webcam, "
            "video or RTSP."
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--config",
        help="FCOS3D config file"
    )

    parser.add_argument(
        "--checkpoint",
        help="FCOS3D checkpoint"
    )

    parser.add_argument(
        "--device",
        default="cuda:0"
    )

    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA autocast FP16 "
            "during inference."
        )
    )

    # --------------------------------------------------------
    # nuScenes
    # --------------------------------------------------------

    parser.add_argument(
        "--nuscenes-root",
        default="/mnt/z/nuscenes"
    )

    parser.add_argument(
        "--metadata-version",
        default="v1.0-trainval"
    )

    parser.add_argument(
        "--scene-csv",
        help="Your keep/skip CSV file"
    )

    parser.add_argument(
        "--scene",
        help=(
            "Kept index, scene name, "
            "or scene token."
        )
    )

    parser.add_argument(
        "--camera",
        default="CAM_FRONT",
        choices=[
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
    )

    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help=(
            "1.0 = recorded speed, "
            "0.5 = half speed, "
            "2.0 = double speed."
        )
    )

    parser.add_argument(
        "--keyframes-only",
        action="store_true",
        help=(
            "Use only nuScenes keyframes. "
            "Default is all available "
            "camera frames."
        )
    )

    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help=(
            "Print all CSV scenes marked "
            "'keep' and exit."
        )
    )

    # --------------------------------------------------------
    # Webcam / video
    # --------------------------------------------------------

    parser.add_argument(
        "--source",
        help=(
            "Camera index, video path, "
            "or RTSP URL."
        )
    )

    parser.add_argument(
        "--fx",
        type=float
    )

    parser.add_argument(
        "--fy",
        type=float
    )

    parser.add_argument(
        "--cx",
        type=float
    )

    parser.add_argument(
        "--cy",
        type=float
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    # --------------------------------------------------------
    # Just list CSV scenes.
    # --------------------------------------------------------

    if args.list_scenes:

        if not args.scene_csv:
            parser.error(
                "--list-scenes requires "
                "--scene-csv"
            )

        print_kept_scenes(
            args.scene_csv
        )

        return

    # --------------------------------------------------------
    # Actual inference needs model.
    # --------------------------------------------------------

    if not args.config:
        parser.error(
            "--config is required "
            "for inference"
        )

    if not args.checkpoint:
        parser.error(
            "--checkpoint is required "
            "for inference"
        )

    detector = FCOS3DRealtime(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        score_thr=args.score_thr,
        amp=args.amp,
    )

    # --------------------------------------------------------
    # nuScenes kept-scene mode.
    # --------------------------------------------------------

    if args.scene_csv:

        if args.scene is None:

            print_kept_scenes(
                args.scene_csv
            )

            args.scene = input(
                "Choose kept scene "
                "(index or scene name): "
            ).strip()

        run_kept_nuscenes_scene(
            detector=detector,

            nuscenes_root=(
                args.nuscenes_root
            ),

            metadata_version=(
                args.metadata_version
            ),

            csv_path=(
                args.scene_csv
            ),

            selector=(
                args.scene
            ),

            camera=(
                args.camera
            ),

            playback_speed=(
                args.playback_speed
            ),

            keyframes_only=(
                args.keyframes_only
            ),
        )

        return

    # --------------------------------------------------------
    # Webcam / video mode.
    # --------------------------------------------------------

    if args.source is not None:

        required_intrinsics = [
            args.fx,
            args.fy,
            args.cx,
            args.cy,
        ]

        if any(
            x is None
            for x in required_intrinsics
        ):
            parser.error(
                "Webcam/video mode requires "
                "--fx --fy --cx --cy"
            )

        K = np.array(
            [
                [
                    args.fx,
                    0.0,
                    args.cx
                ],
                [
                    0.0,
                    args.fy,
                    args.cy
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ],
            ],
            dtype=np.float32,
        )

        run_video_source(
            detector,
            args.source,
            K,
        )

        return

    parser.error(
        "Provide either --scene-csv "
        "or --source."
    )


if __name__ == "__main__":
    main()