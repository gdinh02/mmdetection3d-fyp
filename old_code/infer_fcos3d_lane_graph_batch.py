# infer_fcos3d_keep_scenes.py

from copy import deepcopy
from pathlib import Path

import mmengine
import pandas as pd
import torch

from mmengine.dataset import Compose, pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type

from nuscenes.nuscenes import NuScenes

from lane_graph import (
    LaneGraphConfig,
    build_lane_compatibility_graph,
    get_lane_streams,
    plot_lane_graph,
)


# ============================================================
# CONFIG
# ============================================================

CONFIG = (
    "configs/fcos3d/"
    "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
)

CHECKPOINT = (
    "checkpoints/"
    "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_"
    "nus-mono3d_finetune_20210717_095645-8d806dc2.pth"
)

DEVICE = "cuda:0"

CAM_TYPE = "CAM_FRONT"


# ------------------------------------------------------------
# nuScenes
# ------------------------------------------------------------

# Change this to your nuScenes root.
#
# Expected structure:
#
# NUSCENES_ROOT/
#   samples/
#   sweeps/
#   maps/
#   v1.0-trainval/
#
NUSCENES_ROOT = Path("/mnt/z/nuscenes")

NUSCENES_VERSION = "v1.0-trainval"


# ------------------------------------------------------------
# Scene review CSV
# ------------------------------------------------------------

REVIEWS_CSV = Path("nuscenes_tools/nuscenes_scene_reviews.csv")


# ------------------------------------------------------------
# ONE calibration/info file
#
# This is loaded exactly once.
# We only take CAM_FRONT.cam2img from it.
# ------------------------------------------------------------

INFO_FILE = Path(
    "/mnt/z/nuscenes"
    "nuscenes_infos_val.pkl"
)


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

OUTPUT_ROOT = Path("/outputs/fcos3d_keep_scenes")


# ============================================================
# SHARED FCOS3D INFERENCE
# ============================================================

def prepare_shared_inference(
    model,
    info_file,
    cam_type="CAM_FRONT",
):
    """
    Prepare everything that can be reused across images.

    Critically, mmengine.load(info_file) occurs HERE ONLY ONCE.
    """

    print(f"Loading calibration info ONCE:\n  {info_file}")

    info = mmengine.load(str(info_file))

    data_list = info["data_list"]

    if len(data_list) == 0:
        raise ValueError(
            f"No entries found in info file: {info_file}"
        )

    first_info = data_list[0]

    if cam_type not in first_info["images"]:
        raise KeyError(
            f"{cam_type} not present in info file. "
            f"Available cameras: {list(first_info['images'].keys())}"
        )

    # Only retain the intrinsic matrix.
    #
    # Do NOT retain sample-specific information such as
    # img_path, timestamp, sample token, etc.
    shared_cam2img = deepcopy(
        first_info["images"][cam_type]["cam2img"]
    )

    # We no longer need the loaded pickle contents.
    del info
    del data_list
    del first_info

    # --------------------------------------------------------
    # Build MMDetection3D test pipeline ONCE
    # --------------------------------------------------------

    cfg = model.cfg

    test_pipeline_cfg = deepcopy(
        cfg.test_dataloader.dataset.pipeline
    )

    test_pipeline = Compose(
        test_pipeline_cfg
    )

    box_type_3d, box_mode_3d = get_box_type(
        cfg.test_dataloader.dataset.box_type_3d
    )

    print("Shared inference pipeline prepared.")

    return (
        shared_cam2img,
        test_pipeline,
        box_type_3d,
        box_mode_3d,
    )


def inference_mono_shared_calibration(
    model,
    image_path,
    shared_cam2img,
    test_pipeline,
    box_type_3d,
    box_mode_3d,
    cam_type="CAM_FRONT",
):
    """
    FCOS3D inference for one image without reading any .pkl file.

    This is essentially the relevant part of
    inference_mono_3d_detector(), except that cam2img is already
    in memory.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    # Important:
    #
    # deepcopy cam2img because some image transforms may modify
    # the projection matrix when resizing/cropping.
    camera_info = {
        "img_path": str(image_path),
        "cam2img": deepcopy(shared_cam2img),
    }

    data = {
        "images": {
            cam_type: camera_info,
        },
        "box_type_3d": box_type_3d,
        "box_mode_3d": box_mode_3d,
    }

    # Image loading, resizing, normalization, packing, etc.
    data = test_pipeline(data)

    batch = pseudo_collate([data])

    with torch.inference_mode():
        results = model.test_step(batch)

    return results[0]


# ============================================================
# NUSCENES SCENE ITERATION
# ============================================================

def iter_scene_keyframes(
    nusc,
    scene_token,
    cam_type="CAM_FRONT",
):
    """
    Yield every nuScenes keyframe for the requested camera in a scene.

    Yields
    ------
    frame_index
    sample_token
    sample_data_token
    image_path
    """

    scene = nusc.get(
        "scene",
        scene_token,
    )

    sample_token = scene["first_sample_token"]

    frame_index = 0

    while sample_token:

        sample = nusc.get(
            "sample",
            sample_token,
        )

        sample_data_token = sample["data"][cam_type]

        sample_data = nusc.get(
            "sample_data",
            sample_data_token,
        )

        image_path = (
            Path(nusc.dataroot)
            / sample_data["filename"]
        )

        yield (
            frame_index,
            sample_token,
            sample_data_token,
            image_path,
        )

        sample_token = sample["next"]
        frame_index += 1


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Read scene review CSV
    # --------------------------------------------------------

    print("Reading scene reviews...")

    reviews = pd.read_csv(
        REVIEWS_CSV,
        dtype=str,
    ).fillna("")

    required_columns = {
        "scene_name",
        "scene_token",
        "decision",
    }

    missing = required_columns - set(reviews.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    keep_reviews = reviews[
        reviews["decision"]
        .str.strip()
        .str.casefold()
        .eq("keep")
    ].copy()

    print(
        f"Scenes marked keep: {len(keep_reviews)}"
    )

    if len(keep_reviews) == 0:
        print("Nothing to process.")
        return

    # --------------------------------------------------------
    # 2. Load nuScenes metadata
    # --------------------------------------------------------

    print("\nLoading nuScenes metadata...")

    nusc = NuScenes(
        version=NUSCENES_VERSION,
        dataroot=str(NUSCENES_ROOT),
        verbose=False,
    )

    print("nuScenes metadata loaded.")

    # --------------------------------------------------------
    # 3. Load FCOS3D ONCE
    # --------------------------------------------------------

    print("\nLoading FCOS3D...")

    model = init_model(
        CONFIG,
        CHECKPOINT,
        device=DEVICE,
    )

    print("Model loaded.")

    class_names = model.dataset_meta["classes"]

    print("\nModel classes:")

    for i, name in enumerate(class_names):
        print(
            f"  {i:2d}: {name}"
        )

    # --------------------------------------------------------
    # 4. Classes used by lane graph
    # --------------------------------------------------------

    wanted_vehicle_classes = {
        "car",
        "truck",
        "bus",
    }

    vehicle_label_ids = {
        i
        for i, name in enumerate(class_names)
        if name in wanted_vehicle_classes
    }

    print(
        "\nVehicle class IDs:",
        vehicle_label_ids,
    )

    # --------------------------------------------------------
    # 5. Graph configuration
    # --------------------------------------------------------

    graph_cfg = LaneGraphConfig(
        score_thresh=0.30,
        max_depth=50.0,
        max_cross_track=1.5,
        max_yaw_diff_deg=15.0,
        max_along_track=25.0,
        sigma_cross_track=0.8,
        sigma_yaw_deg=8.0,
    )

    # --------------------------------------------------------
    # 6. Load .pkl exactly ONCE
    #
    # Everything returned here is reused for every scene/frame.
    # --------------------------------------------------------

    (
        shared_cam2img,
        test_pipeline,
        box_type_3d,
        box_mode_3d,
    ) = prepare_shared_inference(
        model=model,
        info_file=INFO_FILE,
        cam_type=CAM_TYPE,
    )

    print("\nShared CAM_FRONT intrinsic matrix:")

    for row in shared_cam2img:
        print(" ", row)

    # --------------------------------------------------------
    # 7. Output setup
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    total_frames = 0

    # ========================================================
    # 8. Process every KEEP scene
    # ========================================================

    for keep_idx, (_, review) in enumerate(
        keep_reviews.iterrows(),
        start=1,
    ):

        scene_name = review["scene_name"]
        scene_token = review["scene_token"]

        print("\n")
        print("=" * 70)
        print(
            f"SCENE {keep_idx}/{len(keep_reviews)}: "
            f"{scene_name}"
        )
        print(f"token: {scene_token}")
        print("=" * 70)

        scene_output = (
            OUTPUT_ROOT
            / scene_name
        )

        bev_output = (
            scene_output
            / "bev"
        )

        bev_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Iterate every CAM_FRONT keyframe in this scene
        # ----------------------------------------------------

        for (
            frame_index,
            sample_token,
            sample_data_token,
            image_path,
        ) in iter_scene_keyframes(
            nusc=nusc,
            scene_token=scene_token,
            cam_type=CAM_TYPE,
        ):

            total_frames += 1

            print(
                f"[{scene_name}] "
                f"frame {frame_index:03d} | "
                f"{image_path.name}"
            )

            # ------------------------------------------------
            # FCOS3D inference
            #
            # NO PKL LOAD HERE.
            # ------------------------------------------------

            result = inference_mono_shared_calibration(
                model=model,
                image_path=image_path,
                shared_cam2img=shared_cam2img,
                test_pipeline=test_pipeline,
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d,
                cam_type=CAM_TYPE,
            )

            pred = result.pred_instances_3d

            # ------------------------------------------------
            # Lane graph
            # ------------------------------------------------

            G, vehicles = build_lane_compatibility_graph(
                pred_instances_3d=pred,
                vehicle_label_ids=vehicle_label_ids,
                cfg=graph_cfg,
            )

            streams = get_lane_streams(
                G,
                min_vehicles=2,
            )

            # ------------------------------------------------
            # Save BEV
            # ------------------------------------------------

            bev_path = (
                bev_output
                / f"{frame_index:03d}_{sample_token}.png"
            )

            plot_lane_graph(
                G,
                vehicles,
                streams=streams,
                max_depth=50,
                x_range=(-12, 12),
                save_path=str(bev_path),
                show=False,
            )

            # ------------------------------------------------
            # Summary record
            # ------------------------------------------------

            summary_rows.append({
                "scene_name": scene_name,
                "scene_token": scene_token,
                "frame_index": frame_index,
                "sample_token": sample_token,
                "sample_data_token": sample_data_token,
                "image_path": str(image_path),
                "raw_detections": len(pred.bboxes_3d),
                "usable_vehicles": len(vehicles),
                "graph_nodes": G.number_of_nodes(),
                "graph_edges": G.number_of_edges(),
                "lane_streams": len(streams),
                "bev_path": str(bev_path),
            })

            print(
                f"    detections={len(pred.bboxes_3d):3d} | "
                f"vehicles={len(vehicles):2d} | "
                f"edges={G.number_of_edges():2d} | "
                f"streams={len(streams):2d}"
            )

    # ========================================================
    # 9. Save batch summary
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_ROOT
        / "inference_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"Keep scenes processed: {len(keep_reviews)}"
    )

    print(
        f"Frames processed:      {total_frames}"
    )

    print(
        f"Summary:               {summary_path}"
    )

    print(
        f"Outputs:               {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()