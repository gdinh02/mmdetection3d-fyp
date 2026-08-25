# infer_fcos3d_lane_graph.py

from pathlib import Path

from mmdet3d.apis import (
    init_model,
    inference_mono_3d_detector,
)

from lane_graph import (
    LaneGraphConfig,
    build_lane_compatibility_graph,
    get_lane_streams,
    print_graph_edges,
    plot_lane_graph,
    plot_front_and_lane_graph,
    load_cam2img
)


# ============================================================
# CONFIG
# ============================================================

CONFIG = (
    "configs/fcos3d/"
    "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
)

CHECKPOINT = "checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

IMAGE = "demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"

# MMDetection3D info file containing camera calibration for IMAGE.
# For a single image, this file should have exactly one entry in data_list.
INFO_FILE = "demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl"

DEVICE = "cuda:0"

# nuScenes FCOS3D normally uses CAM_FRONT
CAM_TYPE = "CAM_FRONT"


# ============================================================
# MAIN
# ============================================================

def main():

    from mmdet3d.apis import (
    init_model,
    inference_mono_3d_detector,
)

    # ==========================================================
    # 1. Model
    # ==========================================================

    model = init_model(
        CONFIG,
        CHECKPOINT,
        device="cuda:0",
    )


    # ==========================================================
    # 2. FCOS3D inference
    # ==========================================================

    result = inference_mono_3d_detector(
        model,
        IMAGE,
        INFO_FILE,
        cam_type=CAM_TYPE,
    )

    pred = result.pred_instances_3d


    # ==========================================================
    # 3. Vehicle classes
    # ==========================================================

    class_names = model.dataset_meta[
        "classes"
    ]

    wanted_classes = {
        "car",
        "truck",
        "bus",
    }

    vehicle_label_ids = {
        i
        for i, cls in enumerate(
            class_names
        )
        if cls in wanted_classes
    }


    # ==========================================================
    # 4. Graph
    # ==========================================================

    cfg = LaneGraphConfig(
        score_thresh=0.30,
        max_depth=50.0,
        max_cross_track=1.5,
        max_yaw_diff_deg=15.0,
        max_along_track=25.0,
    )

    G, vehicles = (
        build_lane_compatibility_graph(
            pred_instances_3d=pred,
            vehicle_label_ids=vehicle_label_ids,
            cfg=cfg,
        )
    )


    # ==========================================================
    # 5. Streams
    # ==========================================================

    # Keep singletons while debugging
    streams = get_lane_streams(
        G,
        min_vehicles=1,
    )


    # ==========================================================
    # 6. Camera intrinsics
    # ==========================================================

    cam2img = load_cam2img(
        INFO_FILE,
        CAM_TYPE,
    )


    # ==========================================================
    # 7. Visualise
    # ==========================================================

    plot_front_and_lane_graph(
        image_path=IMAGE,
        pred_instances_3d=pred,
        G=G,
        vehicles=vehicles,
        cam2img=cam2img,
        streams=streams,
        class_names=class_names,
        max_depth=50,
        x_range=(-12, 12),
        save_path="fcos3d_lane_debug.png",
    )


if __name__ == "__main__":
    main()