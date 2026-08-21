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
    plot_lane_graph
)


# ============================================================
# CONFIG
# ============================================================

CONFIG = (
    "configs/fcos3d/"
    "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
)

CHECKPOINT = "checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

IMAGE = "demo/data/nuscenes/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.jpg"

# MMDetection3D info file containing camera calibration for IMAGE.
# For a single image, this file should have exactly one entry in data_list.
INFO_FILE = "demo/data/nuscenes/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.pkl"

DEVICE = "cuda:0"

# nuScenes FCOS3D normally uses CAM_FRONT
CAM_TYPE = "CAM_FRONT"


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load FCOS3D
    # --------------------------------------------------------

    print("Loading FCOS3D...")

    model = init_model(
        CONFIG,
        CHECKPOINT,
        device=DEVICE,
    )

    print("Model loaded.")

    # Print class mapping so we do not hard-code IDs
    class_names = model.dataset_meta["classes"]

    print("\nModel classes:")
    for i, name in enumerate(class_names):
        print(f"  {i:2d}: {name}")

    # --------------------------------------------------------
    # 2. Run monocular 3D inference
    # --------------------------------------------------------

    print(f"\nRunning inference on:\n  {IMAGE}")

    result = inference_mono_3d_detector(
        model,
        IMAGE,
        INFO_FILE,
        cam_type=CAM_TYPE,
    )

    # result is Det3DDataSample
    pred = result.pred_instances_3d

    print("\nRaw detections:", len(pred.bboxes_3d))

    # --------------------------------------------------------
    # 3. Select classes useful for lane inference
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
    # 4. Configure lane compatibility graph
    # --------------------------------------------------------

    graph_cfg = LaneGraphConfig(
        score_thresh=0.30,

        # Ignore extremely distant detections initially
        max_depth=50.0,

        # Vehicles can differ laterally by at most ~1.5 m
        # and still be considered potentially same-lane
        max_cross_track=1.5,

        # Similar heading requirement
        max_yaw_diff_deg=15.0,

        # Only connect reasonably local neighbours.
        # Longer streams form through graph chaining.
        max_along_track=25.0,

        # Soft edge weighting
        sigma_cross_track=0.8,
        sigma_yaw_deg=8.0,
    )

    # --------------------------------------------------------
    # 5. Feed FCOS3D result directly into graph
    # --------------------------------------------------------

    G, vehicles = build_lane_compatibility_graph(
        pred_instances_3d=pred,
        vehicle_label_ids=vehicle_label_ids,
        cfg=graph_cfg,
    )

    print("\n========================================")
    print("LANE GRAPH")
    print("========================================")

    print(f"Usable vehicles: {len(vehicles)}")
    print(f"Graph nodes:     {G.number_of_nodes()}")
    print(f"Graph edges:     {G.number_of_edges()}")

    # --------------------------------------------------------
    # 6. Inspect retained vehicles
    # --------------------------------------------------------

    print("\nVehicles:")

    for node_id, v in enumerate(vehicles):
        print(
            f"{node_id:2d}: "
            f"x={v['x']:7.2f} m | "
            f"z={v['z']:7.2f} m | "
            f"yaw={v['yaw']:7.3f} rad | "
            f"score={v['score']:.3f} | "
            f"class={class_names[v['label']]}"
        )

    # --------------------------------------------------------
    # 7. Inspect pairwise compatibility edges
    # --------------------------------------------------------

    print("\nCompatible vehicle pairs:")

    print_graph_edges(G)

    # --------------------------------------------------------
    # 8. Connected-component lane streams
    # --------------------------------------------------------

    streams = get_lane_streams(
        G,
        min_vehicles=2,
    )

    plot_lane_graph(
        G,
        vehicles,
        streams=streams,
        max_depth=50,
        x_range=(-12, 12),
        save_path="lane_graph_bev.png",
    )

    print("\n========================================")
    print("CANDIDATE LANE STREAMS")
    print("========================================")

    if len(streams) == 0:
        print("No multi-vehicle streams found.")
        return

    for stream_idx, stream in enumerate(streams):

        # sort vehicles near -> far
        stream = sorted(
            stream,
            key=lambda n: G.nodes[n]["z"],
        )

        print(f"\nLane stream {stream_idx}:")

        for node in stream:

            vehicle = G.nodes[node]

            print(
                f"  node={node:2d} "
                f"x={vehicle['x']:7.2f} "
                f"z={vehicle['z']:7.2f} "
                f"yaw={vehicle['yaw']:7.3f} "
                f"score={vehicle['score']:.3f}"
            )


if __name__ == "__main__":
    main()