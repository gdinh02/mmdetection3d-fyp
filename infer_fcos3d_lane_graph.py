from mmdet3d.apis import inference_mono_3d_detector, init_model

from lane_graph import (
    LaneGraphConfig,
    build_lane_compatibility_graph,
    get_lane_streams,
    plot_lane_graph,
    print_graph_edges,
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
IMAGE = (
    "demo/data/nuscenes/"
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.jpg"
)
INFO_FILE = (
    "demo/data/nuscenes/"
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.pkl"
)
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


def main():
    print("Loading FCOS3D...")
    model = init_model(CONFIG, CHECKPOINT, device=DEVICE)
    print("Model loaded.")

    class_names = model.dataset_meta["classes"]
    print("\nModel classes:")
    for class_id, name in enumerate(class_names):
        print(f"  {class_id:2d}: {name}")

    print(f"\nRunning inference on:\n  {IMAGE}")
    result = inference_mono_3d_detector(
        model,
        IMAGE,
        INFO_FILE,
        cam_type=CAM_TYPE,
    )
    pred = result.pred_instances_3d
    print(f"\nRaw detections: {len(pred.bboxes_3d)}")

    vehicle_label_ids = {
        class_id
        for class_id, name in enumerate(class_names)
        if name in WANTED_VEHICLE_CLASSES
    }
    print("\nVehicle class IDs:", vehicle_label_ids)

    graph, vehicles = build_lane_compatibility_graph(
        pred_instances_3d=pred,
        vehicle_label_ids=vehicle_label_ids,
        cfg=GRAPH_CONFIG,
    )

    print("\n========================================")
    print("LANE GRAPH")
    print("========================================")
    print(f"Usable vehicles: {len(vehicles)}")
    print(f"Graph nodes:     {graph.number_of_nodes()}")
    print(f"Graph edges:     {graph.number_of_edges()}")

    print("\nVehicles:")
    for node_id, vehicle in enumerate(vehicles):
        print(
            f"{node_id:2d}: "
            f"x={vehicle['x']:7.2f} m | "
            f"z={vehicle['z']:7.2f} m | "
            f"yaw={vehicle['yaw']:7.3f} rad | "
            f"score={vehicle['score']:.3f} | "
            f"class={class_names[vehicle['label']]}"
        )

    print("\nCompatible vehicle pairs:")
    print_graph_edges(graph)

    streams = get_lane_streams(graph, min_vehicles=2)
    plot_lane_graph(
        graph,
        streams=streams,
        max_depth=50.0,
        x_range=(-12.0, 12.0),
        save_path="lane_graph_bev.png",
    )

    print("\n========================================")
    print("CANDIDATE LANE STREAMS")
    print("========================================")

    if not streams:
        print("No multi-vehicle streams found.")
        return

    for stream_idx, stream in enumerate(streams):
        stream = sorted(stream, key=lambda node: graph.nodes[node]["z"])
        print(f"\nLane stream {stream_idx}:")

        for node in stream:
            vehicle = graph.nodes[node]
            print(
                f"  node={node:2d} "
                f"x={vehicle['x']:7.2f} "
                f"z={vehicle['z']:7.2f} "
                f"yaw={vehicle['yaw']:7.3f} "
                f"score={vehicle['score']:.3f}"
            )


if __name__ == "__main__":
    main()
