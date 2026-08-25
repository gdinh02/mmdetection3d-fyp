from mmdet3d.apis import inference_mono_3d_detector, init_model

from lane_graph import (
    LaneBoundaryConfig,
    LaneFitConfig,
    LaneGraphConfig,
    build_lane_compatibility_graph,
    fit_lane_streams,
    get_lane_streams,
    infer_lane_boundaries,
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
    if not streams:
        print("\nNo multi-vehicle streams found.")
        plot_lane_graph(
            graph,
            streams=streams,
            max_depth=50.0,
            x_range=(-12.0, 12.0),
            save_path="lane_graph_bev.png",
        )
        return

    lane_fits = fit_lane_streams(graph, streams, cfg=FIT_CONFIG)
    lane_boundaries = infer_lane_boundaries(lane_fits, cfg=BOUNDARY_CONFIG)

    print("\n========================================")
    print("ROBUST LANE FITS")
    print("========================================")
    for fit in lane_fits:
        coeff_text = ", ".join(f"{value:.5f}" for value in fit["coefficients"])
        print(
            f"Stream {fit['stream_id']}: "
            f"degree={fit['degree']} | "
            f"coefficients=[{coeff_text}] | "
            f"RMSE={fit['rmse']:.3f} m | "
            f"inliers={fit['inliers']} | "
            f"outliers={fit['outliers']}"
        )

    print("\n========================================")
    print("INFERRED LANE BOUNDARIES")
    print("========================================")
    if not lane_boundaries:
        print("No adjacent fitted streams passed the boundary checks.")
    else:
        for boundary_id, boundary in enumerate(lane_boundaries):
            coeff_text = ", ".join(
                f"{value:.5f}" for value in boundary["coefficients"]
            )
            print(
                f"Boundary {boundary_id}: "
                f"S{boundary['left_stream_id']} | S{boundary['right_stream_id']} | "
                f"width={boundary['lane_width']:.2f} m "
                f"(std={boundary['lane_width_std']:.2f}) | "
                f"overlap={boundary['overlap']:.1f} m | "
                f"coefficients=[{coeff_text}]"
            )

    plot_lane_graph(
        graph,
        streams=streams,
        lane_fits=lane_fits,
        lane_boundaries=lane_boundaries,
        max_depth=50.0,
        x_range=(-12.0, 12.0),
        save_path="lane_graph_bev.png",
    )


if __name__ == "__main__":
    main()
