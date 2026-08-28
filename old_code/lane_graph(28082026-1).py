import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon


@dataclass
class LaneGraphConfig:
    score_thresh: float = 0.25
    max_depth: float = 60.0
    max_cross_track: float = 1.6
    max_yaw_diff_deg: float = 15.0
    max_along_track: float = 30.0
    sigma_cross_track: float = 0.8
    sigma_yaw_deg: float = 8.0


@dataclass
class TemporalConfig:
    history_frames: int = 5
    max_track_distance: float = 12.0
    max_track_yaw_diff_deg: float = 30.0
    max_track_frame_gap: int = 1
    temporal_decay: float = 0.90
    min_track_observations: int = 1


@dataclass
class LaneFitConfig:
    degree: int = 2
    residual_threshold: float = 0.75
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneBoundaryConfig:
    min_overlap: float = 10.0
    min_lane_width: float = 2.5
    max_lane_width: float = 5.0
    sample_count: int = 50


@dataclass
class RoadPlaneConfig:
    residual_threshold: float = 0.35
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneProjectionConfig:
    sample_count: int = 200
    min_depth: float = 1.0
    clip_to_image: bool = True


def axial_angle_diff(a, b):
    diff = np.mod(np.abs(a - b), np.pi)
    return np.minimum(diff, np.pi - diff)


def heading_from_yaw(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)


def pair_metrics(p_i, yaw_i, p_j, yaw_j):
    p_i = np.asarray(p_i, dtype=np.float64)
    p_j = np.asarray(p_j, dtype=np.float64)
    delta = p_j - p_i

    h_i = heading_from_yaw(yaw_i)
    h_j = heading_from_yaw(yaw_j)
    n_i = np.array([-h_i[1], h_i[0]])
    n_j = np.array([-h_j[1], h_j[0]])

    cross_track = 0.5 * (
        abs(np.dot(n_i, delta)) + abs(np.dot(n_j, -delta))
    )
    yaw_diff = axial_angle_diff(yaw_i, yaw_j)
    along_track = 0.5 * (
        abs(np.dot(h_i, delta)) + abs(np.dot(h_j, -delta))
    )

    return {
        "cross_track": float(cross_track),
        "yaw_diff": float(yaw_diff),
        "along_track": float(along_track),
    }


def extract_vehicles_from_prediction(pred_instances_3d, vehicle_label_ids, cfg=None):
    if cfg is None:
        cfg = LaneGraphConfig()

    boxes = pred_instances_3d.bboxes_3d
    scores = pred_instances_3d.scores_3d.detach().cpu()
    labels = pred_instances_3d.labels_3d.detach().cpu()
    bev = boxes.bev.detach().cpu().numpy()

    if hasattr(boxes, "gravity_center"):
        centres = boxes.gravity_center.detach().cpu().numpy()
    else:
        centres = np.column_stack(
            [bev[:, 0], np.zeros(len(bev), dtype=np.float64), bev[:, 1]]
        )

    vehicle_label_ids = set(vehicle_label_ids)
    vehicles = []

    for original_idx in range(len(boxes)):
        score = float(scores[original_idx])
        label = int(labels[original_idx])
        if score < cfg.score_thresh or label not in vehicle_label_ids:
            continue

        x = float(bev[original_idx, 0])
        z = float(bev[original_idx, 1])
        if z <= 0 or z > cfg.max_depth:
            continue

        vehicles.append(
            {
                "original_idx": original_idx,
                "x": x,
                "y": float(centres[original_idx, 1]),
                "z": z,
                "yaw": float(bev[original_idx, 4]),
                "length": float(bev[original_idx, 2]),
                "width": float(bev[original_idx, 3]),
                "score": score,
                "evidence_weight": 1.0,
                "label": label,
            }
        )

    return vehicles


def build_lane_compatibility_graph_from_vehicles(vehicles, cfg=None):
    if cfg is None:
        cfg = LaneGraphConfig()

    graph = nx.Graph()
    for node_id, vehicle in enumerate(vehicles):
        graph.add_node(node_id, **vehicle)

    max_yaw = math.radians(cfg.max_yaw_diff_deg)
    sigma_yaw = math.radians(cfg.sigma_yaw_deg)

    for i, vi in enumerate(vehicles):
        p_i = np.array([vi["x"], vi["z"]])
        for j in range(i + 1, len(vehicles)):
            vj = vehicles[j]
            p_j = np.array([vj["x"], vj["z"]])
            metrics = pair_metrics(p_i, vi["yaw"], p_j, vj["yaw"])

            cross = metrics["cross_track"]
            yaw_diff = metrics["yaw_diff"]
            along = metrics["along_track"]
            if (
                cross > cfg.max_cross_track
                or yaw_diff > max_yaw
                or along > cfg.max_along_track
            ):
                continue

            compatibility = np.exp(
                -0.5
                * (
                    (cross / cfg.sigma_cross_track) ** 2
                    + (yaw_diff / sigma_yaw) ** 2
                )
            )
            score_i = vi["score"] * vi.get("evidence_weight", 1.0)
            score_j = vj["score"] * vj.get("evidence_weight", 1.0)
            confidence = math.sqrt(max(score_i, 0.0) * max(score_j, 0.0))

            graph.add_edge(
                i,
                j,
                weight=float(compatibility * confidence),
                cross_track=cross,
                yaw_diff_deg=math.degrees(yaw_diff),
                along_track=along,
            )

    return graph


def build_lane_compatibility_graph(pred_instances_3d, vehicle_label_ids, cfg=None):
    vehicles = extract_vehicles_from_prediction(
        pred_instances_3d, vehicle_label_ids, cfg=cfg
    )
    return build_lane_compatibility_graph_from_vehicles(vehicles, cfg=cfg), vehicles


def transform_vehicles_to_reference(
    vehicles,
    current_cam_to_global,
    reference_cam_to_global,
    frame_index=None,
    frame_age=0,
    temporal_decay=1.0,
):
    current_cam_to_global = np.asarray(current_cam_to_global, dtype=np.float64)
    reference_cam_to_global = np.asarray(reference_cam_to_global, dtype=np.float64)
    reference_from_current = (
        np.linalg.inv(reference_cam_to_global) @ current_cam_to_global
    )
    rotation = reference_from_current[:3, :3]

    transformed = []
    for vehicle in vehicles:
        point = np.array(
            [vehicle["x"], vehicle.get("y", 0.0), vehicle["z"], 1.0]
        )
        point_ref = reference_from_current @ point

        heading = np.array(
            [np.cos(vehicle["yaw"]), 0.0, np.sin(vehicle["yaw"])]
        )
        heading_ref = rotation @ heading
        if np.hypot(heading_ref[0], heading_ref[2]) < 1e-8:
            continue

        updated = dict(vehicle)
        updated.update(
            {
                "x": float(point_ref[0]),
                "y": float(point_ref[1]),
                "z": float(point_ref[2]),
                "yaw": float(np.arctan2(heading_ref[2], heading_ref[0])),
                "frame_index": frame_index,
                "frame_age": int(frame_age),
                "evidence_weight": float(temporal_decay**frame_age),
            }
        )
        transformed.append(updated)

    return transformed


def _track_cost(track, detection, max_yaw_diff):
    if track["label"] != detection["label"]:
        return None
    if axial_angle_diff(track["yaw"], detection["yaw"]) > max_yaw_diff:
        return None
    return float(
        np.hypot(
            detection["x"] - track["x"],
            detection["z"] - track["z"],
        )
    )


def assign_temporal_tracks(frame_vehicles, cfg=None):
    if cfg is None:
        cfg = TemporalConfig()

    max_yaw_diff = math.radians(cfg.max_track_yaw_diff_deg)
    tracks = {}
    next_track_id = 0
    all_detections = []

    for frame_order, (frame_index, detections) in enumerate(frame_vehicles):
        candidates = []
        for det_idx, detection in enumerate(detections):
            for track_id, track in tracks.items():
                frame_gap = frame_order - track["last_frame_order"]
                if frame_gap <= 0 or frame_gap > cfg.max_track_frame_gap:
                    continue
                cost = _track_cost(track, detection, max_yaw_diff)
                if cost is not None and cost <= cfg.max_track_distance:
                    candidates.append((cost, track_id, det_idx))

        candidates.sort(key=lambda item: item[0])
        matched_tracks = set()
        matched_detections = set()

        for _, track_id, det_idx in candidates:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            detection = detections[det_idx]
            detection["track_id"] = track_id
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            tracks[track_id].update(
                {
                    "x": detection["x"],
                    "z": detection["z"],
                    "yaw": detection["yaw"],
                    "last_frame_order": frame_order,
                    "observations": tracks[track_id]["observations"] + 1,
                }
            )

        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = next_track_id
                next_track_id += 1
                detection["track_id"] = track_id
                tracks[track_id] = {
                    "label": detection["label"],
                    "x": detection["x"],
                    "z": detection["z"],
                    "yaw": detection["yaw"],
                    "last_frame_order": frame_order,
                    "observations": 1,
                }
            all_detections.append(detection)

    if cfg.min_track_observations <= 1:
        return all_detections, tracks

    keep = {
        track_id
        for track_id, track in tracks.items()
        if track["observations"] >= cfg.min_track_observations
    }
    return [d for d in all_detections if d["track_id"] in keep], tracks


def accumulate_temporal_vehicle_evidence(
    frame_records, reference_record_index=-1, cfg=None
):
    if cfg is None:
        cfg = TemporalConfig()
    if not frame_records:
        return [], {}

    ref_position = (
        len(frame_records) + reference_record_index
        if reference_record_index < 0
        else reference_record_index
    )
    reference_transform = frame_records[ref_position]["cam_to_global"]

    transformed_frames = []
    for position, record in enumerate(frame_records):
        frame_age = abs(ref_position - position)
        transformed = transform_vehicles_to_reference(
            record["vehicles"],
            record["cam_to_global"],
            reference_transform,
            frame_index=record["frame_index"],
            frame_age=frame_age,
            temporal_decay=cfg.temporal_decay,
        )
        transformed_frames.append((record["frame_index"], transformed))

    return assign_temporal_tracks(transformed_frames, cfg=cfg)


def get_lane_streams(graph, min_vehicles=2):
    return [
        sorted(component)
        for component in nx.connected_components(graph)
        if len(component) >= min_vehicles
    ]


def print_graph_edges(graph):
    for i, j, data in graph.edges(data=True):
        print(
            f"{i:2d} <-> {j:2d} | "
            f"cross={data['cross_track']:.2f} m | "
            f"yaw={data['yaw_diff_deg']:.1f} deg | "
            f"along={data['along_track']:.1f} m | "
            f"weight={data['weight']:.3f}"
        )


def evaluate_lane_polynomial(coefficients, z):
    z = np.asarray(z, dtype=np.float64)
    x = np.zeros_like(z, dtype=np.float64)
    for power, coefficient in enumerate(coefficients):
        x += coefficient * z**power
    return x


def lane_polynomial_slope(coefficients, z):
    z = np.asarray(z, dtype=np.float64)
    slope = np.zeros_like(z, dtype=np.float64)
    for power, coefficient in enumerate(coefficients[1:], start=1):
        slope += power * coefficient * z ** (power - 1)
    return slope


def fit_lane_stream(graph, stream, cfg=None):
    if cfg is None:
        cfg = LaneFitConfig()

    nodes = sorted(stream, key=lambda node: graph.nodes[node]["z"])
    if len(nodes) < 2:
        return None

    z = np.array([graph.nodes[n]["z"] for n in nodes], dtype=np.float64)
    x = np.array([graph.nodes[n]["x"] for n in nodes], dtype=np.float64)
    scores = np.array(
        [
            graph.nodes[n]["score"] * graph.nodes[n].get("evidence_weight", 1.0)
            for n in nodes
        ],
        dtype=np.float64,
    )

    degree = min(cfg.degree, len(nodes) - 1)
    sample_size = degree + 1
    trials = 1 if len(nodes) == sample_size else cfg.max_trials
    rng = np.random.default_rng(cfg.random_seed)
    best_mask = None
    best_count = -1
    best_rmse = np.inf

    for _ in range(trials):
        sample_idx = (
            np.arange(len(nodes))
            if len(nodes) == sample_size
            else rng.choice(len(nodes), size=sample_size, replace=False)
        )
        if np.unique(z[sample_idx]).size < sample_size:
            continue
        try:
            poly_desc = np.polyfit(z[sample_idx], x[sample_idx], degree)
        except (np.linalg.LinAlgError, ValueError):
            continue

        predicted = np.polyval(poly_desc, z)
        mask = np.abs(x - predicted) <= cfg.residual_threshold
        count = int(mask.sum())
        if count < sample_size:
            continue
        rmse = float(np.sqrt(np.mean((x[mask] - predicted[mask]) ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_mask, best_count, best_rmse = mask, count, rmse

    if best_mask is None:
        return None

    inlier_z = z[best_mask]
    inlier_x = x[best_mask]
    weights = np.sqrt(np.clip(scores[best_mask], 1e-6, None))
    try:
        final_desc = np.polyfit(inlier_z, inlier_x, degree, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None

    prediction = np.polyval(final_desc, inlier_z)
    return {
        "degree": degree,
        "coefficients": final_desc[::-1].copy(),
        "inliers": [n for n, keep in zip(nodes, best_mask) if keep],
        "outliers": [n for n, keep in zip(nodes, best_mask) if not keep],
        "rmse": float(np.sqrt(np.mean((inlier_x - prediction) ** 2))),
        "z_min": float(inlier_z.min()),
        "z_max": float(inlier_z.max()),
    }


def fit_lane_streams(graph, streams, cfg=None):
    fits = []
    for stream_id, stream in enumerate(streams):
        fit = fit_lane_stream(graph, stream, cfg=cfg)
        if fit is not None:
            fit["stream_id"] = stream_id
            fits.append(fit)
    return fits


def _average_polynomials(a, b):
    degree = max(len(a), len(b))
    a = np.pad(np.asarray(a, dtype=np.float64), (0, degree - len(a)))
    b = np.pad(np.asarray(b, dtype=np.float64), (0, degree - len(b)))
    return 0.5 * (a + b)


def infer_lane_boundaries(lane_fits, cfg=None):
    if cfg is None:
        cfg = LaneBoundaryConfig()

    boundaries = []

    for i in range(len(lane_fits)):
        for j in range(i + 1, len(lane_fits)):
            a, b = lane_fits[i], lane_fits[j]

            z_min = max(a["z_min"], b["z_min"])
            z_max = min(a["z_max"], b["z_max"])
            overlap = z_max - z_min

            print(
                f"\nS{a['stream_id']} vs S{b['stream_id']}: "
                f"z=[{z_min:.1f}, {z_max:.1f}], "
                f"overlap={overlap:.2f} m"
            )

            if overlap < cfg.min_overlap:
                print(
                    f"  -> rejected: insufficient overlap "
                    f"({overlap:.2f} < {cfg.min_overlap:.2f} m)"
                )
                continue

            z = np.linspace(
                z_min,
                z_max,
                cfg.sample_count,
            )

            x_a = evaluate_lane_polynomial(
                a["coefficients"],
                z,
            )
            x_b = evaluate_lane_polynomial(
                b["coefficients"],
                z,
            )

            delta = x_b - x_a
            median_delta = float(np.median(delta))
            median_x_gap = float(np.median(np.abs(delta)))

            if abs(median_delta) < 1e-6:
                print(
                    "  -> rejected: median lateral separation "
                    "is effectively zero"
                )
                continue

            sign_consistency = float(
                np.mean(
                    np.sign(delta)
                    == np.sign(median_delta)
                )
            )

            print(
                f"  median x-gap={median_x_gap:.2f} m | "
                f"left/right consistency={sign_consistency:.2%}"
            )

            if sign_consistency < 0.9:
                print(
                    "  -> rejected: streams do not maintain a "
                    "consistent left/right ordering"
                )
                continue

            if median_delta > 0:
                left, right = a, b
                x_left, x_right = x_a, x_b
            else:
                left, right = b, a
                x_left, x_right = x_b, x_a

            slope_left = lane_polynomial_slope(
                left["coefficients"],
                z,
            )
            slope_right = lane_polynomial_slope(
                right["coefficients"],
                z,
            )

            mean_slope = 0.5 * (
                slope_left + slope_right
            )

            widths = (
                (x_right - x_left)
                / np.sqrt(1.0 + mean_slope**2)
            )

            median_width = float(
                np.median(widths)
            )

            width_std = float(
                np.std(widths)
            )

            print(
                f"  normal lane width={median_width:.2f} m | "
                f"std={width_std:.2f} m"
            )

            if not (
                cfg.min_lane_width
                <= median_width
                <= cfg.max_lane_width
            ):
                print(
                    f"  -> rejected: lane width outside allowed range "
                    f"[{cfg.min_lane_width:.2f}, "
                    f"{cfg.max_lane_width:.2f}] m"
                )
                continue

            boundary = {
                "left_stream_id": left["stream_id"],
                "right_stream_id": right["stream_id"],
                "coefficients": _average_polynomials(
                    left["coefficients"],
                    right["coefficients"],
                ),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "overlap": float(overlap),
                "lane_width": median_width,
                "lane_width_std": width_std,
            }

            boundaries.append(boundary)

            print(
                f"  -> ACCEPTED boundary "
                f"S{left['stream_id']} | S{right['stream_id']}"
            )

    boundaries.sort(
        key=lambda boundary: float(
            evaluate_lane_polynomial(
                boundary["coefficients"],
                0.5
                * (
                    boundary["z_min"]
                    + boundary["z_max"]
                ),
            )
        )
    )

    print(
        f"\nTotal inferred lane boundaries: "
        f"{len(boundaries)}"
    )

    return boundaries


def estimate_road_plane(pred_instances_3d, vehicles, cfg=None):
    if cfg is None:
        cfg = RoadPlaneConfig()
    if not vehicles:
        return None

    boxes = pred_instances_3d.bboxes_3d
    if not hasattr(boxes, "bottom_center"):
        return None

    all_bottom = boxes.bottom_center.detach().cpu().numpy()
    indices = [vehicle["original_idx"] for vehicle in vehicles]
    points = np.asarray(all_bottom[indices], dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0)]
    if len(points) == 0:
        return None

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if len(points) == 1:
        return {
            "coefficients": np.array([0.0, 0.0, float(y[0])]),
            "rmse": 0.0,
            "inlier_count": 1,
            "total_count": 1,
            "model": "constant",
        }

    if len(points) == 2:
        design = np.column_stack([z, np.ones_like(z)])
        b, c = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ np.array([b, c])
        return {
            "coefficients": np.array([0.0, b, c]),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "inlier_count": 2,
            "total_count": 2,
            "model": "z_line",
        }

    design_all = np.column_stack([x, z, np.ones_like(x)])
    rng = np.random.default_rng(cfg.random_seed)
    best_mask = None
    best_count = -1
    best_rmse = np.inf

    for _ in range(cfg.max_trials):
        idx = rng.choice(len(points), size=3, replace=False)
        if np.linalg.matrix_rank(design_all[idx]) < 3:
            continue
        coeff = np.linalg.lstsq(design_all[idx], y[idx], rcond=None)[0]
        prediction = design_all @ coeff
        mask = np.abs(y - prediction) <= cfg.residual_threshold
        count = int(mask.sum())
        if count < 3:
            continue
        rmse = float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_mask, best_count, best_rmse = mask, count, rmse

    if best_mask is None:
        design = np.column_stack([z, np.ones_like(z)])
        b, c = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ np.array([b, c])
        return {
            "coefficients": np.array([0.0, b, c]),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "inlier_count": len(points),
            "total_count": len(points),
            "model": "z_line_fallback",
        }

    coeff = np.linalg.lstsq(
        design_all[best_mask], y[best_mask], rcond=None
    )[0]
    prediction = design_all[best_mask] @ coeff
    return {
        "coefficients": coeff,
        "rmse": float(np.sqrt(np.mean((y[best_mask] - prediction) ** 2))),
        "inlier_count": int(best_mask.sum()),
        "total_count": len(points),
        "model": "plane",
    }


def road_plane_y(road_plane, x, z):
    a, b, c = np.asarray(road_plane["coefficients"], dtype=np.float64)
    return a * np.asarray(x) + b * np.asarray(z) + c


def project_camera_points(points_3d, cam2img):
    points_3d = np.asarray(points_3d, dtype=np.float64)
    cam2img = np.asarray(cam2img, dtype=np.float64)
    if cam2img.shape == (3, 3):
        projected = points_3d @ cam2img.T
    elif cam2img.shape == (3, 4):
        projected = np.column_stack([points_3d, np.ones(len(points_3d))]) @ cam2img.T
    elif cam2img.shape == (4, 4):
        projected = (
            np.column_stack([points_3d, np.ones(len(points_3d))]) @ cam2img.T
        )[:, :3]
    else:
        raise ValueError(f"Unsupported cam2img shape: {cam2img.shape}")

    pixels = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-8
    pixels[valid] = projected[valid, :2] / projected[valid, 2, None]
    return pixels


def project_lane_boundaries_to_image(
    lane_boundaries, road_plane, cam2img, image_shape=None, cfg=None
):
    if cfg is None:
        cfg = LaneProjectionConfig()
    if road_plane is None:
        return []

    projected_boundaries = []
    for boundary_id, boundary in enumerate(lane_boundaries):
        z = np.linspace(boundary["z_min"], boundary["z_max"], cfg.sample_count)
        x = evaluate_lane_polynomial(boundary["coefficients"], z)
        y = road_plane_y(road_plane, x, z)
        points_3d = np.column_stack([x, y, z])
        pixels = project_camera_points(points_3d, cam2img)

        valid = (
            np.isfinite(pixels[:, 0])
            & np.isfinite(pixels[:, 1])
            & (z >= cfg.min_depth)
        )
        if cfg.clip_to_image and image_shape is not None:
            height, width = image_shape[:2]
            valid &= (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
        if not np.any(valid):
            continue

        projected_boundaries.append(
            {
                "boundary_id": boundary_id,
                "left_stream_id": boundary["left_stream_id"],
                "right_stream_id": boundary["right_stream_id"],
                "pixels": pixels[valid],
                "points_3d": points_3d[valid],
                "lane_width": boundary["lane_width"],
            }
        )

    return projected_boundaries


def plot_projected_lane_boundaries(
    image_path, projected_boundaries, save_path=None, show=True
):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)
    for boundary in projected_boundaries:
        pixels = boundary["pixels"]
        if len(pixels) >= 2:
            ax.plot(
                pixels[:, 0],
                pixels[:, 1],
                linewidth=3,
                label=(
                    f"boundary S{boundary['left_stream_id']}|"
                    f"S{boundary['right_stream_id']}"
                ),
            )
    ax.set_title("Projected inferred lane boundaries")
    ax.axis("off")
    if projected_boundaries:
        ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved image-space lane projection to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _vehicle_rectangle(x, z, yaw, length, width):
    heading = heading_from_yaw(yaw)
    normal = np.array([-heading[1], heading[0]])
    centre = np.array([x, z])
    half_length = length / 2.0
    half_width = width / 2.0
    return np.stack(
        [
            centre + half_length * heading + half_width * normal,
            centre + half_length * heading - half_width * normal,
            centre - half_length * heading - half_width * normal,
            centre - half_length * heading + half_width * normal,
        ]
    )


def plot_lane_graph(
    graph,
    streams=None,
    lane_fits=None,
    lane_boundaries=None,
    max_depth=60.0,
    x_range=(-15.0, 15.0),
    show_labels=True,
    save_path=None,
    show=True,
):
    if streams is None:
        streams = [list(c) for c in nx.connected_components(graph)]
    node_to_stream = {
        node: stream_id
        for stream_id, stream in enumerate(streams)
        for node in stream
    }

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.scatter(0, 0, marker="^", s=150, label="Camera")
    ax.text(0.3, 0.5, "camera")

    for i, j, edge in graph.edges(data=True):
        vi, vj = graph.nodes[i], graph.nodes[j]
        ax.plot(
            [vi["x"], vj["x"]],
            [vi["z"], vj["z"]],
            linewidth=0.5 + 3.0 * edge.get("weight", 1.0),
            alpha=0.5,
        )

    for node, vehicle in graph.nodes(data=True):
        x, z, yaw = vehicle["x"], vehicle["z"], vehicle["yaw"]
        corners = _vehicle_rectangle(
            x, z, yaw, vehicle["length"], vehicle["width"]
        )
        ax.add_patch(Polygon(corners, closed=True, fill=False, linewidth=2))
        ax.scatter(x, z, s=50)
        ax.arrow(
            x,
            z,
            2.5 * np.cos(yaw),
            2.5 * np.sin(yaw),
            width=0.025,
            head_width=0.35,
            head_length=0.5,
            length_includes_head=True,
        )
        if show_labels:
            label = (
                f"{node}\nS{node_to_stream.get(node, -1)}\n{vehicle['score']:.2f}"
            )
            if "track_id" in vehicle:
                label += f"\nT{vehicle['track_id']} F{vehicle['frame_index']}"
            ax.text(x + 0.25, z + 0.25, label, fontsize=8)

    if lane_fits:
        for fit in lane_fits:
            z_curve = np.linspace(fit["z_min"], fit["z_max"], 100)
            x_curve = evaluate_lane_polynomial(fit["coefficients"], z_curve)
            ax.plot(
                x_curve,
                z_curve,
                linewidth=3,
                label=f"centre S{fit['stream_id']}",
            )

    if lane_boundaries:
        for boundary in lane_boundaries:
            z_curve = np.linspace(boundary["z_min"], boundary["z_max"], 100)
            x_curve = evaluate_lane_polynomial(boundary["coefficients"], z_curve)
            ax.plot(
                x_curve,
                z_curve,
                linestyle="--",
                linewidth=3,
                label=(
                    f"boundary S{boundary['left_stream_id']}|"
                    f"S{boundary['right_stream_id']}"
                ),
            )

    ax.set_xlim(*x_range)
    ax.set_ylim(0, max_depth)
    ax.set_xlabel("Lateral x [m]")
    ax.set_ylabel("Forward z [m]")
    ax.set_title("Temporal FCOS3D lane graph")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    if lane_fits or lane_boundaries:
        ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved BEV visualisation to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
