import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon


@dataclass
class LaneGraphConfig:
    """Thresholds used to build the vehicle lane-compatibility graph."""

    score_thresh: float = 0.25
    max_depth: float = 60.0
    max_cross_track: float = 1.6
    max_yaw_diff_deg: float = 15.0
    max_along_track: float = 30.0
    sigma_cross_track: float = 0.8
    sigma_yaw_deg: float = 8.0


@dataclass
class LaneFitConfig:
    """Settings for robustly fitting a lane centreline x(z)."""

    degree: int = 2
    residual_threshold: float = 0.75
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneBoundaryConfig:
    """Settings for inferring lane boundaries between fitted lane streams."""

    min_overlap: float = 10.0
    min_lane_width: float = 2.5
    max_lane_width: float = 5.0
    sample_count: int = 50


@dataclass
class RoadPlaneConfig:
    """Settings for estimating camera-coordinate road height y(x, z)."""

    residual_threshold: float = 0.35
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneProjectionConfig:
    """Settings for projecting inferred lane boundaries into the camera image."""

    sample_count: int = 200
    min_depth: float = 1.0
    clip_to_image: bool = True


@dataclass
class TemporalConfig:
    """Settings for temporal ego-motion compensation and vehicle tracking."""

    history_frames: int = 5
    max_track_distance: float = 12.0
    max_track_yaw_diff_deg: float = 30.0
    max_track_frame_gap: int = 1
    temporal_decay: float = 0.90
    min_track_observations: int = 1
    max_reference_distance: float = 60.0
    max_time_gap_s: float = 3.0


def axial_angle_diff(a, b):
    """Return orientation difference in [0, pi/2], treating theta and theta+pi as equivalent."""
    diff = np.mod(np.abs(a - b), np.pi)
    return np.minimum(diff, np.pi - diff)


def heading_from_yaw(yaw):
    """Convert BEV yaw to a unit heading vector [x, z]."""
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)


def pair_metrics(p_i, yaw_i, p_j, yaw_j):
    """Compute symmetric cross-track, yaw, and along-track differences for a vehicle pair."""
    p_i = np.asarray(p_i, dtype=np.float64)
    p_j = np.asarray(p_j, dtype=np.float64)
    delta = p_j - p_i

    h_i = heading_from_yaw(yaw_i)
    h_j = heading_from_yaw(yaw_j)
    n_i = np.array([-h_i[1], h_i[0]])
    n_j = np.array([-h_j[1], h_j[0]])

    cross_track = 0.5 * (
        abs(np.dot(n_i, delta))
        + abs(np.dot(n_j, -delta))
    )
    yaw_diff = axial_angle_diff(yaw_i, yaw_j)
    along_track = 0.5 * (
        abs(np.dot(h_i, delta))
        + abs(np.dot(h_j, -delta))
    )

    return {
        "cross_track": float(cross_track),
        "yaw_diff": float(yaw_diff),
        "along_track": float(along_track),
    }


def extract_vehicles_from_prediction(pred_instances_3d, vehicle_label_ids, cfg=None):
    """Extract filtered FCOS3D vehicle detections as plain dictionaries."""
    if cfg is None:
        cfg = LaneGraphConfig()

    boxes = pred_instances_3d.bboxes_3d
    scores = pred_instances_3d.scores_3d.detach().cpu()
    labels = pred_instances_3d.labels_3d.detach().cpu()
    bev = boxes.bev.detach().cpu().numpy()

    if hasattr(boxes, "gravity_center"):
        centres = boxes.gravity_center.detach().cpu().numpy()
    else:
        centres = np.column_stack([
            bev[:, 0],
            np.zeros(len(bev), dtype=np.float64),
            bev[:, 1],
        ])

    vehicle_label_ids = set(vehicle_label_ids)
    vehicles = []

    for original_idx in range(len(boxes)):
        score = float(scores[original_idx])
        label = int(labels[original_idx])

        if score < cfg.score_thresh or label not in vehicle_label_ids:
            continue

        x = float(bev[original_idx, 0])
        z = float(bev[original_idx, 1])
        y = float(centres[original_idx, 1])
        length = float(bev[original_idx, 2])
        width = float(bev[original_idx, 3])
        yaw = float(bev[original_idx, 4])

        if z <= 0 or z > cfg.max_depth:
            continue

        vehicles.append({
            "original_idx": original_idx,
            "x": x,
            "y": y,
            "z": z,
            "yaw": yaw,
            "width": width,
            "length": length,
            "score": score,
            "evidence_weight": 1.0,
            "label": label,
        })

    return vehicles


def build_lane_compatibility_graph_from_vehicles(vehicles, cfg=None):
    """Build a lane-compatibility graph from vehicle dictionaries."""
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

            cross_term = (cross / cfg.sigma_cross_track) ** 2
            yaw_term = (yaw_diff / sigma_yaw) ** 2
            compatibility = np.exp(-0.5 * (cross_term + yaw_term))

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
    """Build a graph directly from an MMDetection3D prediction."""
    vehicles = extract_vehicles_from_prediction(
        pred_instances_3d,
        vehicle_label_ids,
        cfg=cfg,
    )
    graph = build_lane_compatibility_graph_from_vehicles(vehicles, cfg=cfg)
    return graph, vehicles


def transform_vehicles_to_reference(
    vehicles,
    current_cam_to_global,
    reference_cam_to_global,
    frame_index=None,
    frame_age=0,
    temporal_decay=1.0,
):
    """Transform camera-frame vehicle detections into one reference camera frame."""
    current_cam_to_global = np.asarray(current_cam_to_global, dtype=np.float64)
    reference_cam_to_global = np.asarray(reference_cam_to_global, dtype=np.float64)

    if current_cam_to_global.shape != (4, 4) or reference_cam_to_global.shape != (4, 4):
        raise ValueError("Camera-to-global transforms must both be 4x4 matrices")

    reference_from_current = np.linalg.inv(reference_cam_to_global) @ current_cam_to_global
    rotation = reference_from_current[:3, :3]
    transformed = []

    for vehicle in vehicles:
        point = np.array([
            vehicle["x"],
            vehicle.get("y", 0.0),
            vehicle["z"],
            1.0,
        ])
        point_ref = reference_from_current @ point

        heading = np.array([
            np.cos(vehicle["yaw"]),
            0.0,
            np.sin(vehicle["yaw"]),
        ])
        heading_ref = rotation @ heading
        horizontal_norm = np.hypot(heading_ref[0], heading_ref[2])
        if horizontal_norm < 1e-8:
            continue

        updated = dict(vehicle)
        updated.update({
            "x": float(point_ref[0]),
            "y": float(point_ref[1]),
            "z": float(point_ref[2]),
            "yaw": float(np.arctan2(heading_ref[2], heading_ref[0])),
            "frame_index": frame_index,
            "frame_age": int(frame_age),
            "evidence_weight": float(temporal_decay ** frame_age),
        })
        transformed.append(updated)

    return transformed


def _track_cost(track, detection, max_yaw_diff):
    """Return a simple spatial association cost, or None when a match is implausible."""
    if track["label"] != detection["label"]:
        return None

    yaw_diff = axial_angle_diff(track["yaw"], detection["yaw"])
    if yaw_diff > max_yaw_diff:
        return None

    distance = float(np.hypot(
        detection["x"] - track["x"],
        detection["z"] - track["z"],
    ))
    return distance


def assign_temporal_tracks(frame_vehicles, cfg=None):
    """Assign persistent track IDs to ego-motion-compensated detections.

    Parameters
    ----------
    frame_vehicles : list[tuple[int, list[dict]]]
        Chronologically ordered ``(frame_index, detections)`` pairs, with all
        detections already expressed in the same reference camera frame.
    """
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
                if cost is None or cost > cfg.max_track_distance:
                    continue

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

            tracks[track_id].update({
                "x": detection["x"],
                "z": detection["z"],
                "yaw": detection["yaw"],
                "last_frame_order": frame_order,
                "observations": tracks[track_id]["observations"] + 1,
            })

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

    keep_tracks = {
        track_id
        for track_id, track in tracks.items()
        if track["observations"] >= cfg.min_track_observations
    }
    filtered = [
        detection
        for detection in all_detections
        if detection["track_id"] in keep_tracks
    ]
    return filtered, tracks


def accumulate_temporal_vehicle_evidence(frame_records, reference_record_index=-1, cfg=None):
    """Ego-compensate and track detections from several frames.

    Each frame record must contain ``frame_index``, ``vehicles`` and
    ``cam_to_global``. The returned detections all live in the reference
    camera coordinate system and can be fed directly to
    :func:`build_lane_compatibility_graph_from_vehicles`.
    """
    if cfg is None:
        cfg = TemporalConfig()

    if not frame_records:
        return [], {}

    reference_record = frame_records[reference_record_index]
    reference_transform = reference_record["cam_to_global"]
    reference_position = (
        len(frame_records) + reference_record_index
        if reference_record_index < 0
        else reference_record_index
    )

    transformed_frames = []
    for position, record in enumerate(frame_records):
        frame_age = abs(reference_position - position)
        transformed = transform_vehicles_to_reference(
            record["vehicles"],
            current_cam_to_global=record["cam_to_global"],
            reference_cam_to_global=reference_transform,
            frame_index=record["frame_index"],
            frame_age=frame_age,
            temporal_decay=cfg.temporal_decay,
        )
        transformed_frames.append((record["frame_index"], transformed))

    return assign_temporal_tracks(transformed_frames, cfg=cfg)


def get_lane_streams(graph, min_vehicles=2):
    """Return connected components large enough to be candidate lane streams."""
    return [
        sorted(component)
        for component in nx.connected_components(graph)
        if len(component) >= min_vehicles
    ]


def print_graph_edges(graph):
    """Print geometric compatibility metrics for each retained graph edge."""
    for i, j, data in graph.edges(data=True):
        print(
            f"{i:2d} <-> {j:2d} | "
            f"cross={data['cross_track']:.2f} m | "
            f"yaw={data['yaw_diff_deg']:.1f} deg | "
            f"along={data['along_track']:.1f} m | "
            f"weight={data['weight']:.3f}"
        )


def evaluate_lane_polynomial(coefficients, z):
    """Evaluate x(z) for coefficients stored as [a0, a1, a2, ...]."""
    z = np.asarray(z, dtype=np.float64)
    x = np.zeros_like(z, dtype=np.float64)

    for power, coefficient in enumerate(coefficients):
        x += coefficient * z ** power

    return x


def lane_polynomial_slope(coefficients, z):
    """Evaluate dx/dz for coefficients stored as [a0, a1, a2, ...]."""
    z = np.asarray(z, dtype=np.float64)
    slope = np.zeros_like(z, dtype=np.float64)

    for power, coefficient in enumerate(coefficients[1:], start=1):
        slope += power * coefficient * z ** (power - 1)

    return slope


def fit_lane_stream(graph, stream, cfg=None):
    """
    Robustly fit a polynomial lane centreline x(z) to one candidate stream.

    Coefficients are returned as [a0, a1, a2, ...], so a quadratic is
    x(z) = a0 + a1*z + a2*z^2.
    """
    if cfg is None:
        cfg = LaneFitConfig()

    nodes = sorted(stream, key=lambda node: graph.nodes[node]["z"])
    if len(nodes) < 2:
        return None

    z = np.array([graph.nodes[node]["z"] for node in nodes], dtype=np.float64)
    x = np.array([graph.nodes[node]["x"] for node in nodes], dtype=np.float64)
    scores = np.array([
        graph.nodes[node]["score"] * graph.nodes[node].get("evidence_weight", 1.0)
        for node in nodes
    ], dtype=np.float64)

    degree = min(cfg.degree, len(nodes) - 1)
    sample_size = degree + 1
    rng = np.random.default_rng(cfg.random_seed)

    best_mask = None
    best_inlier_count = -1
    best_rmse = np.inf
    trials = 1 if len(nodes) == sample_size else cfg.max_trials

    for _ in range(trials):
        if len(nodes) == sample_size:
            sample_idx = np.arange(len(nodes))
        else:
            sample_idx = rng.choice(len(nodes), size=sample_size, replace=False)

        if np.unique(z[sample_idx]).size < sample_size:
            continue

        try:
            poly_desc = np.polyfit(z[sample_idx], x[sample_idx], degree)
        except (np.linalg.LinAlgError, ValueError):
            continue

        predicted = np.polyval(poly_desc, z)
        residuals = np.abs(x - predicted)
        inlier_mask = residuals <= cfg.residual_threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count < sample_size:
            continue

        rmse = float(np.sqrt(np.mean((x[inlier_mask] - predicted[inlier_mask]) ** 2)))
        if (
            inlier_count > best_inlier_count
            or (inlier_count == best_inlier_count and rmse < best_rmse)
        ):
            best_mask = inlier_mask
            best_inlier_count = inlier_count
            best_rmse = rmse

    if best_mask is None:
        return None

    inlier_z = z[best_mask]
    inlier_x = x[best_mask]
    inlier_weights = np.sqrt(np.clip(scores[best_mask], 1e-6, None))

    try:
        final_desc = np.polyfit(
            inlier_z,
            inlier_x,
            degree,
            w=inlier_weights,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None

    final_prediction = np.polyval(final_desc, inlier_z)
    rmse = float(np.sqrt(np.mean((inlier_x - final_prediction) ** 2)))
    coefficients = final_desc[::-1].copy()

    return {
        "degree": degree,
        "coefficients": coefficients,
        "inliers": [node for node, keep in zip(nodes, best_mask) if keep],
        "outliers": [node for node, keep in zip(nodes, best_mask) if not keep],
        "rmse": rmse,
        "z_min": float(inlier_z.min()),
        "z_max": float(inlier_z.max()),
    }


def fit_lane_streams(graph, streams, cfg=None):
    """Fit every candidate stream and return only successful lane fits."""
    fits = []

    for stream_id, stream in enumerate(streams):
        fit = fit_lane_stream(graph, stream, cfg=cfg)
        if fit is not None:
            fit["stream_id"] = stream_id
            fits.append(fit)

    return fits


def _average_polynomials(coefficients_a, coefficients_b):
    """Return the pointwise midpoint polynomial between two x(z) curves."""
    degree = max(len(coefficients_a), len(coefficients_b))
    a = np.pad(np.asarray(coefficients_a, dtype=np.float64), (0, degree - len(coefficients_a)))
    b = np.pad(np.asarray(coefficients_b, dtype=np.float64), (0, degree - len(coefficients_b)))
    return 0.5 * (a + b)


def infer_lane_boundaries(lane_fits, cfg=None):
    """
    Infer shared lane boundaries between adjacent fitted lane streams.

    A candidate pair must overlap in z and have a plausible lane-centre
    separation. The inferred marking is the midpoint curve between the two
    centrelines over their shared depth range.
    """
    if cfg is None:
        cfg = LaneBoundaryConfig()

    boundaries = []

    for i in range(len(lane_fits)):
        fit_a = lane_fits[i]

        for j in range(i + 1, len(lane_fits)):
            fit_b = lane_fits[j]

            z_min = max(fit_a["z_min"], fit_b["z_min"])
            z_max = min(fit_a["z_max"], fit_b["z_max"])
            overlap = z_max - z_min

            if overlap < cfg.min_overlap:
                continue

            z_samples = np.linspace(z_min, z_max, cfg.sample_count)
            x_a = evaluate_lane_polynomial(fit_a["coefficients"], z_samples)
            x_b = evaluate_lane_polynomial(fit_b["coefficients"], z_samples)

            # Determine left/right using the median lateral ordering. Reject
            # crossing centrelines because they are not a stable adjacent pair.
            delta_x = x_b - x_a
            median_delta = float(np.median(delta_x))
            if abs(median_delta) < 1e-6:
                continue

            same_order = np.sign(delta_x) == np.sign(median_delta)
            if np.mean(same_order) < 0.9:
                continue

            if median_delta > 0:
                left_fit, right_fit = fit_a, fit_b
                x_left, x_right = x_a, x_b
            else:
                left_fit, right_fit = fit_b, fit_a
                x_left, x_right = x_b, x_a

            # Correct the simple x-gap by the average lane slope so the width
            # is measured approximately along the local normal direction.
            slope_left = lane_polynomial_slope(left_fit["coefficients"], z_samples)
            slope_right = lane_polynomial_slope(right_fit["coefficients"], z_samples)
            mean_slope = 0.5 * (slope_left + slope_right)

            x_gap = x_right - x_left
            normal_width = x_gap / np.sqrt(1.0 + mean_slope ** 2)
            median_width = float(np.median(normal_width))
            width_std = float(np.std(normal_width))

            if not (cfg.min_lane_width <= median_width <= cfg.max_lane_width):
                continue

            boundary_coefficients = _average_polynomials(
                left_fit["coefficients"],
                right_fit["coefficients"],
            )

            boundaries.append({
                "left_stream_id": left_fit["stream_id"],
                "right_stream_id": right_fit["stream_id"],
                "coefficients": boundary_coefficients,
                "z_min": float(z_min),
                "z_max": float(z_max),
                "overlap": float(overlap),
                "lane_width": median_width,
                "lane_width_std": width_std,
            })

    # Present boundaries from left to right at the middle of their visible span.
    boundaries.sort(
        key=lambda boundary: float(
            evaluate_lane_polynomial(
                boundary["coefficients"],
                0.5 * (boundary["z_min"] + boundary["z_max"]),
            )
        )
    )
    return boundaries


def estimate_road_plane(pred_instances_3d, vehicles, cfg=None):
    """
    Estimate the road plane from FCOS3D vehicle bottom centres.

    Camera coordinates use x right, y down, z forward. The road is modelled as
        y = a*x + b*z + c
    and returned as coefficients [a, b, c].

    With only two usable vehicles, the fit falls back to y = b*z + c. With one
    vehicle, it falls back to a constant y plane.
    """
    if cfg is None:
        cfg = RoadPlaneConfig()

    if not vehicles:
        return None

    boxes = pred_instances_3d.bboxes_3d
    if not hasattr(boxes, "bottom_center"):
        return None

    bottom_centres = boxes.bottom_center.detach().cpu().numpy()
    original_indices = [vehicle["original_idx"] for vehicle in vehicles]
    points = np.asarray(bottom_centres[original_indices], dtype=np.float64)

    finite = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0)
    points = points[finite]
    if len(points) == 0:
        return None

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # One point: constant-height fallback.
    if len(points) == 1:
        return {
            "coefficients": np.array([0.0, 0.0, float(y[0])]),
            "inlier_count": 1,
            "total_count": 1,
            "rmse": 0.0,
            "model": "constant",
        }

    # Two points: fit y(z), because a full 2-D plane is underdetermined.
    if len(points) == 2:
        design = np.column_stack([z, np.ones_like(z)])
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        b, c = coeff
        prediction = design @ coeff
        rmse = float(np.sqrt(np.mean((y - prediction) ** 2)))
        return {
            "coefficients": np.array([0.0, float(b), float(c)]),
            "inlier_count": 2,
            "total_count": 2,
            "rmse": rmse,
            "model": "z_line",
        }

    rng = np.random.default_rng(cfg.random_seed)
    best_mask = None
    best_inlier_count = -1
    best_rmse = np.inf

    design_all = np.column_stack([x, z, np.ones_like(x)])

    for _ in range(cfg.max_trials):
        sample_idx = rng.choice(len(points), size=3, replace=False)
        design = design_all[sample_idx]

        # Reject degenerate samples that cannot define y = ax + bz + c.
        if np.linalg.matrix_rank(design) < 3:
            continue

        try:
            coeff, *_ = np.linalg.lstsq(design, y[sample_idx], rcond=None)
        except np.linalg.LinAlgError:
            continue

        prediction = design_all @ coeff
        residuals = np.abs(y - prediction)
        inlier_mask = residuals <= cfg.residual_threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count < 3:
            continue

        rmse = float(np.sqrt(np.mean((y[inlier_mask] - prediction[inlier_mask]) ** 2)))
        if (
            inlier_count > best_inlier_count
            or (inlier_count == best_inlier_count and rmse < best_rmse)
        ):
            best_mask = inlier_mask
            best_inlier_count = inlier_count
            best_rmse = rmse

    if best_mask is None:
        # Conservative fallback: fit y(z) to all points rather than returning
        # an unstable full plane.
        design = np.column_stack([z, np.ones_like(z)])
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        b, c = coeff
        prediction = design @ coeff
        rmse = float(np.sqrt(np.mean((y - prediction) ** 2)))
        return {
            "coefficients": np.array([0.0, float(b), float(c)]),
            "inlier_count": len(points),
            "total_count": len(points),
            "rmse": rmse,
            "model": "z_line_fallback",
        }

    final_design = design_all[best_mask]
    final_y = y[best_mask]
    final_coeff, *_ = np.linalg.lstsq(final_design, final_y, rcond=None)
    prediction = final_design @ final_coeff
    rmse = float(np.sqrt(np.mean((final_y - prediction) ** 2)))

    return {
        "coefficients": np.asarray(final_coeff, dtype=np.float64),
        "inlier_count": int(best_mask.sum()),
        "total_count": len(points),
        "rmse": rmse,
        "model": "plane",
    }


def road_plane_y(road_plane, x, z):
    """Evaluate camera-coordinate road height y from y = a*x + b*z + c."""
    a, b, c = np.asarray(road_plane["coefficients"], dtype=np.float64)
    return a * np.asarray(x) + b * np.asarray(z) + c


def project_camera_points(points_3d, cam2img):
    """Project Nx3 camera-coordinate points into image pixels."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    cam2img = np.asarray(cam2img, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")

    if cam2img.shape == (3, 3):
        projected = points_3d @ cam2img.T
    elif cam2img.shape == (3, 4):
        homogeneous = np.column_stack([points_3d, np.ones(len(points_3d))])
        projected = homogeneous @ cam2img.T
    elif cam2img.shape == (4, 4):
        homogeneous = np.column_stack([points_3d, np.ones(len(points_3d))])
        projected = homogeneous @ cam2img.T
        projected = projected[:, :3]
    else:
        raise ValueError(f"Unsupported cam2img shape: {cam2img.shape}")

    denominator = projected[:, 2]
    pixels = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
    valid = np.abs(denominator) > 1e-8
    pixels[valid] = projected[valid, :2] / denominator[valid, None]
    return pixels


def project_lane_boundaries_to_image(
    lane_boundaries,
    road_plane,
    cam2img,
    image_shape=None,
    cfg=None,
):
    """Project inferred BEV lane-boundary curves into camera-image pixels."""
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

        projected_boundaries.append({
            "boundary_id": boundary_id,
            "left_stream_id": boundary["left_stream_id"],
            "right_stream_id": boundary["right_stream_id"],
            "pixels": pixels[valid],
            "points_3d": points_3d[valid],
            "lane_width": boundary["lane_width"],
        })

    return projected_boundaries


def plot_projected_lane_boundaries(
    image_path,
    projected_boundaries,
    save_path=None,
    show=True,
):
    """Overlay projected inferred lane boundaries on the original camera image."""
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)

    for boundary in projected_boundaries:
        pixels = boundary["pixels"]
        if len(pixels) < 2:
            continue

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
    """Return the four corners of an oriented vehicle rectangle in BEV."""
    heading = heading_from_yaw(yaw)
    normal = np.array([-heading[1], heading[0]])
    centre = np.array([x, z])

    half_length = length / 2.0
    half_width = width / 2.0

    return np.stack([
        centre + half_length * heading + half_width * normal,
        centre + half_length * heading - half_width * normal,
        centre - half_length * heading - half_width * normal,
        centre - half_length * heading + half_width * normal,
    ])


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
    """Plot vehicles, compatibility graph, fitted centrelines, and inferred boundaries in BEV."""
    if streams is None:
        streams = [list(component) for component in nx.connected_components(graph)]

    node_to_stream = {
        node: stream_id
        for stream_id, stream in enumerate(streams)
        for node in stream
    }

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.scatter(0, 0, marker="^", s=150, label="Camera")
    ax.text(0.3, 0.5, "camera")

    for i, j, edge in graph.edges(data=True):
        vi = graph.nodes[i]
        vj = graph.nodes[j]
        linewidth = 0.5 + 3.0 * edge.get("weight", 1.0)
        ax.plot(
            [vi["x"], vj["x"]],
            [vi["z"], vj["z"]],
            linewidth=linewidth,
            alpha=0.5,
        )

    for node, vehicle in graph.nodes(data=True):
        x = vehicle["x"]
        z = vehicle["z"]
        yaw = vehicle["yaw"]

        corners = _vehicle_rectangle(
            x=x,
            z=z,
            yaw=yaw,
            length=vehicle["length"],
            width=vehicle["width"],
        )
        ax.add_patch(Polygon(corners, closed=True, fill=False, linewidth=2))
        ax.scatter(x, z, s=50)

        heading_length = 2.5
        ax.arrow(
            x,
            z,
            heading_length * np.cos(yaw),
            heading_length * np.sin(yaw),
            width=0.025,
            head_width=0.35,
            head_length=0.5,
            length_includes_head=True,
        )

        if show_labels:
            stream_id = node_to_stream.get(node, -1)
            ax.text(
                x + 0.25,
                z + 0.25,
                (
                    f"{node}\nS{stream_id}\n{vehicle['score']:.2f}"
                    + (
                        f"\nT{vehicle['track_id']} F{vehicle['frame_index']}"
                        if "track_id" in vehicle
                        else ""
                    )
                ),
                fontsize=8,
            )

    if lane_fits is not None:
        for fit in lane_fits:
            z_curve = np.linspace(fit["z_min"], fit["z_max"], 100)
            x_curve = evaluate_lane_polynomial(fit["coefficients"], z_curve)
            ax.plot(
                x_curve,
                z_curve,
                linewidth=3,
                label=f"centre S{fit['stream_id']}",
            )

            for node in fit["outliers"]:
                vehicle = graph.nodes[node]
                ax.scatter(
                    vehicle["x"],
                    vehicle["z"],
                    marker="x",
                    s=100,
                    linewidths=2,
                )

    if lane_boundaries is not None:
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
    ax.set_title("FCOS3D lane-stream graph and inferred boundaries")
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
