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


def build_lane_compatibility_graph(
    pred_instances_3d,
    vehicle_label_ids,
    cfg=None,
):
    """Build a graph whose edges connect vehicle detections compatible with the same lane."""
    if cfg is None:
        cfg = LaneGraphConfig()

    boxes = pred_instances_3d.bboxes_3d
    scores = pred_instances_3d.scores_3d.detach().cpu()
    labels = pred_instances_3d.labels_3d.detach().cpu()
    bev = boxes.bev.detach().cpu().numpy()

    vehicle_label_ids = set(vehicle_label_ids)
    vehicles = []

    for original_idx in range(len(boxes)):
        score = float(scores[original_idx])
        label = int(labels[original_idx])

        if score < cfg.score_thresh or label not in vehicle_label_ids:
            continue

        x = float(bev[original_idx, 0])
        z = float(bev[original_idx, 1])
        length = float(bev[original_idx, 2])
        width = float(bev[original_idx, 3])
        yaw = float(bev[original_idx, 4])

        if z <= 0 or z > cfg.max_depth:
            continue

        vehicles.append({
            "original_idx": original_idx,
            "x": x,
            "z": z,
            "yaw": yaw,
            "width": width,
            "length": length,
            "score": score,
            "label": label,
        })

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
            confidence = math.sqrt(vi["score"] * vj["score"])

            graph.add_edge(
                i,
                j,
                weight=float(compatibility * confidence),
                cross_track=cross,
                yaw_diff_deg=math.degrees(yaw_diff),
                along_track=along,
            )

    return graph, vehicles


def get_lane_streams(graph, min_vehicles=2):
    """Return connected components large enough to be candidate lane streams."""
    return [
        sorted(component)
        for component in nx.connected_components(graph)
        if len(component) >= min_vehicles
    ]


def print_graph_edges(graph):
    """Print the geometric compatibility metrics for each retained graph edge."""
    for i, j, data in graph.edges(data=True):
        print(
            f"{i:2d} <-> {j:2d} | "
            f"cross={data['cross_track']:.2f} m | "
            f"yaw={data['yaw_diff_deg']:.1f} deg | "
            f"along={data['along_track']:.1f} m | "
            f"weight={data['weight']:.3f}"
        )


def _evaluate_polynomial(coefficients, z):
    """Evaluate x(z) for coefficients stored as [a0, a1, a2, ...]."""
    z = np.asarray(z, dtype=np.float64)
    x = np.zeros_like(z, dtype=np.float64)

    for power, coefficient in enumerate(coefficients):
        x += coefficient * z ** power

    return x


def fit_lane_stream(graph, stream, cfg=None):
    """
    Robustly fit a polynomial lane centreline x(z) to one candidate stream.

    The returned coefficients are ordered [a0, a1, a2, ...], so a
    quadratic fit represents x(z) = a0 + a1*z + a2*z^2.
    """
    if cfg is None:
        cfg = LaneFitConfig()

    nodes = sorted(stream, key=lambda node: graph.nodes[node]["z"])
    if len(nodes) < 2:
        return None

    z = np.array([graph.nodes[node]["z"] for node in nodes], dtype=np.float64)
    x = np.array([graph.nodes[node]["x"] for node in nodes], dtype=np.float64)
    scores = np.array([graph.nodes[node]["score"] for node in nodes], dtype=np.float64)

    # A quadratic needs three points. With only two vehicles, fall back to a line.
    degree = min(cfg.degree, len(nodes) - 1)
    sample_size = degree + 1
    rng = np.random.default_rng(cfg.random_seed)

    best_mask = None
    best_inlier_count = -1
    best_rmse = np.inf

    # If the stream only has the minimum number of points, there is nothing
    # for RANSAC to reject, so fit all points directly.
    trials = 1 if len(nodes) == sample_size else cfg.max_trials

    for _ in range(trials):
        if len(nodes) == sample_size:
            sample_idx = np.arange(len(nodes))
        else:
            sample_idx = rng.choice(len(nodes), size=sample_size, replace=False)

        # Repeated/near-identical z values make the polynomial poorly defined.
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

    # Refit using every RANSAC inlier. Higher-confidence detections receive
    # slightly more influence in the final least-squares estimate.
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

    # np.polyfit returns descending powers; expose the more readable
    # [a0, a1, a2, ...] convention to the rest of the lane pipeline.
    coefficients = final_desc[::-1].copy()
    inlier_nodes = [node for node, keep in zip(nodes, best_mask) if keep]
    outlier_nodes = [node for node, keep in zip(nodes, best_mask) if not keep]

    return {
        "degree": degree,
        "coefficients": coefficients,
        "inliers": inlier_nodes,
        "outliers": outlier_nodes,
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
    max_depth=60.0,
    x_range=(-15.0, 15.0),
    show_labels=True,
    save_path=None,
    show=True,
):
    """Plot FCOS3D vehicle detections and their lane-compatibility graph in BEV."""
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
                f"{node}\nS{stream_id}\n{vehicle['score']:.2f}",
                fontsize=9,
            )

    if lane_fits is not None:
        for fit in lane_fits:
            z_curve = np.linspace(fit["z_min"], fit["z_max"], 100)
            x_curve = _evaluate_polynomial(fit["coefficients"], z_curve)
            ax.plot(
                x_curve,
                z_curve,
                linewidth=3,
                label=f"lane fit S{fit['stream_id']}",
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

    ax.set_xlim(*x_range)
    ax.set_ylim(0, max_depth)
    ax.set_xlabel("Lateral x [m]")
    ax.set_ylabel("Forward z [m]")
    ax.set_title("FCOS3D lane-stream compatibility graph")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved BEV visualisation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
