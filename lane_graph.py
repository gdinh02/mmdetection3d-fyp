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
