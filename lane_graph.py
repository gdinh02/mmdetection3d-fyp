import math
from dataclasses import dataclass

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import cv2
import mmcv

from mmdet3d.structures import points_cam2img
from mmdet3d.visualization import Det3DLocalVisualizer

import mmengine



@dataclass
class LaneGraphConfig:
    # FCOS3D detection filtering
    score_thresh: float = 0.25
    max_depth: float = 60.0

    # Pairwise compatibility gates
    max_cross_track: float = 1.6       # metres
    max_yaw_diff_deg: float = 15.0     # degrees
    max_along_track: float = 30.0      # metres

    # For soft edge weighting
    sigma_cross_track: float = 0.8
    sigma_yaw_deg: float = 8.0


def load_cam2img(
    info_file,
    cam_type="CAM_FRONT",
):
    info = mmengine.load(
        info_file
    )

    sample = info[
        "data_list"
    ][0]

    cam_info = sample[
        "images"
    ][cam_type]

    cam2img = np.asarray(
        cam_info["cam2img"],
        dtype=np.float32,
    )

    return cam2img

def _vehicle_rectangle(x, z, yaw, length, width):
    """
    Construct an oriented rectangle in BEV.
    """

    heading = np.array([
        np.cos(yaw),
        np.sin(yaw),
    ])

    normal = np.array([
        -heading[1],
        heading[0],
    ])

    centre = np.array([x, z])

    hl = length / 2.0
    hw = width / 2.0

    return np.stack([
        centre + hl * heading + hw * normal,
        centre + hl * heading - hw * normal,
        centre - hl * heading - hw * normal,
        centre - hl * heading + hw * normal,
    ])


def render_fcos3d_front_view(
    image_path,
    pred_instances_3d,
    vehicles,
    cam2img,
    class_names=None,
):
    """
    Render the original camera image with FCOS3D 3D bounding boxes.

    Only vehicles retained by the lane graph are drawn.

    Parameters
    ----------
    image_path : str

    pred_instances_3d :
        result.pred_instances_3d

    vehicles : list[dict]
        Output from build_lane_compatibility_graph()

    cam2img : ndarray
        Camera intrinsic / projection matrix.

    class_names : sequence[str] or None

    Returns
    -------
    image : np.ndarray RGB
    box_centres_2d : np.ndarray [N, 2]
    """

    # ------------------------------------------------------
    # Read original image
    # ------------------------------------------------------

    image = mmcv.imread(image_path)

    # OpenCV/mmcv -> RGB
    image = mmcv.imconvert(
        image,
        "bgr",
        "rgb",
    )

    boxes = pred_instances_3d.bboxes_3d

    # ------------------------------------------------------
    # Select EXACT detections used in the graph
    # ------------------------------------------------------

    original_indices = [
        v["original_idx"]
        for v in vehicles
    ]

    if len(original_indices) == 0:
        return image, np.empty((0, 2))

    selected_boxes = boxes[original_indices]

    # ------------------------------------------------------
    # Official MMDetection3D visualizer
    # ------------------------------------------------------

    visualizer = Det3DLocalVisualizer()

    visualizer.set_image(image.copy())

    input_meta = {
        "cam2img": np.asarray(
            cam2img,
            dtype=np.float32,
        )
    }

    visualizer.draw_proj_bboxes_3d(
        selected_boxes,
        input_meta,
    )

    rendered = visualizer.get_image()

    # ------------------------------------------------------
    # Project box centres so we can annotate node IDs
    # ------------------------------------------------------

    centres_3d = selected_boxes.gravity_center

    centres_2d = points_cam2img(
        centres_3d,
        np.asarray(
            cam2img,
            dtype=np.float32,
        ),
    )

    if hasattr(centres_2d, "detach"):
        centres_2d = (
            centres_2d
            .detach()
            .cpu()
            .numpy()
        )

    return rendered, centres_2d


def plot_front_and_lane_graph(
    image_path,
    pred_instances_3d,
    G,
    vehicles,
    cam2img,
    streams=None,
    class_names=None,
    max_depth=60.0,
    x_range=(-15, 15),
    save_path=None,
):
    """
    Side-by-side visualisation:

        LEFT:
            original camera image + projected FCOS3D boxes

        RIGHT:
            BEV lane-stream compatibility graph
    """

    if streams is None:
        streams = [
            list(component)
            for component
            in nx.connected_components(G)
        ]

    # ======================================================
    # Stream lookup
    # ======================================================

    node_to_stream = {}

    for stream_id, stream in enumerate(streams):
        for node in stream:
            node_to_stream[node] = stream_id

    # ======================================================
    # Render FCOS3D projected boxes
    # ======================================================

    front_image, centres_2d = render_fcos3d_front_view(
        image_path=image_path,
        pred_instances_3d=pred_instances_3d,
        vehicles=vehicles,
        cam2img=cam2img,
        class_names=class_names,
    )

    # ======================================================
    # Figure
    # ======================================================

    fig, (ax_front, ax_bev) = plt.subplots(
        1,
        2,
        figsize=(18, 9),
    )

    # ======================================================
    # LEFT: CAMERA VIEW
    # ======================================================

    ax_front.imshow(front_image)

    ax_front.set_title(
        "FCOS3D front-view detections"
    )

    ax_front.axis("off")

    # ------------------------------------------------------
    # Add graph node IDs / confidence to image
    # ------------------------------------------------------

    for node_id, (vehicle, centre) in enumerate(
        zip(vehicles, centres_2d)
    ):
        u = float(centre[0])
        v = float(centre[1])

        if not (
            np.isfinite(u)
            and np.isfinite(v)
        ):
            continue

        label = f"node {node_id}"

        if class_names is not None:
            cls = class_names[
                vehicle["label"]
            ]

            label += (
                f" | {cls}"
                f" | {vehicle['score']:.2f}"
            )
        else:
            label += (
                f" | {vehicle['score']:.2f}"
            )

        ax_front.text(
            u,
            v,
            label,
            fontsize=9,
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
            ),
            color="white",
        )

    # ======================================================
    # RIGHT: BEV GRAPH
    # ======================================================

    cmap = plt.get_cmap("tab10")

    # ------------------------------------------------------
    # Camera
    # ------------------------------------------------------

    ax_bev.scatter(
        0,
        0,
        marker="^",
        s=130,
        color="black",
    )

    ax_bev.text(
        0.3,
        0.5,
        "camera",
    )

    # ------------------------------------------------------
    # Compatibility edges
    # ------------------------------------------------------

    for i, j, edge in G.edges(data=True):

        vi = G.nodes[i]
        vj = G.nodes[j]

        stream_i = node_to_stream.get(i)
        stream_j = node_to_stream.get(j)

        if (
            stream_i is not None
            and stream_i == stream_j
        ):
            edge_colour = cmap(
                stream_i % 10
            )
        else:
            edge_colour = "gray"

        weight = edge.get(
            "weight",
            1.0,
        )

        ax_bev.plot(
            [
                vi["x"],
                vj["x"],
            ],
            [
                vi["z"],
                vj["z"],
            ],
            color=edge_colour,
            linewidth=0.75 + 3 * weight,
            alpha=0.65,
        )

    # ------------------------------------------------------
    # Vehicles
    # ------------------------------------------------------

    for node, vehicle in G.nodes(data=True):

        x = vehicle["x"]
        z = vehicle["z"]
        yaw = vehicle["yaw"]

        # Use real FCOS3D dimensions if we've stored them
        width = vehicle.get(
            "width",
            1.8,
        )

        length = vehicle.get(
            "length",
            4.5,
        )

        stream_id = node_to_stream.get(
            node
        )

        if stream_id is None:
            colour = "gray"
        else:
            colour = cmap(
                stream_id % 10
            )

        # --------------------------------------------------
        # Vehicle rectangle
        # --------------------------------------------------

        corners = _vehicle_rectangle(
            x=x,
            z=z,
            yaw=yaw,
            length=length,
            width=width,
        )

        polygon = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=colour,
            linewidth=2,
        )

        ax_bev.add_patch(
            polygon
        )

        # centre
        ax_bev.scatter(
            x,
            z,
            s=50,
            color=colour,
        )

        # --------------------------------------------------
        # Heading arrow
        # --------------------------------------------------

        heading_length = 2.5

        dx = (
            heading_length
            * np.cos(yaw)
        )

        dz = (
            heading_length
            * np.sin(yaw)
        )

        ax_bev.arrow(
            x,
            z,
            dx,
            dz,
            width=0.025,
            head_width=0.35,
            head_length=0.5,
            length_includes_head=True,
            color=colour,
        )

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        stream_text = (
            f"S{stream_id}"
            if stream_id is not None
            else "unclustered"
        )

        ax_bev.text(
            x + 0.25,
            z + 0.3,
            (
                f"node {node}\n"
                f"{stream_text}\n"
                f"{vehicle['score']:.2f}"
            ),
            fontsize=9,
        )

    # ======================================================
    # BEV formatting
    # ======================================================

    ax_bev.set_xlim(
        x_range
    )

    ax_bev.set_ylim(
        0,
        max_depth,
    )

    ax_bev.set_xlabel(
        "Lateral x [m]"
    )

    ax_bev.set_ylabel(
        "Forward z [m]"
    )

    ax_bev.set_title(
        "Lane-stream compatibility graph"
    )

    ax_bev.grid(
        True,
        alpha=0.3,
    )

    ax_bev.set_aspect(
        "equal",
        adjustable="box",
    )

    # ======================================================
    # Save/show
    # ======================================================

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved visualisation: "
            f"{save_path}"
        )

    plt.show()

def axial_angle_diff(a, b):
    """
    Difference between orientations where theta and theta + pi
    are considered equivalent.

    Returns angle in [0, pi/2].
    """
    d = np.abs(a - b)
    d = np.mod(d, np.pi)
    return np.minimum(d, np.pi - d)


def heading_from_yaw(yaw):
    """
    Convert BEV yaw to a unit heading vector [x, z].

    Assumes MMDetection3D's BEV yaw convention:
        yaw = 0     -> +x
        yaw = pi/2  -> +z
    """
    return np.array([
        np.cos(yaw),
        np.sin(yaw)
    ], dtype=np.float64)


def pair_metrics(p_i, yaw_i, p_j, yaw_j):
    """
    Compute pairwise lane-stream compatibility measurements.

    Parameters
    ----------
    p_i, p_j : array-like [x, z]
        Vehicle centres in camera BEV coordinates.

    yaw_i, yaw_j : float
        BEV yaw angles in radians.

    Returns
    -------
    dict containing:
        cross_track
        yaw_diff
        along_track
    """

    p_i = np.asarray(p_i, dtype=np.float64)
    p_j = np.asarray(p_j, dtype=np.float64)

    delta = p_j - p_i

    # Heading vectors
    h_i = heading_from_yaw(yaw_i)
    h_j = heading_from_yaw(yaw_j)

    # Normal vectors perpendicular to headings
    n_i = np.array([-h_i[1], h_i[0]])
    n_j = np.array([-h_j[1], h_j[0]])

    # Cross-track error measured from both vehicles
    cross_i = abs(np.dot(n_i, delta))
    cross_j = abs(np.dot(n_j, -delta))

    # Symmetric cross-track error
    cross_track = 0.5 * (cross_i + cross_j)

    # Orientation comparison, modulo 180 degrees
    yaw_diff = axial_angle_diff(yaw_i, yaw_j)

    # Longitudinal gap.
    # Again make this symmetric because headings won't be identical.
    along_i = abs(np.dot(h_i, delta))
    along_j = abs(np.dot(h_j, -delta))

    along_track = 0.5 * (along_i + along_j)

    return {
        "cross_track": float(cross_track),
        "yaw_diff": float(yaw_diff),
        "along_track": float(along_track),
    }

def get_lane_streams(G, min_vehicles=2):
    """
    Return connected components representing candidate lane streams.
    """
    streams = []

    for component in nx.connected_components(G):
        component = sorted(component)

        if len(component) >= min_vehicles:
            streams.append(component)

    return streams

def print_graph_edges(G):
    for i, j, data in G.edges(data=True):
        print(
            f"{i:2d} <-> {j:2d} | "
            f"cross={data['cross_track']:.2f} m | "
            f"yaw={data['yaw_diff_deg']:.1f} deg | "
            f"along={data['along_track']:.1f} m | "
            f"weight={data['weight']:.3f}"
        )

def _vehicle_rectangle(x, z, yaw, length=4.5, width=1.8):
    """
    Return 4 corners of an oriented vehicle rectangle in BEV.

    Coordinates are [x, z].
    yaw follows the BEV convention used by your graph code.
    """

    # Forward direction
    h = np.array([
        np.cos(yaw),
        np.sin(yaw),
    ])

    # Perpendicular/lateral direction
    n = np.array([
        -h[1],
        h[0],
    ])

    centre = np.array([x, z])

    half_l = length / 2.0
    half_w = width / 2.0

    front_left = centre + half_l * h + half_w * n
    front_right = centre + half_l * h - half_w * n
    rear_right = centre - half_l * h - half_w * n
    rear_left = centre - half_l * h + half_w * n

    return np.stack([
        front_left,
        front_right,
        rear_right,
        rear_left,
    ])


def plot_lane_graph(
    G,
    vehicles,
    streams=None,
    max_depth=60,
    x_range=(-15, 15),
    show_labels=True,
    save_path=None,
    show=True
):
    """
    Plot FCOS3D vehicle detections and the lane compatibility graph.

    Parameters
    ----------
    G : networkx.Graph
        Output from build_lane_compatibility_graph()

    vehicles : list[dict]
        Vehicle list returned by build_lane_compatibility_graph()

    streams : list[list[int]] or None
        Output from get_lane_streams().
        If None, connected components are calculated automatically.

    max_depth : float
        Maximum forward range shown.

    x_range : tuple
        Lateral plotting limits.

    show_labels : bool
        Show node IDs next to detections.

    save_path : str or None
        If given, save figure to this path.
    """

    if streams is None:
        import networkx as nx
        streams = [
            list(c)
            for c in nx.connected_components(G)
        ]

    # Assign stream ID to each node
    node_to_stream = {}

    for stream_id, stream in enumerate(streams):
        for node in stream:
            node_to_stream[node] = stream_id

    fig, ax = plt.subplots(figsize=(9, 12))

    # --------------------------------------------------------
    # Camera / ego position
    # --------------------------------------------------------

    ax.scatter(
        0,
        0,
        marker="^",
        s=150,
        label="Camera",
    )

    ax.text(
        0.3,
        0.5,
        "camera",
    )

    # --------------------------------------------------------
    # Graph edges
    # --------------------------------------------------------

    for i, j, edge in G.edges(data=True):

        vi = G.nodes[i]
        vj = G.nodes[j]

        x = [
            vi["x"],
            vj["x"],
        ]

        z = [
            vi["z"],
            vj["z"],
        ]

        weight = edge.get("weight", 1.0)

        # Higher compatibility -> thicker edge
        linewidth = 0.5 + 3.0 * weight

        ax.plot(
            x,
            z,
            linewidth=linewidth,
            alpha=0.5,
        )

    # --------------------------------------------------------
    # Vehicles
    # --------------------------------------------------------

    for node, vehicle in G.nodes(data=True):

        x = vehicle["x"]
        z = vehicle["z"]
        yaw = vehicle["yaw"]


        width = vehicle["width"]
        length = vehicle["length"]

        corners = _vehicle_rectangle(
            x=x,
            z=z,
            yaw=yaw,
            length=length,
            width=width,
        )

        poly = Polygon(
            corners,
            closed=True,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(poly)

        # Vehicle centre
        ax.scatter(
            x,
            z,
            s=50,
        )

        # ----------------------------------------------------
        # Heading arrow
        # ----------------------------------------------------

        heading_length = 2.5

        dx = heading_length * np.cos(yaw)
        dz = heading_length * np.sin(yaw)

        ax.arrow(
            x,
            z,
            dx,
            dz,
            width=0.025,
            head_width=0.35,
            head_length=0.5,
            length_includes_head=True,
        )

        # ----------------------------------------------------
        # Node label
        # ----------------------------------------------------

        if show_labels:

            stream_id = node_to_stream.get(
                node,
                -1,
            )

            text = (
                f"{node}\n"
                f"S{stream_id}\n"
                f"{vehicle['score']:.2f}"
            )

            ax.text(
                x + 0.25,
                z + 0.25,
                text,
                fontsize=9,
            )

    # --------------------------------------------------------
    # Plot configuration
    # --------------------------------------------------------

    ax.set_xlim(
        x_range[0],
        x_range[1],
    )

    ax.set_ylim(
        0,
        max_depth,
    )

    ax.set_xlabel(
        "Lateral x [m]"
    )

    ax.set_ylabel(
        "Forward z [m]"
    )

    ax.set_title(
        "FCOS3D lane-stream compatibility graph"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # Important: equal aspect lets metres look like metres
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved BEV visualisation to {save_path}"
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

def build_lane_compatibility_graph(
    pred_instances_3d,
    vehicle_label_ids,
    cfg=LaneGraphConfig(),
):
    """
    Build a vehicle lane-stream compatibility graph from an
    MMDetection3D prediction.

    Parameters
    ----------
    pred_instances_3d:
        result.pred_instances_3d from MMDetection3D.

        Expected to contain:
            .bboxes_3d
            .scores_3d
            .labels_3d

    vehicle_label_ids:
        Iterable of class IDs considered road vehicles.

        Example:
            vehicle_label_ids = {0, 1, 3}

        Check these against YOUR dataset config.

    cfg:
        LaneGraphConfig

    Returns
    -------
    G : networkx.Graph
        Node = vehicle detection
        Edge = pair is geometrically compatible with same lane

    vehicle_data : list[dict]
        Filtered detections used to construct graph.
    """

    boxes = pred_instances_3d.bboxes_3d
    scores = pred_instances_3d.scores_3d.detach().cpu()
    labels = pred_instances_3d.labels_3d.detach().cpu()

    # ----------------------------------------------------------
    # MMDetection3D BEV box
    #
    # CameraInstance3DBoxes.bev gives approximately:
    #
    #   [x, z, dx, dz, yaw_bev]
    #
    # so we don't manually deal with camera-coordinate yaw.
    # ----------------------------------------------------------
    bev = boxes.bev.detach().cpu().numpy()

    vehicle_label_ids = set(vehicle_label_ids)

    vehicle_data = []

    for original_idx in range(len(boxes)):
        score = float(scores[original_idx])
        label = int(labels[original_idx])

        if score < cfg.score_thresh:
            continue

        if label not in vehicle_label_ids:
            continue

        # x = float(bev[original_idx, 0])
        # z = float(bev[original_idx, 1])
        # yaw = float(bev[original_idx, 4])

        x = float(bev[original_idx, 0])
        z = float(bev[original_idx, 1])

        bev_width = float(bev[original_idx, 3])
        bev_length = float(bev[original_idx, 2])

        yaw = float(bev[original_idx, 4])

        # Remove objects behind camera / implausibly far away
        if z <= 0:
            continue

        if z > cfg.max_depth:
            continue

        vehicle_data.append({
            "original_idx": original_idx,
            "x": x,
            "z": z,
            "yaw": yaw,
            "width": bev_width,
            "length": bev_length,
            "score": score,
            "label": label,
        })

    # ----------------------------------------------------------
    # Create graph
    # ----------------------------------------------------------

    G = nx.Graph()

    for i, vehicle in enumerate(vehicle_data):
        G.add_node(
            i,
            **vehicle
        )

    max_yaw = math.radians(cfg.max_yaw_diff_deg)
    sigma_yaw = math.radians(cfg.sigma_yaw_deg)

    # ----------------------------------------------------------
    # Test every pair
    # ----------------------------------------------------------

    for i in range(len(vehicle_data)):
        vi = vehicle_data[i]

        p_i = np.array([vi["x"], vi["z"]])

        for j in range(i + 1, len(vehicle_data)):
            vj = vehicle_data[j]

            p_j = np.array([vj["x"], vj["z"]])

            metrics = pair_metrics(
                p_i,
                vi["yaw"],
                p_j,
                vj["yaw"],
            )

            cross = metrics["cross_track"]
            yaw_diff = metrics["yaw_diff"]
            along = metrics["along_track"]

            # --------------------------------------------------
            # Hard compatibility gates
            # --------------------------------------------------

            if cross > cfg.max_cross_track:
                continue

            if yaw_diff > max_yaw:
                continue

            if along > cfg.max_along_track:
                continue

            # --------------------------------------------------
            # Soft similarity score
            #
            # 1.0 = highly compatible
            # ~0  = poorly compatible
            # --------------------------------------------------

            cross_term = (
                cross / cfg.sigma_cross_track
            ) ** 2

            yaw_term = (
                yaw_diff / sigma_yaw
            ) ** 2

            compatibility = np.exp(
                -0.5 * (cross_term + yaw_term)
            )

            # Incorporate detector confidence if desired
            confidence = math.sqrt(
                vi["score"] * vj["score"]
            )

            edge_weight = compatibility * confidence

            G.add_edge(
                i,
                j,
                weight=float(edge_weight),
                cross_track=cross,
                yaw_diff_rad=yaw_diff,
                yaw_diff_deg=math.degrees(yaw_diff),
                along_track=along,
            )

    return G, vehicle_data