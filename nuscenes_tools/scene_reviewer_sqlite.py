from pathlib import Path
from datetime import datetime
import os
import sqlite3

import pandas as pd
import streamlit as st

# run with: streamlit run scene_reviewer_sqlite.py --server.fileWatcherType none

# ============================================================
# CONFIG
# ============================================================

# Your actual nuScenes root.
DATAROOT = Path(
    os.environ.get(
        "NUSCENES_ROOT",
        "/mnt/z/nuscenes"
    )
)


# Directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# SQLite database next to this script
DB_PATH = SCRIPT_DIR / "nuscenes_camera.db"

# Review CSV next to this script
REVIEW_CSV = SCRIPT_DIR / "nuscenes_scene_reviews.csv"


CAMERA_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]

ALL_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

REVIEW_COLUMNS = [
    "scene_name",
    "scene_token",
    "decision",
    "tags",
    "notes",
    "representative_sample_token",
    "representative_cam_front",
    "location",
    "description",
    "reviewed_at",
]


# ============================================================
# DATABASE HELPERS
# ============================================================

def db_query(sql, params=()):
    """
    Run a small SQLite query and return rows as dictionaries.
    A fresh connection avoids Streamlit/threading issues.
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    finally:
        conn.close()


@st.cache_data
def load_scenes():
    """
    Only ~850 rows, so keeping the scene list in memory is fine.
    """

    return db_query(
        """
        SELECT
            s.token,
            s.name,
            s.description,
            s.log_token,
            s.first_sample_token,
            s.last_sample_token,
            s.nbr_samples,
            COALESCE(l.location, 'unknown') AS location
        FROM scenes s
        LEFT JOIN logs l
            ON s.log_token = l.token
        ORDER BY s.name
        """
    )


@st.cache_data
def load_scene_samples(scene_token):
    """
    Load only the samples belonging to one scene.
    """

    return db_query(
        """
        SELECT
            token,
            scene_token,
            timestamp,
            prev,
            next
        FROM samples
        WHERE scene_token = ?
        ORDER BY timestamp
        """,
        (scene_token,)
    )


@st.cache_data
def load_scene_keyframes(scene_token):
    """
    Load the six keyframe camera images for one scene.

    Returns:
        sample_token -> camera channel -> frame metadata
    """

    rows = db_query(
        """
        SELECT
            cf.token,
            cf.sample_token,
            cf.channel,
            cf.filename,
            cf.timestamp,
            cf.is_key_frame,
            cf.prev,
            cf.next
        FROM camera_frames cf
        JOIN samples s
            ON cf.sample_token = s.token
        WHERE
            s.scene_token = ?
            AND cf.is_key_frame = 1
        ORDER BY
            s.timestamp,
            cf.channel
        """,
        (scene_token,)
    )

    result = {}

    for row in rows:
        sample_token = row["sample_token"]
        channel = row["channel"]

        if sample_token not in result:
            result[sample_token] = {}

        result[sample_token][channel] = row

    return result


# ============================================================
# REVIEW HELPERS
# ============================================================

def load_reviews():

    if not REVIEW_CSV.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    df = pd.read_csv(REVIEW_CSV)

    for column in REVIEW_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[REVIEW_COLUMNS]


def existing_review(scene_token):

    df = load_reviews()

    if df.empty:
        return None

    match = df[
        df["scene_token"].astype(str) == scene_token
    ]

    if match.empty:
        return None

    return match.iloc[0]


def save_review(
    scene,
    sample,
    decision,
    tags,
    notes,
    cam_front_filename,
):

    df = load_reviews()

    record = {
        "scene_name": scene["name"],
        "scene_token": scene["token"],
        "decision": decision,
        "tags": tags,
        "notes": notes,
        "representative_sample_token": sample["token"],
        "representative_cam_front": cam_front_filename,
        "location": scene["location"],
        "description": scene["description"],
        "reviewed_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    if not df.empty:

        mask = (
            df["scene_token"].astype(str)
            == scene["token"]
        )

        if mask.any():

            for key, value in record.items():
                df.loc[mask, key] = value

        else:

            df = pd.concat(
                [
                    df,
                    pd.DataFrame([record])
                ],
                ignore_index=True,
            )

    else:

        df = pd.DataFrame([record])

    df.to_csv(
        REVIEW_CSV,
        index=False
    )


# ============================================================
# VALIDATION
# ============================================================

st.set_page_config(
    page_title="nuScenes Camera Reviewer",
    layout="wide",
)

st.title("nuScenes Camera Scene Reviewer")


if not DATAROOT.exists():

    st.error(
        f"nuScenes root not found:\n\n{DATAROOT}"
    )

    st.stop()


if not DB_PATH.exists():

    st.error(
        f"SQLite database not found:\n\n{DB_PATH}"
    )

    st.stop()


# ============================================================
# LOAD SCENE LIST
# ============================================================

scenes = load_scenes()

if not scenes:

    st.error("No scenes found in database.")
    st.stop()


scene_names = [
    scene["name"]
    for scene in scenes
]

scene_lookup = {
    scene["name"]: scene
    for scene in scenes
}


# ============================================================
# SESSION STATE
# ============================================================

if "selected_scene_name" not in st.session_state:
    st.session_state.selected_scene_name = scene_names[0]


def move_scene(amount):

    current = st.session_state.selected_scene_name
    index = scene_names.index(current)

    new_index = max(
        0,
        min(
            len(scene_names) - 1,
            index + amount
        )
    )

    st.session_state.selected_scene_name = (
        scene_names[new_index]
    )

def review_and_next(
    decision,
    scene,
    sample,
    cam_front_filename,
    tags_key,
    notes_key,
):
    """
    Save the current review and automatically move
    to the next scene.
    """

    tags = st.session_state.get(tags_key, "")
    notes = st.session_state.get(notes_key, "")

    save_review(
        scene=scene,
        sample=sample,
        decision=decision,
        tags=tags,
        notes=notes,
        cam_front_filename=cam_front_filename,
    )

    # Show confirmation after the rerun.
    st.session_state["review_message"] = (
        f"{scene['name']} marked {decision.upper()}"
    )

    # Advance to next scene.
    current_index = scene_names.index(scene["name"])

    if current_index < len(scene_names) - 1:
        st.session_state.selected_scene_name = (
            scene_names[current_index + 1]
        )


def move_to_next_unreviewed():

    reviews = load_reviews()

    reviewed_tokens = set(
        reviews["scene_token"]
        .astype(str)
        .tolist()
    )

    current_name = (
        st.session_state.selected_scene_name
    )

    current_index = scene_names.index(
        current_name
    )

    # Search forward first.
    for i in range(
        current_index + 1,
        len(scenes)
    ):

        if scenes[i]["token"] not in reviewed_tokens:

            st.session_state.selected_scene_name = (
                scenes[i]["name"]
            )

            return

    # Wrap around to the beginning.
    for i in range(0, current_index):

        if scenes[i]["token"] not in reviewed_tokens:

            st.session_state.selected_scene_name = (
                scenes[i]["name"]
            )

            return


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Navigation")

    nav1, nav2 = st.columns(2)

    nav1.button(
        "← Scene",
        on_click=move_scene,
        args=(-1,),
        # width="stretch",
    )

    nav2.button(
        "Scene →",
        on_click=move_scene,
        args=(1,),
        # width="stretch",
    )

    st.selectbox(
        "Scene",
        scene_names,
        key="selected_scene_name",
    )

    st.button(
        "Next unreviewed",
        on_click=move_to_next_unreviewed,
        # width="stretch",
    )

    st.divider()

    reviews = load_reviews()

    if reviews.empty:
        kept = 0
        skipped = 0
    else:
        kept = int(
            (reviews["decision"] == "keep").sum()
        )

        skipped = int(
            (reviews["decision"] == "skip").sum()
        )

    st.metric(
        "Total scenes",
        len(scenes)
    )

    st.metric(
        "Reviewed",
        len(reviews)
    )

    st.metric(
        "Kept",
        kept
    )

    st.metric(
        "Skipped",
        skipped
    )

    st.divider()

    st.caption("Database")
    st.code(str(DB_PATH))

    st.caption("Review manifest")
    st.code(str(REVIEW_CSV))


# ============================================================
# CURRENT SCENE
# ============================================================

scene = scene_lookup[
    st.session_state.selected_scene_name
]

scene_index = scene_names.index(
    scene["name"]
)

review = existing_review(
    scene["token"]
)


st.header(
    f"{scene['name']} "
    f"— {scene_index + 1}/{len(scenes)}"
)


info1, info2, info3 = st.columns(3)

info1.metric(
    "Keyframes",
    scene["nbr_samples"]
)

info2.metric(
    "Location",
    scene["location"]
)

if review is None:
    status = "Not reviewed"
else:
    status = str(
        review["decision"]
    ).upper()

info3.metric(
    "Status",
    status
)


description = scene["description"]

if description:
    st.write(
        f"**Description:** {description}"
    )


# ============================================================
# LOAD CURRENT SCENE ONLY
# ============================================================

samples = load_scene_samples(
    scene["token"]
)

frames = load_scene_keyframes(
    scene["token"]
)


if not samples:

    st.error(
        "No samples found for this scene."
    )

    st.stop()


# ============================================================
# SAMPLE NAVIGATION
# ============================================================

sample_state_key = (
    f"sample_index_{scene['token']}"
)

if sample_state_key not in st.session_state:

    # Start in the middle because it is often more
    # representative than the first frame.
    st.session_state[sample_state_key] = (
        len(samples) // 2
    )


sample_index = st.slider(
    "Position within scene",
    min_value=0,
    max_value=len(samples) - 1,
    key=sample_state_key,
)


sample = samples[
    sample_index
]


st.caption(
    f"Keyframe {sample_index + 1} "
    f"of {len(samples)}"
)


# ============================================================
# IMAGE DISPLAY + REVIEW PANEL
# ============================================================

sample_frames = frames.get(
    sample["token"],
    {}
)

cam_front_filename = ""

front_frame = sample_frames.get(
    "CAM_FRONT"
)

if front_frame is not None:
    cam_front_filename = front_frame["filename"]


# ------------------------------------------------------------
# Keys used for tags / notes
# ------------------------------------------------------------

tags_key = f"tags_{scene['token']}"
notes_key = f"notes_{scene['token']}"


existing_tags = ""
existing_notes = ""

if review is not None:

    if pd.notna(review["tags"]):
        existing_tags = str(review["tags"])

    if pd.notna(review["notes"]):
        existing_notes = str(review["notes"])


if tags_key not in st.session_state:
    st.session_state[tags_key] = existing_tags

if notes_key not in st.session_state:
    st.session_state[notes_key] = existing_notes


# ------------------------------------------------------------
# Confirmation from previous scene
# ------------------------------------------------------------

if "review_message" in st.session_state:

    st.success(
        st.session_state.pop("review_message")
    )


# ------------------------------------------------------------
# Main layout
#
# Left  = six cameras
# Right = scene review controls
# ------------------------------------------------------------

image_area, review_area = st.columns(
    [4, 1]
)


# ============================================================
# LEFT: CAMERA IMAGES
# ============================================================

with image_area:

    st.subheader("Camera views")

    for camera_row in CAMERA_LAYOUT:

        columns = st.columns(3)

        for column, camera in zip(
            columns,
            camera_row
        ):

            with column:

                st.markdown(
                    f"**{camera}**"
                )

                frame = sample_frames.get(
                    camera
                )

                if frame is None:

                    st.warning(
                        "No keyframe found"
                    )

                    continue

                image_path = (
                    DATAROOT
                    / frame["filename"]
                )

                if image_path.exists():

                    st.image(
                        str(image_path)
                    )

                else:

                    st.error(
                        "Image file missing"
                    )

                    st.code(
                        str(image_path)
                    )


# ============================================================
# RIGHT: REVIEW PANEL
# ============================================================

with review_area:

    st.subheader("Review")

    if review is None:

        st.caption(
            "Not reviewed"
        )

    else:

        st.caption(
            f"Current: "
            f"{str(review['decision']).upper()}"
        )


    st.text_input(
        "Tags",
        key=tags_key,
        placeholder=(
            "pedestrian, intersection, "
            "night, rain"
        ),
    )


    st.text_area(
        "Notes",
        key=notes_key,
        placeholder=(
            "Why is this scene useful?"
        ),
        height=150,
    )


    st.button(
        "✓ KEEP SCENE",
        type="primary",
        on_click=review_and_next,
        args=(
            "keep",
            scene,
            sample,
            cam_front_filename,
            tags_key,
            notes_key,
        ),
    )


    st.button(
        "✗ SKIP SCENE",
        on_click=review_and_next,
        args=(
            "skip",
            scene,
            sample,
            cam_front_filename,
            tags_key,
            notes_key,
        ),
    )


# ============================================================
# SAMPLE DETAILS
# ============================================================

with st.expander(
    "Current sample details"
):

    st.write("**Sample token**")

    st.code(
        sample["token"]
    )

    st.write("**Timestamp**")

    st.code(
        str(sample["timestamp"])
    )

    for camera in ALL_CAMERAS:

        frame = sample_frames.get(
            camera
        )

        if frame is None:
            continue

        st.markdown(
            f"**{camera}**"
        )

        st.code(
            frame["filename"]
        )

        st.caption(
            "sample_data token: "
            + frame["token"]
        )


# ============================================================
# REVIEW TABLE
# ============================================================

st.divider()

st.subheader("Reviewed scenes")

reviews = load_reviews()


if reviews.empty:

    st.info(
        "No scenes reviewed yet."
    )

else:

    st.dataframe(
        reviews[
            [
                "scene_name",
                "decision",
                "tags",
                "location",
                "representative_sample_token",
                "notes",
            ]
        ],
        hide_index=True,
        # width="stretch",
    )

    st.download_button(
        "Download review CSV",
        data=reviews.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="nuscenes_scene_reviews.csv",
        mime="text/csv",
    )