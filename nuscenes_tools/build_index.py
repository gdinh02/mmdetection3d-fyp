import json
import sqlite3
from pathlib import Path

import ijson


DATAROOT = Path("/mnt/z/nuscenes")
META = DATAROOT / "v1.0-trainval"
DB_PATH = Path("nuscenes_camera.db")


def load_json(name):
    with open(META / name, "r") as f:
        return json.load(f)


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS scenes;
DROP TABLE IF EXISTS samples;
DROP TABLE IF EXISTS camera_frames;
DROP TABLE IF EXISTS logs;

CREATE TABLE scenes (
    token TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    log_token TEXT,
    first_sample_token TEXT,
    last_sample_token TEXT,
    nbr_samples INTEGER
);

CREATE TABLE samples (
    token TEXT PRIMARY KEY,
    scene_token TEXT,
    timestamp INTEGER,
    prev TEXT,
    next TEXT
);

CREATE TABLE camera_frames (
    token TEXT PRIMARY KEY,
    sample_token TEXT,
    channel TEXT,
    filename TEXT,
    timestamp INTEGER,
    is_key_frame INTEGER,
    prev TEXT,
    next TEXT
);

CREATE TABLE logs (
    token TEXT PRIMARY KEY,
    location TEXT
);

CREATE INDEX idx_samples_scene
ON samples(scene_token, timestamp);

CREATE INDEX idx_frames_sample
ON camera_frames(sample_token);

CREATE INDEX idx_frames_channel
ON camera_frames(channel);

CREATE INDEX idx_frames_timestamp
ON camera_frames(timestamp);
""")


# ------------------------------------------------------------
# Small metadata tables
# ------------------------------------------------------------

print("Loading sensors...")

sensors = load_json("sensor.json")

sensor_by_token = {
    s["token"]: s
    for s in sensors
}


calibrated = load_json(
    "calibrated_sensor.json"
)

calibration_to_channel = {}

for cal in calibrated:
    sensor = sensor_by_token[
        cal["sensor_token"]
    ]

    if sensor["modality"] == "camera":
        calibration_to_channel[
            cal["token"]
        ] = sensor["channel"]


# ------------------------------------------------------------
# Scenes
# ------------------------------------------------------------

print("Indexing scenes...")

scenes = load_json("scene.json")

cur.executemany(
    """
    INSERT INTO scenes VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    [
        (
            s["token"],
            s["name"],
            s.get("description", ""),
            s.get("log_token", ""),
            s["first_sample_token"],
            s["last_sample_token"],
            s["nbr_samples"],
        )
        for s in scenes
    ]
)


# ------------------------------------------------------------
# Logs
# ------------------------------------------------------------

log_path = META / "log.json"

if log_path.exists():
    print("Indexing logs...")

    logs = load_json("log.json")

    cur.executemany(
        "INSERT INTO logs VALUES (?, ?)",
        [
            (
                log["token"],
                log.get("location", "unknown"),
            )
            for log in logs
        ]
    )


conn.commit()


# ------------------------------------------------------------
# Samples - streamed rather than loaded into memory
# ------------------------------------------------------------

print("Indexing samples...")

batch = []

with open(META / "sample.json", "rb") as f:

    for sample in ijson.items(f, "item"):

        batch.append(
            (
                sample["token"],
                sample["scene_token"],
                sample["timestamp"],
                sample.get("prev", ""),
                sample.get("next", ""),
            )
        )

        if len(batch) >= 5000:
            cur.executemany(
                """
                INSERT INTO samples
                VALUES (?, ?, ?, ?, ?)
                """,
                batch,
            )

            conn.commit()
            batch.clear()


if batch:
    cur.executemany(
        """
        INSERT INTO samples
        VALUES (?, ?, ?, ?, ?)
        """,
        batch,
    )

    conn.commit()


# ------------------------------------------------------------
# Camera sample_data
#
# Includes BOTH:
#   samples/
#   sweeps/
#
# Radar/LiDAR records are ignored.
# ------------------------------------------------------------

print("Indexing camera frames...")
print("This is the longest step.")

batch = []
count = 0

with open(
    META / "sample_data.json",
    "rb",
) as f:

    for sd in ijson.items(f, "item"):

        channel = calibration_to_channel.get(
            sd["calibrated_sensor_token"]
        )

        if channel is None:
            continue

        batch.append(
            (
                sd["token"],
                sd["sample_token"],
                channel,
                sd["filename"],
                sd["timestamp"],
                int(sd["is_key_frame"]),
                sd.get("prev", ""),
                sd.get("next", ""),
            )
        )

        count += 1

        if len(batch) >= 5000:

            cur.executemany(
                """
                INSERT INTO camera_frames
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )

            conn.commit()
            batch.clear()

            print(
                f"\rCamera frames indexed: {count:,}",
                end="",
                flush=True,
            )


if batch:
    cur.executemany(
        """
        INSERT INTO camera_frames
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )

conn.commit()
conn.close()

print()
print(f"Done.")
print(f"Database: {DB_PATH.resolve()}")
print(f"Camera frames indexed: {count:,}")