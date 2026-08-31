from pathlib import Path

import numpy as np


OUTPUT_FILE = Path("live_camera_calibration.npz")

# Replace with the actual live-camera calibration.
FX = None
FY = None
CX = None
CY = None
CAM2EGO = np.eye(4, dtype=np.float64)


def main():
    if None in (FX, FY, CX, CY):
        raise ValueError(
            "Set FX, FY, CX and CY to the actual calibrated camera values first."
        )

    cam2img = np.array(
        [
            [float(FX), 0.0, float(CX)],
            [0.0, float(FY), float(CY)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if np.asarray(CAM2EGO).shape != (4, 4):
        raise ValueError("CAM2EGO must have shape (4, 4)")

    np.savez(
        OUTPUT_FILE,
        cam2img=cam2img,
        cam2ego=np.asarray(CAM2EGO, dtype=np.float64),
    )
    print(f"Saved live calibration to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
