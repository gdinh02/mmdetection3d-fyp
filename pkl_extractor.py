import os
import mmengine


def extract_single_camera_sample(
    source_info_file,
    image_path,
    output_info_file,
    cam_type="CAM_FRONT",
):
    info = mmengine.load(source_info_file)

    target_name = os.path.basename(image_path)

    matches = []

    for sample in info["data_list"]:

        images = sample.get("images", {})

        if cam_type not in images:
            continue

        stored_path = images[cam_type]["img_path"]

        if os.path.basename(stored_path) == target_name:
            matches.append(sample)

    if len(matches) == 0:
        raise RuntimeError(
            f"Could not find {target_name} "
            f"under camera {cam_type}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Found {len(matches)} matching samples."
        )

    single_info = {
        "metainfo": info.get("metainfo", {}),
        "data_list": [matches[0]],
    }

    mmengine.dump(
        single_info,
        output_info_file,
    )

    print(
        f"Saved single-sample info to "
        f"{output_info_file}"
    )


extract_single_camera_sample(
    source_info_file="data/nuscenes/nuscenes_infos_val.pkl",
    image_path="demo/data/nuscenes/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.jpg",
    output_info_file="demo/data/nuscenes/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.pkl",
    cam_type="CAM_FRONT",
)