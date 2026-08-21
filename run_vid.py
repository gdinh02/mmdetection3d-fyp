import os
from mmengine.fileio import load, dump

from bevconvert import update_visualization, setup_visualization, generate_binary_bev_map, align_multi_frame_history, smooth_binary_map
from custom_inferencer import MonoDet3DInferencerWithFilter
import matplotlib.pyplot as plt
import cv2
from collections import deque
import numpy as np

MODEL_FILE = "configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
WEIGHTS_FILE = "checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"
DEVICE = "cuda:0"

DATASET_ROOT = "./data/nuscenes"
INFO_FILE = f"{DATASET_ROOT}/nuscenes_infos_val.pkl"
IMAGE_DIR = f"{DATASET_ROOT}/samples"
OUT_DIR = 'outputs/samples_inference'
TEMP_INFO_FILE = 'temp_single_sample_info.pkl' # Temporary file to trick the inferencer

PRED_SCORE_THR = 0.25


def filter_predictions(predictions, threshold, allowed_classes):
    """Filters 3D object detection predictions based on a score threshold.

    Args:
        predictions (list of dict): The list of prediction dictionaries.
        threshold (float): The minimum score required to keep a bounding box.

    Returns:
        list of dict: A new list of predictions with filtered boxes, scores, and
        labels.
    """
    filtered_predictions = []

    for pred in predictions:
        boxes = pred.get('bboxes_3d', [])
        scores = pred.get('scores_3d', [])
        labels = pred.get('labels_3d', [])

        # Ensure all lists match in length before zipping
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError(
                'Mismatch in lengths of bboxes_3d, scores_3d, and labels_3d.'
            )

        # Filter items where score >= threshold
        filtered_items = [
            (box, score, label)
            for box, score, label in zip(boxes, scores, labels)
            if score >= threshold and label in allowed_classes
        ]


        # Unpack filtered results back into lists (handles empty cases gracefully)
        if filtered_items:
            f_boxes, f_scores, f_labels = zip(*filtered_items)
        else:
            f_boxes, f_scores, f_labels = [], [], []

        # Construct the filtered prediction dictionary
        filtered_pred = {
            'bboxes_3d': list(f_boxes),
            'scores_3d': list(f_scores),
            'labels_3d': list(f_labels),
            'box_type_3d': pred.get('box_type_3d', 'Camera'),
        }

        filtered_predictions.append(filtered_pred)

    return filtered_predictions

def main():
    print("Initializing Inferencer...")
    inferencer = MonoDet3DInferencerWithFilter(model=MODEL_FILE, weights=WEIGHTS_FILE, device=DEVICE)
    inferencer.forward

    print("Loading massive NuScenes info file...")
    nuscenes_info = load(INFO_FILE)
    
    # We need this to keep the dataset metainfo (classes, etc.) intact
    metainfo = nuscenes_info.get('metainfo', {})

    fig, ax_img, ax_bev = setup_visualization()

    history_buffer = deque(maxlen=3)

    target_weights = [0.4, 0.3, 0.2, 0.1]

    prev_blended_map = None
    alpha = 0.5

    for sample_idx, sample in enumerate(nuscenes_info['data_list']):

        for cam_type, cam_info in sample['images'].items():
            if cam_type != "CAM_FRONT":
                continue

            img_rel_path = cam_info['img_path']
            img_full_path = os.path.join(IMAGE_DIR, cam_type, img_rel_path)

            print(f"Processing Sample {sample_idx} - {cam_type}")

            # 3. Pass the single image and the single-sample temporary pkl
            inputs = dict(
                img=img_full_path,
                infos=INFO_FILE
            )

            # Because both inputs and the temp pkl have a length of 1, 
            # the assert len(info_list) == len(inputs) will pass!
            result = inferencer(
                inputs=inputs,
                cam_type=cam_type,
                cam_type_dir=cam_type,
                out_dir=OUT_DIR,
                # show=True,
                print_result=False,
                wait_time=0.01,
                pred_score_thr=PRED_SCORE_THR,
                class_filter=[0]
            )

            break # Stop at chosen camera

        # filtered_preds = filter_predictions(result['predictions'], PRED_SCORE_THR, [0, 1, 2, 3, 4, 8 ,9])

        curr_binary_map = generate_binary_bev_map(
            result['predictions'][0]['bboxes_3d'], 
            result['predictions'][0]['scores_3d'], 
            score_thresh=0.2
        )

        curr_binary_map = smooth_binary_map(
            bev_map=curr_binary_map,
            buffer_meters=0.3
        )    

        if len(history_buffer) > 0:
            # Slice weights based on how many frames are actually in the buffer
            # (e.g., on frame 2, we only have Current and t-1)
            active_weights = target_weights[:len(history_buffer) + 1]
            
            # Normalize the weights so they always sum exactly to 1.0
            weight_sum = sum(active_weights)
            active_weights = [w / weight_sum for w in active_weights]
            
            # Blend the current map with the historical maps
            blended_map = align_multi_frame_history(curr_binary_map, history_buffer, active_weights)
        else:
            # First frame has no history
            blended_map = curr_binary_map.copy()

        history_buffer.appendleft(curr_binary_map)

        threshold = 0.1
        _, final_map = cv2.threshold(blended_map, threshold, 1.0, cv2.THRESH_TOZERO)
        final_map = np.where(final_map > threshold, 1, 0).astype(np.uint8)

        ax_bev.imshow(final_map)

        update_visualization(
            fig,
            ax_img,
            ax_bev,
            os.path.join(OUT_DIR, "vis_camera", cam_type, img_rel_path),
            # img_full_path,
            result['predictions'][0]['bboxes_3d'],
            result['predictions'][0]['scores_3d'],
            result['predictions'][0]['labels_3d'],
            blended_map=final_map,      # FIX: Pass the map directly into the visualizer
            score_thresh=PRED_SCORE_THR,
            timeout=0.1
        )

        # prev_blended_map = blended_map

        # if sample_idx == 64:
        #     update_visualization(
        #         fig,
        #         ax_img,
        #         ax_bev,
        #         os.path.join(OUT_DIR, "vis_camera", "CAM2", img_rel_path),
        #         filtered_preds[0]['bboxes_3d'],
        #         filtered_preds[0]['scores_3d'],
        #         filtered_preds[0]['labels_3d'],
        #         score_thresh=PRED_SCORE_THR,
        #         timeout=0.1
        #     )

    plt.ioff()
    plt.show()

    if os.path.exists(TEMP_INFO_FILE):
        os.remove(TEMP_INFO_FILE)
        
    print(f"Finished Inference. Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()