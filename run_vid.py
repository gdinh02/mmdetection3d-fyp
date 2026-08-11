import os
import mmengine
from mmengine.fileio import load, dump
from mmdet3d.apis import MonoDet3DInferencer

MODEL_FILE = "configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
WEIGHTS_FILE = "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"
DEVICE = "cuda:0"

DATASET_ROOT = "./data/nuscenes"
INFO_FILE = f"{DATASET_ROOT}/nuscenes_infos_val.pkl"
IMAGE_DIR = f"{DATASET_ROOT}/samples"
OUT_DIR = 'outputs/samples_inference'
TEMP_INFO_FILE = 'temp_single_sample_info.pkl' # Temporary file to trick the inferencer

def main():
    print("Initializing Inferencer...")
    inferencer = MonoDet3DInferencer(model=MODEL_FILE, weights=WEIGHTS_FILE, device=DEVICE)

    print("Loading massive NuScenes info file...")
    nuscenes_info = load(INFO_FILE)
    
    # We need this to keep the dataset metainfo (classes, etc.) intact
    metainfo = nuscenes_info.get('metainfo', {})

    for sample_idx, sample in enumerate(nuscenes_info['data_list']):
        
        single_sample_dict = {
            'metainfo': metainfo,
            'data_list': [sample] # Length is exactly 1!
        }
        
        dump(single_sample_dict, TEMP_INFO_FILE)

        for cam_type, cam_info in sample['images'].items():
            if cam_type != "CAM_FRONT":
                continue

            img_rel_path = cam_info['img_path']
            img_full_path = os.path.join(IMAGE_DIR, cam_type, img_rel_path)

            print(f"Processing Sample {sample_idx} - {cam_type}")

            # 3. Pass the single image and the single-sample temporary pkl
            inputs = dict(
                img=img_full_path,
                infos=TEMP_INFO_FILE 
            )

            # Because both inputs and the temp pkl have a length of 1, 
            # the assert len(info_list) == len(inputs) will pass!
            result = inferencer(
                inputs=inputs,
                cam_type=cam_type,
                # out_dir=OUT_DIR,
                show=True,
                print_result=False,
                wait_time=0.01,
                pred_score_thr=0.3
            )

        # if sample_idx == 1:
        #     print(result['predictions'].keys())
        #     break
    if os.path.exists(TEMP_INFO_FILE):
        os.remove(TEMP_INFO_FILE)
        
    print(f"Finished Inference. Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()