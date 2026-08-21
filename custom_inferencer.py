import torch
import numpy as np
import mmcv
import mmengine
import os.path as osp

from mmdet3d.apis import MonoDet3DInferencer
from mmdet3d.apis.inferencers.mono_det3d_inferencer import InputsType, PredType
from mmengine.fileio import get_file_backend, isdir, join_path, list_dir_or_file
                 

from typing import List, Union

class MonoDet3DInferencerWithFilter(MonoDet3DInferencer):
    # Add class_filter arg to natively filter classes 
    forward_kwargs = MonoDet3DInferencer.forward_kwargs | {'class_filter'}

    # Override for class filtering
    def forward(self, inputs, class_filter = None, **kwargs):
        preds = super().forward(inputs, **kwargs)

        # print(preds)

        if class_filter:
            for pred in preds:
                if 'pred_instances_3d' in pred:
                    labels = pred.pred_instances_3d.labels_3d
                    mask = torch.isin(
                        labels, 
                        torch.tensor(class_filter, device=labels.device)
                    )
                    pred.pred_instances_3d = pred.pred_instances_3d[mask]

        # print(preds)

        return preds

    # Override to fix no image saved with no pred
    def visualize(self,
                    inputs: InputsType,
                    preds: PredType,
                    return_vis: bool = False,
                    show: bool = False,
                    wait_time: int = 0,
                    draw_pred: bool = True,
                    pred_score_thr: float = 0.3,
                    no_save_vis: bool = False,
                    img_out_dir: str = '',
                    cam_type_dir: str = 'CAM2') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Dict]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to save visualization results.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.
            cam_type_dir (str): Camera type directory. Defaults to 'CAM2'.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                                'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input['img'], str):
                img_bytes = mmengine.fileio.get(single_input['img'])
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input['img'])
            elif isinstance(single_input['img'], np.ndarray):
                img = single_input['img'].copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                    f"{type(single_input['img'])}")

            out_file = osp.join(img_out_dir, 'vis_camera', cam_type_dir,
                                img_name) if img_out_dir != '' else None

            data_input = dict(img=img)
            self.visualizer.add_datasample(
                img_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
                vis_task='mono_det',
            )

            # Save the image even with no prediction made
            if out_file is not None and not osp.exists(out_file):
                mmengine.mkdir_or_exist(osp.dirname(out_file))
                mmcv.imwrite(img[:,:,::-1], out_file)

            results.append(img)
            self.num_visualized_imgs += 1

        return results

    # Handle image list and single pkl file
    def _inputs_to_list(self,
                        inputs: Union[dict, list],
                        cam_type='CAM2',
                        **kwargs) -> list:
        """Preprocess the inputs to a list."""
        if isinstance(inputs, dict):
            assert 'infos' in inputs
            infos = inputs.pop('infos')

            if isinstance(inputs['img'], str):
                img = inputs['img']
                backend = get_file_backend(img)
                if hasattr(backend, 'isdir') and isdir(img):
                    filename_list = list_dir_or_file(img, list_dir=False)
                    inputs = [{
                        'img': join_path(img, filename)
                    } for filename in filename_list]

            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            # Load the pkl once
            info_list = mmengine.load(infos)['data_list']
            
            # --- FIX: Create an O(1) lookup dictionary matching filenames to their data ---
            info_dict = {
                osp.basename(info['images'][cam_type]['img_path']): info 
                for info in info_list
            }

            for input in inputs:
                img_name = osp.basename(input['img'])
                if img_name not in info_dict:
                    raise ValueError(f'The info file for {img_name} is not provided in the .pkl.')
                
                data_info = info_dict[img_name]
                
                cam2img = np.asarray(
                    data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam[0:3]
                                        
                input['cam2img'] = cam2img
                input['lidar2cam'] = lidar2cam
                input['lidar2img'] = lidar2img

        elif isinstance(inputs, (list, tuple)):
            for input in inputs:
                assert 'infos' in input
                infos = input.pop('infos')
                
                info_list = mmengine.load(infos)['data_list']
                
                # --- FIX: Create an O(1) lookup dictionary for the list branch as well ---
                info_dict = {
                    osp.basename(info['images'][cam_type]['img_path']): info 
                    for info in info_list
                }
                
                img_name = osp.basename(input['img'])
                if img_name not in info_dict:
                    raise ValueError(f'The info file for {img_name} is not provided in the .pkl.')
                
                data_info = info_dict[img_name]
                
                cam2img = np.asarray(
                    data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                    
                input['cam2img'] = cam2img
                input['lidar2cam'] = lidar2cam
                input['lidar2img'] = lidar2img

        return list(inputs)
                