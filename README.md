<h2 align="center">LaFUA: Exploiting Dense Imaging Feature for White Matter Hyperintensity Segmentation</h2>

<p align="center">
  <b>Anonymous Authors</b>
</p>

# Installation
Make sure to install the latest [**nnUNet-v2**](https://github.com/MIC-DKFZ/nnUNet) and [**DINOv3**](https://github.com/facebookresearch/dinov3).

### Setup LaFUA
To configure **LaFUA**, download [``LaFUA_Trainer.py``](./LaFUA_Trainer.py) to ``nnunetv2/training/nnUNetTrainer/LaFUA_Trainer.py``.

### Run Plan and Preprocessing

Run the default nnU-Net plan and preprocessing.

```python
import os

working_dir = 'path/to/working/directory'
os.environ['nnUNet_raw'] = os.path.join(working_dir, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(working_dir, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(working_dir, 'nnUNet_results')

dataset_id = 1  # your dataset id
plan = 'nnUNetResEncUNetLPlans'
!nnUNetv2_plan_and_preprocess -c 3d_fullres -d "{dataset_id}" -pl "{plan}" --verbose
```

# Training with LaFUA

### Configure LFA Backbone

Select the LFA backbone to run the LaFUA training framework. Possible models are:
<table style="margin: auto">
  <thead>
    <tr>
      <th>LFA Backbone</th>
      <th>Trainer Class</th>
      <th>Trainer Class (Depth Padding Enabled)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/16 distilled </td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTS16</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTS16_PadDepth</td>
    </tr>
    <tr>
      <td>ViT-S+/16 distilled</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTS16Plus</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTS16Plus_PadDepth</td>
    </tr>
    <tr>
      <td>ViT-B/16 distilled</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTB16</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTB16_PadDepth</td>
    </tr>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTL16</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTL16_PadDepth</td>
    </tr>
    <tr>
      <td>ViT-L+/16 distilled</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTL16Plus</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTL16Plus_PadDepth</td>
    </tr>
    <tr>
      <td>ViT-H+/16 distilled</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTH16Plus</td>
      <td align="center">LaFUA_Trainer_DINOv3_ViTH16Plus_PadDepth</td>
    </tr>
  </tbody>
</table>

Depth padding requires a sufficiently large input shape (e.g., all input axes must be at least 64 when using 6 stages).

If your input is smaller, use the standard trainer class without depth padding.

Make sure to prepare DINOv3 model checkpoints (e.g., `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` for ViT-S/16 distilled).

### Modify nnUNet Plans

To modify plans (e.g., batch size, input shape, or number of stages), modify `nnUNet_preprocessed/DatasetXXX_YourDataset/nnUNetResEncUNetLPlans.json`:

When adjusting `n_stages` to `N`, ensure that `features_per_stage`, `kernel_sizes`, `strides`, and `n_blocks_per_stage` to have `N` elements, and `n_conv_per_stage_decoder` to have `N-1` elements.

```json
{
    "configurations": {
        "3d_fullres": {
            "batch_size": 4,
            "patch_size": [
                32,
                128,
                128
            ],
            "architecture": {
                "network_class_name": "...",
                "arch_kwargs": {
                    "n_stages": 6,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320
                    ],
                    "kernel_sizes": "...",
                    "strides": "...",
                    "n_blocks_per_stage": "...",
                    "n_conv_per_stage_decoder": "..."
                }
            }
        }
    }
}
```

# Training example

Run the following script to execute the 5-fold cross-validation using ViT-S/16 LFA backbone:

```python
import os

working_dir = 'path/to/working/directory'
os.environ['nnUNet_raw'] = os.path.join(working_dir, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(working_dir, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(working_dir, 'nnUNet_results')
os.environ['LFA_Backbone_Repo'] = os.path.join(working_dir, 'dinov3')
os.environ['LFA_Backbone_Checkpoint'] = os.path.join(working_dir, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')

dataset_id = 1  # your dataset id
plan = 'nnUNetResEncUNetLPlans'
trainer = 'LaFUA_Trainer_DINOv3_ViTS16'
for fold in (0, 1, 2, 3, 4):
    !CUDA_VISIBLE_DEVICES=0 nnUNetv2_train "{dataset_id}" -c 3d_fullres -f "{fold}" -p "{plan}" -tr "{trainer}" --c
```

# Inference example

```python
import os

working_dir = 'path/to/working/directory'
os.environ['nnUNet_raw'] = os.path.join(working_dir, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(working_dir, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(working_dir, 'nnUNet_results')
os.environ['LFA_Backbone_Repo'] = os.path.join(working_dir, 'dinov3')
os.environ['LFA_Backbone_Checkpoint'] = os.path.join(working_dir, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')

dataset_id = 1  # your dataset id
plan = "nnUNetResEncUNetLPlans"
trainer = 'LaFUA_Trainer_DINOv3_ViTS16'
input_path = 'path/to/test/set'
output_path = 'path/to/output/path'
for fold in (0, 1, 2, 3, 4):
    !CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d "{dataset_id}" -c 3d_fullres -f "{fold}" -i "{input_path}" -o "{output_path}" -p "{plan}" -tr "{trainer}"
```

# Acknowledgement

Part of the codes are referred from the following open-source projects:

* https://github.com/MIC-DKFZ/nnUNet
* https://github.com/facebookresearch/dinov3/
* https://github.com/wimmerth/anyup/
* https://github.com/junmokwon/AgeCMBNet
