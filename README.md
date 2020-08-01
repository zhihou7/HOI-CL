# Visual Compositional Learning for Human-Object Interaction Detection.

Official TensorFlow implementation for VCL ([Visual Compositional Learning for Human-Object Interaction Detection](https://arxiv.org/abs/2007.12407)) in ECCV2020

Code is waiting for test

Welcome to create issues if you have any questions. 


<img src='misc/imagine.png'>

## Prerequisites

This codebase was developed and tested with Python3.7, Tensorflow 1.14.0 CUDA 10.0 and Ubuntu 18.04.


## Installation
1. Clone the repository. 
    ```Shell
    git clone https://github.com/zhihou7/VCL.git
    ```
2. Download V-COCO and HICO-DET dataset. Setup V-COCO and COCO API. Setup HICO-DET evaluation code.
    ```Shell
    chmod +x ./misc/download_dataset.sh 
    ./misc/download_dataset.sh 
    
    # Assume you cloned the repository to `VCL_DIR'.
    # If you have downloaded V-COCO or HICO-DET dataset somewhere else, you can create a symlink
    # ln -s /path/to/your/v-coco/folder Data/
    # ln -s /path/to/your/hico-det/folder Data/
    ```

## Training
1. Download COCO pre-trained weights and training data
    ```Shell
    chmod +x ./misc/download_training_data.sh 
    ./misc/download_training_data.sh
    ```
2. Train an VCL on V-COCO
    ```Shell
    python tools/Train_VCL_ResNet_VCOCO.py
    ```
3. Train an VCL on HICO-DET
    ```Shell
    python tools/Train_VCL_ResNet_HICO.py --num_iteration 800000
    ```
    
4. Train an VCL for rare first zero-shot on HICO-DET
    ```Shell
    python tools/Train_VCL_ResNet_HICO.py --model VCL_union_multi_zs3_def1_l2_ml5_rew51_aug5_3_x5new --num_iteration 600000
    ```
  
5. Train an VCL for non-rare first zero-shot on HICO-DET
    ```Shell
    python tools/Train_VCL_ResNet_HICO.py --model VCL_union_multi_zs4_def1_l2_ml5_rew51_aug5_3_x5new --num_iteration 400000
    ```

### Explanation
Here, we design to add the strategies according to model name. 
For example, in *VCL_union_multi_zs3_def1_l2_ml5_rew51_aug5_3_x5new_res101*, 
*zs3* means the type of zero-shot, *ml5* is the hyper-parameter for composite branch, 
*rew* means we use the re-weighting strategy. If you do not use re-weighting, you can remove this in the model name. 
*aug5_3_x5new* means we set multiple interactions in each batch 
and the negative and positive samples partition for spatial-human and verb-object branch. *res101* is the backbone while default is res50
model *VCL_union_multi_base_zs3_def1_l2_ml5_rew51_aug5_3_x5new_res101* is our baseline for the corresponding model 

## Testing
1. Test an VCL on V-COCO
    ```Shell
     python tools/Test_ResNet_VCOCO.py --num_iteration 200000
    ```
3. Test an VCL on HICO-DET
    ```Shell
    python tools/Test_VCL_ResNet_HICO.py --num_iteration 800000
    ```
 
    or 
    ```Shell
   python scripts/full_test.py --model VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101 --num_iteration 800000
    ```

## TODO

- [x] Code & Data

- [ ] Model

- [ ] Test

## Q&A
1. ***The importance of re-weighting strategy.*** 
We follow previous work to use re-weighting. 
It multiplies the weights to the logits before the sigmoid function. 
We empirically find this is important for rare and unseen HOI detection

2. Res101 Detector. The Resnet-101 Detector is fully based on faster-rcnn ([detectron2](https://github.com/facebookresearch/detectron2)).
We fine-tune the [R101-RPN](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml) detector (pretrained on coco) on HICO-DET. We'll release the object detection result and the model. 

## Acknowledgement
Codes are built upon [iCAN: Instance-Centric Attention Network 
for Human-Object Interaction Detection](https://arxiv.org/abs/1808.10437), [Transferable Interactiveness Network](https://arxiv.org/abs/1811.08264), [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn).
