# Visual Compositional Learning for Human-Object Interaction Detection.

Official TensorFlow implementation for VCL ([Visual Compositional Learning for Human-Object Interaction Detection](https://arxiv.org/abs/2007.12407)) in ECCV2020

Welcome to create issues if you have any questions. 

[![Visual Compositional Learning for Human-Object Interaction Detection](https://res.cloudinary.com/marcomontalbano/image/upload/v1598938384/video_to_markdown/images/youtube--_JU5RnxnGxE-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/_JU5RnxnGxE "Visual Compositional Learning for Human-Object Interaction Detection")


## Citation
If you find our work useful in your research, please consider citing:
```
@article{hou2020visual,
  title={Visual Compositional Learning for Human-Object Interaction Detection},
  author={Hou, Zhi and Peng, Xiaojiang and Qiao, Yu and Tao, Dacheng},
  journal={arXiv preprint arXiv:2007.12407},
  year={2020}
}
```

## Prerequisites

This codebase was developed and tested with Python3.7, Tensorflow 1.14.0, Octave/Matlab (for evaluation), CUDA 10.0 and Ubuntu 18.04.


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

2. Train an VCL on V-COCO
    ```Shell
    python tools/Train_VCL_ResNet_VCOCO.py --model VCL_union_multi_ml1_l05_t3_rew_aug5_3_new_VCOCO_test --num_iteration 400000
    ```

### Model Parameters
Our model will converge at around iteration 500000 in HICO-DET. V-COCO will converge after 200000 iterations. We provide the model parameters that we trained as follows,

V-COCO: https://drive.google.com/file/d/1SzzMw6fS6fifZkpuar3B40dIl7YLNoYF/view?usp=sharing. I test the result is 47.82. The baseline also decreases compared to the reported result. The model in my reported result is deleted by accident. Empirically, hyper-parameters $lambda_1$ affects V-COCO more apparently.

HICO: https://drive.google.com/file/d/16unS3joUleoYlweX0iFxlU2cxG8csTQf/view?usp=sharing

HICO(Res101): https://drive.google.com/file/d/1iiCywBR0gn6n5tPzOvOSmZw_abOmgg53/view?usp=sharing

### Rules in model name
Here, we design to add the strategies according to model name for convenience. 

We take the name "VCL_union_multi_zs3_def1_l2_ml5_rew51_aug5_3_x5new_res101" as example, 
    
- "union" means union box for the verb features. "_humans_" is for human box for verb features.
- "multi" has no meaning. It is to avoid running the model by Train_iCAN_ResNet_HICO.
- "zs3" means the type of zero-shot, "zs3" is rare-first selection. "zs4" is nonrare-first selection.
- "def1" is our composition strategy. "def1" is our strategy in the paper and is easily implemented.
- "l2" is the hyper-parameter $lambda_1$ for verb-object branch. It is 2. 
- "ml5" is the hyper-parameter $lambda_2$ for composite branch. See other choice for the two hyper-parameters.
- "rew" means we use the re-weighting strategy. If you do not use re-weighting, you can remove this in the model name. 
   **It is better evaluate the method without re-weighting because the weights should be changed when generating large examples.** 
   We just reduce value of weights and we think there might be weights for the composition branch ("rew2"). Baseline just uses "rew".
   "rew51" means we simply set the maximum value for unseen categories. See more analysis and experiments in our code.
   We also analyze the weights in our main paper based on the baseline model.
   **Current weights in the composition branch is simple, We think better weights can further improve the performance.** 
- "aug5" means we set multiple interactions in each batch. See supplement materials for the effect on the performance. 
   "aug5" means 5 HOIs in each batch. Due to the limitation of GPU memory, we do not try larger size.
- "x5new" means we also use a small number of negative samples for verb-object branch. 
   in iCAN, positive samples are for H-O branch while negative and positive samples for S-P branch. A small trick. This could improve about 0.2.
   This is a bit similar to some findings in No-Frills Human-Object Interaction Detection.
- "res101" is the backbone while default is res50.
- "base" is the model without composition branch. **model "VCL_union_multi_base_zs3_def1_l2_ml5_rew51_aug5_3_x5new_res101" is our baseline for the corresponding model** 
- "VCL_V_". If the model name starts with "VCL_V", the code will only use verb-object branch. **We think this is helpful to explore further composing HOIs.** 

**The rules of model name contain all ablation study in our main paper and supplementary materials.** 
Besides, we keep the code of pose information in the project, which can obtain a bit better performance (around 19.6%) than the reported results. 
In our paper, We do not use pose information.

## Testing
1. Test an VCL on V-COCO
    ```Shell
     python tools/Test_ResNet_VCOCO.py --num_iteration 200000
    ```
3. Test an VCL on HICO-DET
    ```Shell
    python tools/Test_VCL_ResNet_HICO.py --num_iteration 800000
   
    cd Data/ho-rcnn/;python ../../scripts/postprocess_test.py --model VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101 --num_iteration 3 --fuse_type spv
    ```
 
    or 
    ```Shell
   python scripts/full_test.py --model VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101 --num_iteration 800000
    ```

3. Illustration of verb and object features

   ```shell
   python scripts/extract_HO_feature.py --model VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101 --num_iteration 800000
   
   python scripts/tsne.py VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101
   ```
 
## Experiment Results

mAP on HICO-DET (Default)


|Model|Full|Rare|Non-Rare|
|:-|:-:|:-:|:-:|
|GPNN [1]|13.11|9.34|14.23|
|iCAN [2]|14.84|10.45|16.15|
|Xu et al.[3]| 14.70|13.26| 15.13|
|TIN [4]|17.22|13.51|18.32|
|Wang et al. [5]|16.24|11.16|17.75|
|No-Frills [6]|17.18|12.17|18.68|
|RPNN [7]| 17.35 | 12.78| 18.71|
|PMFNet [8]|17.46|15.65|18.00|
|Peyre et al. [9] | 19.40|14.63|20.87
|Baseline (ours) | 18.03 | 13.62 | 19.35
|VCL (ours) | **19.43** | **16.55** | 20.29 |
|VCL + pose (ours) | **19.70** | **16.68** | 20.60 |
|Bansal*et al.[10]  |  21.96 | 16.43 | 23.62 |
|VCL* (ours) |23.63 | 17.21 | 25.55 |
|VCL' (ours) |23.55 | 17.59 | 25.33 | 

* means using res101 backbone and fine-tune the object detector on HICO-DET. VCL' is the result of our resnet50 model under the fine-tuned detector. We have a strong baseline (18.03).
Baseline directly copy two important strategies (re-weighting and box postprocessing) from previous work (See Supplementary materials).
We also illustrates these in the code in detail. If finetuning our model, we can obtain better result (about 19.7) than 19.70. VCL + pose is corresponding to posesp in our code.

**References:**

[1] Qi, S., et al. Learning Human-Object Interactions by Graph Parsing Neural Networks. ECCV.

[2] Gao, C., et al. iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection. BMVC.

[3] Xu, B., et al Learning to detect human-object  interactions  with  knowledge.  CVPR (2019)
[4] Li, Y. et al. Transferable interactiveness knowledge for human-object interaction detection. CVPR.

[5] Wang, T., et al. Deep contextual attention for human-object interaction detection. ICCV.

[6] Gupta, T., et al. No-frills human-object interaction detection: Factorization, layout encodings, and training techniques. ICCV.

[7] Zhou, P., et al. Relation parsing neural network for human-object interaction detection. ICCV.

[8] Wan, B., et al. Pose-aware multi-level feature network for human object interaction detection. ICCV.

[9] Peyre, J., et al. Detecting unseen visual relations usinganalogies.  ICCV2019

[10] Bansal,  A., et al. Detecting  human-object interactions via functional generalization. AAAI


Zero-shot result


|Model|Full|Rare|Non-Rare|
|:-|:-:|:-:|:-:|
|Shen et al.[1] | 5.62 | - | 6.26 |
|Bansal et al.[2]|10.93|12.60|12.2|
|w/o VCL (rare first)|3.30|18.63|15.56|
|w/o VCL (non-rare first)|5.06|12.77|11.23|
|VCL (rare first)|7.55 | 18.84 | 16.58 | 
|VCL (non-rare first)|9.13 | 13.67 | 12.76 |
|VCL* (rare first)|10.06 | 24.28 | 21.43 | 12.12 | 26.71 | 23.79|
|VCL* (non-rare first)|16.22 | 18.52 | 18.06 | 20.93 | 21.02 | 20.90 |

### Object Detector.
**Noticeably, Detector has an important effect on the performance of HOI detection.**
Our experiment is based on the object detection results provided by iCAN. 
We also fine-tune the detector on HICO-DET train. The detection result on HICO-DET test is 30.79 mAP. 
We provide the object detection result [here](https://drive.google.com/file/d/1QI1kcZJqI-ym6AGQ2swwp4CKb39uLf-4/view?usp=sharing) same as the format of iCAN.

**The performance largely varies based on different detector. It is better to provide the mAP of Detector.**  



**References:**

[1] Shen, L. et al. Scaling human-object inter-action recognition through zero-shot learning

[2] Bansal,  A., et al. Detecting  human-object interactions via functional generalization. AAAI


## Q&A
### 1. The importance of re-weighting strategy.
 
We follow previous work to use re-weighting. 
It multiplies the weights to the logits before the sigmoid function. 
We empirically find this is important for rare and unseen HOI detection

### 2. Res101 Detector. 

The Resnet-101 Detector is fully based on faster-rcnn ([detectron2](https://github.com/facebookresearch/detectron2)).
We fine-tune the [R101-RPN](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml) detector (pretrained on coco) on HICO-DET. [Here](https://drive.google.com/file/d/1RgWNoc-lk8HMlcttzLghPg8LAPCdmCCG/view?usp=sharing) is the fine-tuned model.
The detection result of fine-tuned model on HICO-DET test is **30.79 mAP**.
We provide the object detection result [here](https://drive.google.com/file/d/1QI1kcZJqI-ym6AGQ2swwp4CKb39uLf-4/view?usp=sharing) same as the format of iCAN. When using the fine-tuned object detector, you should change the object_thres and humans_thres accordingly (see the test code).
The hico object annotations: [train](https://drive.google.com/file/d/1M4j5-rHcdfHYVfHQToccO0SsEGP4nGC1/view?usp=sharing) and [test](https://drive.google.com/file/d/1qyUURe978WuZRm1s-VWoC_TpTInYTUXd/view?usp=sharing) (coco format)

**Hope the future works who used fine-tuned detector provide the object test mAP.**

### 3. Verb polysemy problem. 

Verb with same name possibly has multiple meaning. For example, fly kite is largely different from fly airplane. 
Similar to previous works [Shen et al, Xu et al, ], We equally treat the verb. 
We also try to solve this problem in VCL with massive attempts (e.g. language priors, RL (search the reasonable pair)). 
However, we do not find any apparent improvement (See our supplementary materials). 

We think there are several reasons: 
1. Most verbs are not polysemy in HICO-DET.
2. Many verbs do not involve multiple objects (there are 39 verbs that interact only one object). This means there are a few composite HOIs with polysemy problem. 
3. This paper Disambiguating Visual Verbs (TPAMI) also illustrates that HICO dataset do not contain much ambiguated verbs. 

Of course, it is also possible the network could learn the commonness of the same verb. 

We think this problem in HOI understanding require to be further exploited. 

For other relation datasets such as VRD, possibly, 
VCL should take this problem into consideration. 

***Thanks for the reviewer who also points out this problem.***

### 4. VRD
We also evaluate VCL on VRD and we could improve a bit than the baseline based on VTransE.

### 5. Composition
Recently, I find our implementation also contains the composition between the HOI pair due to that our base code augment the boxes. e.g. if we augment each box 7 times and obtain 7 pair for a annotated HOI, we can augment the pairs to 7*6. This is equal to increase the batch size. We do not find this part improves the performance in our simple experiment.

### Others

I recently notice that same as iCAN (our base code), we only freeze the first block of resnet during optimization. It is necessary to optimize some resnet blocks for VCL. Otherwise, it might be more difficult to learn sharable verb representation among different HOIs. Meanwhile, I guess the re-weighting strategy from TIN might also require trainable resnet blocks.


If you have any questions about this code and the paper, welcome to contact the Zhi Hou (zhou9878 [at] uni dot sydney dot edu dot au).

## Acknowledgement
Codes are built upon [iCAN: Instance-Centric Attention Network 
for Human-Object Interaction Detection](https://arxiv.org/abs/1808.10437), [Transferable Interactiveness Network](https://arxiv.org/abs/1811.08264), [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn).
