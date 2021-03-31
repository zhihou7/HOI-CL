### Compositional Learning for Human-Object Interaction Detection

This repository includes the code of [Visual Compositional Learning for Human-Object Interaction Detection](https://arxiv.org/abs/2007.12407) (ECCV2020), 
[Detecting Human-Object Interaction via Fabricated Compositional Learning](https://arxiv.org/abs/2103.08214) (CVPR2021), Affordance Transfer Learning for Human-Object Interaction Detection (CVPR2021)

This repository is built from the code of previous approaches. Thanks for their excellent work.


Code is being constructed.

Here ([FCL_VCOCO](https://github.com/zhihou7/FCL_VCOCO)) is the Code of FCL on V-COCO
Here ([HOI-CL-OneStage](https://github.com/zhihou7/HOI-CL-OneStage)) is the Code of VCL and FCL based on One-Stage method.

Thanks for all reviewer's comments. Our new work, an extension of VCL, will be coming soon.

## Prerequisites

This codebase was developed and tested with Python3.7, Tensorflow 1.14.0, Matlab (for evaluation), CUDA 10.0 and Centos 7


## Installation

1. Download HICO-DET dataset. Setup V-COCO and COCO API. Setup HICO-DET evaluation code.
    ```Shell
    chmod +x ./misc/download_dataset.sh 
    ./misc/download_dataset.sh 
    ```

2. Install packages by pip.

    ```
    pip install -r requirements.txt
    ```
   
3. Download COCO pre-trained weights and training data
    ```Shell
    chmod +x ./misc/download_training_data.sh 
    ./misc/download_training_data.sh
    ```


## VCL

## Training

1. Train an VCL on HICO-DET
    ```Shell
    python tools/Train_VCL_ResNet_HICO.py --num_iteration 800000
    ```
    
2. Train an VCL for rare first zero-shot on HICO-DET
    ```Shell
    python tools/Train_VCL_ResNet_HICO.py --model VCL_union_multi_zs3_def1_l2_ml5_rew51_aug5_3_x5new --num_iteration 600000
    ```
  
3. Train an VCL for non-rare first zero-shot on HICO-DET
    ```Shell
    python tools/Train_VCL_ResNet_HICO.py --model VCL_union_multi_zs4_def1_l2_ml5_rew51_aug5_3_x5new --num_iteration 400000
    ```

4. Train an VCL on V-COCO
    ```Shell
    python tools/Train_VCL_ResNet_VCOCO.py --model VCL_union_multi_ml1_l05_t3_rew_aug5_3_new_VCOCO_test --num_iteration 400000
    ```

### Model Parameters
Our model will converge at around iteration 500000 in HICO-DET. V-COCO will converge after 200000 iterations. We provide the model parameters that we trained as follows,

V-COCO: https://drive.google.com/file/d/1SzzMw6fS6fifZkpuar3B40dIl7YLNoYF/view?usp=sharing. I test the result is 47.82. The baseline also decreases compared to the reported result. The model in my reported result is deleted by accident. Empirically, hyper-parameters $lambda_1$ affects V-COCO more apparently.

HICO: https://drive.google.com/file/d/16unS3joUleoYlweX0iFxlU2cxG8csTQf/view?usp=sharing

HICO(Res101): https://drive.google.com/file/d/1iiCywBR0gn6n5tPzOvOSmZw_abOmgg53/view?usp=sharing


## Testing
1. Test an VCL on V-COCO
    ```Shell
     python tools/Test_ResNet_VCOCO.py --num_iteration 200000
    ```
2. Test an VCL on HICO-DET
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
 

## FCL
### Train

3. Train Zero-Shot HOI model with FCL on HICO-DET
    ```Shell
    python tools/Train_FCL_HICO.py
    ```
    
### Test

we provide this scripts to test code and eval the FCL results.

    ```Shell
    python scripts/eval.py
    ```

## ATL

### 1. Train ATL on HICO-DET

### 2. Train ATL on HOI-COCO

### 3. Affordance Recognition

1. extract affordance feature

```Shell
python scripts/affordance/extract_affordance_feature.py

```

2. convert affordance feature to feature bank (select 100 instances for each verb)
```Shell
python scripts/affordance/convert_feats_to_affor_bank_hico.py
```

3. extract object feature
```Shell
python scripts/affordance/extract_obj_feature.py
```

4. obtain hoi prediction
```Shell
python scripts/affordance/extract_hoi_preds.py
```

5. statistic of affordance prediction results.

```Sheel
python scripts/affordance/stat_hico_affordance.py
```

## Data & Model
#### Data
We present the differences between different detector in our paper and analyze the effect of object boxes on HOI detection. VCL detector and DRG detector can be download from the corresponding paper. 
Here, we provide the GT boxes.

GT boxes annotation: https://drive.google.com/file/d/15UXbsoverISJ9wNO-84uI4kQEbRjyRa8/view?usp=sharing

FCL was finished about 10 months ago. In the first submission, we compare the difference among COCO detector, Fine-tuned Detector and GT boxes. We further find DRG object detector largely increases the baseline. 
All these comparisons illustrate the significant effect of object detector on HOI. That's really necessary to provide the performance of object detector.

#### Pre-trained model

FCL Long-tailed Model: https://drive.google.com/file/d/144F7srsnVaXFa92dvsQtWm2Sm0b30jpi/view?usp=sharing

ATL model: 

### Citations
If you find this submission is useful for you, please consider citing:

```
@inproceedings{hou2021fcl,
  title={Detecting Human-Object Interaction via Fabricated Compositional Learning},
  author={Hou, Zhi and Baosheng, Yu and Qiao, Yu and Peng, Xiaojiang and Tao, Dacheng},
  booktitle={CVPR},
  year={2021}
}
```

```
@inproceedings{hou2021vcl,
  title={Visual Compositional Learning for Human-Object Interaction Detection},
  author={Hou, Zhi and Peng, Xiaojiang and Qiao, Yu  and Tao, Dacheng},
  booktitle={ECCV},
  year={2020}
}
```

```
@inproceedings{hou2021atl,
  title={Affordance Transfer Learning for Human-Object Interaction Detection},
  author={Hou, Zhi and Baosheng, Yu and Qiao, Yu and Peng, Xiaojiang and Tao, Dacheng},
  booktitle={CVPR},
  year={2021}
}
```