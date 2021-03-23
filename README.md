### Compositional Learning for Human-Object Interaction Detection

This repository includes the code of Visual Compositional Learning for Human-Object Interaction Detection (ECCV2020), 
Detecting Human-Object Interaction via Fabricated Compositional Learning (CVPR2021)

This repository is built from the code of previous approaches. Thanks for their excellent work.


Code is being constructed.

Here ([FCL_VCOCO](https://github.com/zhihou7/FCL_VCOCO)) is the Code of FCL on V-COCO

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
   
## Training
1. Download COCO pre-trained weights and training data
    ```Shell
    chmod +x ./misc/download_training_data.sh 
    ./misc/download_training_data.sh
    ```

3. Train Zero-Shot HOI model with FCL on HICO-DET
    ```Shell
    python tools/Train_FCL_HICO.py
    ```
    
### Test

we provide this scripts to test code and eval the FCL results.

    ```Shell
    python scripts/eval.py
    ```
    
### Data & Model
#### Data
We present the differences between different detector in our paper and analyze the effect of object boxes on HOI detection. VCL detector and DRG detector can be download from the corresponding paper. 
Here, we provide the GT boxes.

GT boxes annotation: https://drive.google.com/file/d/15UXbsoverISJ9wNO-84uI4kQEbRjyRa8/view?usp=sharing

FCL was finished about 10 months ago. In the first submission, we compare the difference among COCO detector, Fine-tuned Detector and GT boxes. We further find DRG object detector largely increases the baseline. 
All these comparisons illustrate the significant effect of object detector on HOI. That's really necessary to provide the performance of object detector.

#### Pre-trained model

FCL Long-tailed Model: https://drive.google.com/file/d/144F7srsnVaXFa92dvsQtWm2Sm0b30jpi/view?usp=sharing


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
