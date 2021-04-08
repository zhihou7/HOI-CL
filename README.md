### Compositional Learning for Human-Object Interaction Detection

This repository includes the code of [Visual Compositional Learning for Human-Object Interaction Detection](https://arxiv.org/abs/2007.12407) (ECCV2020), 
[Detecting Human-Object Interaction via Fabricated Compositional Learning](https://arxiv.org/abs/2103.08214) (CVPR2021), [Affordance Transfer Learning for Human-Object Interaction Detection](https://arxiv.org/abs/2104.02867) (CVPR2021)

This repository is built from the code of previous approaches. Thanks for their excellent work.


Code is being constructed.

Here ([FCL_VCOCO](https://github.com/zhihou7/FCL_VCOCO)) is the Code of FCL on V-COCO

Here ([HOI-CL-OneStage](https://github.com/zhihou7/HOI-CL-OneStage)) is the Code of VCL and ATL based on One-Stage method.

[Here](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/zhou9878_uni_sydney_edu_au/EXOYJZ1N_phJlFW0nTgnABgBuyghLGqVE8C2t5EfiV--xA?e=cXM24T) we provide the code on VRD. (The code might be 

Thanks for all reviewer's comments.

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
   
   Due to the limitation of space in my google drive, Additional files for ATL are provided in OneDrive.
   

## VCL
See [GETTING_STARTED_VCL.md](GETTING_STARTED_VCL.md). You can also find details in https://github.com/zhihou7/VCL.

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

See [GETTING_STARTED_ATL.md](GETTING_STARTED_ATL.md). We try to use a HOI model to recognize object affordance, i.e. what actions can be applied to an object.

## Data & Model
#### Data
We present the differences between different detector in our paper and analyze the effect of object boxes on HOI detection. VCL detector and DRG detector can be download from the corresponding paper. Due to the space limitation of Google Drive, there are many files provided in CloudStor. Many thanks to CloudStor and The University of Sydney.
Here, we provide the GT boxes.

GT boxes annotation: https://drive.google.com/file/d/15UXbsoverISJ9wNO-84uI4kQEbRjyRa8/view?usp=sharing

FCL was finished about 10 months ago. In the first submission, we compare the difference among COCO detector, Fine-tuned Detector and GT boxes. We further find DRG object detector largely increases the baseline. 
All these comparisons illustrate the significant effect of object detector on HOI. That's really necessary to provide the performance of object detector.

HOI-COCO training data: https://cloudstor.aarnet.edu.au/plus/s/6NzReMWHblQVpht

Please notice train2017 might contain part of V-COCO test data. Thus, we just use train2014 in our experiment. If we use train2017, the result might be better (improve about 0.5%). We think that is the case: we have localized objects, but we do not know the interaction. 

##### Affordance Recognition evaltion dataset

Evalation of Object365_COCO: https://cloudstor.aarnet.edu.au/plus/s/GpNkjOHaS8xN5ar

Evalation of COCO (val2017): https://cloudstor.aarnet.edu.au/plus/s/QBXOYiJ5NHqtcti

Evalation of Object365 (novel classes): https://cloudstor.aarnet.edu.au/plus/s/pRBLkhy9TUm5xGo

Evalation of HICO (test): https://cloudstor.aarnet.edu.au/plus/s/AFrv822lPC30iHt



#### Object Dataset (HICO)
Here we provide the object datasets that we use in this repo. The format is the same as the training data of HICO.

COCO: https://cloudstor.aarnet.edu.au/plus/s/9cgMYKq5B4waawA

#### Object Dataset (HOI-COCO)
Object365_COCO: https://cloudstor.aarnet.edu.au/plus/s/VuHvDdp5msRnpqn. (we do not run this experiment).

COCO: https://cloudstor.aarnet.edu.au/plus/s/N35ovTXWtLmG9ZN

HOI-COCO: https://cloudstor.aarnet.edu.au/plus/s/YEiPiX0B3jaFasU


#### Pre-trained model

FCL Long-tailed Model: https://drive.google.com/file/d/144F7srsnVaXFa92dvsQtWm2Sm0b30jpi/view?usp=sharing

ATL model (HICO-DET): https://cloudstor.aarnet.edu.au/plus/s/NfKOuJKV5bUWiIA

ATL model (HOI-COCO)(COCO): https://cloudstor.aarnet.edu.au/plus/s/zZfJM4ctylwAEiZ

ATL model (HOI-COCO)(COCO, HICO): https://cloudstor.aarnet.edu.au/plus/s/zih9Vcdlwbpt92v

### Citations
If you find this series of work are useful for you, please consider citing:

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
