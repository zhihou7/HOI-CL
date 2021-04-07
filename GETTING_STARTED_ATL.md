### Affordance Transfer Learning for Human-Object Interaction Detection

Thanks for all reviewer's comments. That's very valuable for our next work. 
ATL gives a new insight to HOI understanding and in fact inspires a lot to our next work

Here ([HOI-CL-OneStage](https://github.com/zhihou7/HOI-CL-OneStage)) is the Code of VCL and FCL based on One-Stage method.


### 1. Train ATL on HICO-DET
```Shell
python tools/Train_ATL_HICO.py 
```

### 2. Train ATL on HOI-COCO

```Shell
python tools/Train_ATL_HOI_COCO_21.py
```

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

```Shell
python scripts/affordance/stat_hico_affordance.py
```

## Data & Model
#### Data
We present the differences between different detector in our paper and analyze the effect of object boxes on HOI detection. VCL detector and DRG detector can be download from the corresponding paper. 
Here, we provide the GT boxes.

GT boxes annotation: https://drive.google.com/file/d/15UXbsoverISJ9wNO-84uI4kQEbRjyRa8/view?usp=sharing

#### Pre-trained model

FCL Long-tailed Model: https://drive.google.com/file/d/144F7srsnVaXFa92dvsQtWm2Sm0b30jpi/view?usp=sharing

ATL model (HOI-COCO): 
ATL model (HICO-DET): 

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