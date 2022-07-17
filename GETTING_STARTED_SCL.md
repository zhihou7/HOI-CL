## [Affordance Transfer Learning for Human-Object Interaction Detection](https://arxiv.org/abs/2104.02867)


![](misc/compose_obj1.png)

Here ([HOI-CL-OneStage](https://github.com/zhihou7/HOI-CL-OneStage)) is the Code of VCL and FCL based on One-Stage method.


We notice we can also split V-COCO into 24 verbs. Therefore, we also provides the HOI-COCO with 24 verbs (i.e. both _instr and _obj are kept) 

## Train

#### HICO-DET,

We use two HOI images to compose HOIs among different images (i.e. VCL)

```shell
python tools/Train_ATL_HICO.py --model VCL_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9 --num_iteration 3000000
```

Here, AF713 is just a simple symbol to represent concept discovery with self-training,
'VERB' means we optimize verb category rather than HOI category,
'affordance' means online concept discovery. We will revise this to self-training in the final version. Baseline of HICO-DET is

```shell
python tools/Train_ATL_HICO.py --model VCL_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_9 --num_iteration 3000000
```

#### V-COCO

```shell
python tools/Train_ATL_HOI_COCO_21.py --model VCL_union_multi_ml5_l05_t5_VERB_def2_aug5_3_new_VCOCO_test_CL_21_affordance_9
```

#### Non-rare zero-shot with unknown concepts
```shell
python tools/Train_ATL_HICO.py --model VCL_union_batch_large2_ml5_def1_vloss2_zs3_ICL_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9 --num_iteration 1000000
```

#### Rare zero-shot with unknown concepts
```shell
python tools/Train_ATL_HICO.py --model VCL_union_batch_large2_ml5_def1_vloss2_zs4_ICL_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9 --num_iteration 300000
```

#### Novel object HOI with unknown concepts
```shell
python tools/Train_ATL_HICO.py --model VCL_union_batch1_semi_vloss2_ml5_zs11_ICL_VERB_def4_l2_aug5_3_x5new_bothzs_res101_affordance_AF713_9 --num_iteration 500000
```

`bothzs` means we use HICO and COCO dataset (novel types of objects)

#### Concept Discovery from language embedding
This code is based on pytorch

```
python concepts/discover_concepts_language.py
```

## Evaluation

Concept Discovery

```shell
PYTHONPATH=. python scripts/analysis/analysis_concepts.py Weights/VCL_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9/HOI_iter_3000000.ckpt
```

When we evaluate the unknown concepts, we mask out the known concepts to avoid the disturbance from known concepts.

zero-shot HOI detection with unknown concepts, e.g.

```shell
python tools/Test_HICO.py --model VCL_union_batch1_semi_vloss2_ml5_zs11_ICL_VERB_def4_l2_aug5_3_x5new_bothzs_res101_affordance_AF713_9 --num_iteration 500000 --topk 600
python scripts/postprocess_test.py --model VCL_union_batch1_semi_vloss2_ml5_zs11_ICL_VERB_def4_l2_aug5_3_x5new_bothzs_res101_affordance_AF713_9 --num_iteration 500000 --tag 600
```

We directly convert verb prediction to HOI prediction via co-occurrence matrix (P A_v, where P is the verb prediction, A_v is co-occurrence matrix).

## Object Affordance Recognition

HICO train set
```shell
    PYTHONPATH=. python affordance_prediction_hico.py --model VCL_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9 --num_iteration 3000000 --dataset hico_train --path /path
    PYTHONPATH=. python scripts/stat/stat_hico_affordance1.py --model VCL_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9 --num_iteration 3000000 --dataset hico_train --path /path 
```

val2017

```shell
    PYTHONPATH=. python affordance_prediction_hico.py --model VCL_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9 --num_iteration 3000000 --dataset gtval2017 --path /path
    PYTHONPATH=. python scripts/stat/stat_hico_affordance1.py --model VCL_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9 --num_iteration 3000000 --dataset gtval2017 --path /path 
```

Other datasets, such as 'gtobj365', 'gtobj365_coco', 'gtval2017', 'gthico', are similar.
## Visualization

1. visualized concepts:

```shell
    python scripts/stat/visualize_concepts.py HICO
```


```shell
    python scripts/stat/visualize_concepts.py VCOCO_CL_21
```

2. visualize relationships
```shell
    python scripts/stat/visualize_relations.py
```

## Citations
If you find this submission is useful for you, please consider citing:

```
@inproceedings{hou2022scl,
  title={Discovering Human-Object Interaction Concepts via Self-Compositional Learning},
  author={Hou, Zhi and Yu, Baosheng and Tao, Dacheng},
  booktitle={ECCV},
  year={2022}
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
  author={Hou, Zhi and Yu, Baosheng and Qiao, Yu and Peng, Xiaojiang and Tao, Dacheng},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgement

Thanks for all reviewer's comments.  Codes are built upon [HOI-CL](https://github.com/zhihou7/HOI-CL/), [Visual Compositional Learning for Human-Object Interaction Detection](https://arxiv.org/abs/2007.12407), [iCAN: Instance-Centric Attention Network
for Human-Object Interaction Detection](https://arxiv.org/abs/1808.10437)