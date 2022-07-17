## Visual Compositional Learning for Human-Object Interaction Detection (ECCV2020)

![](misc/imagine.png)

### Training

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


### Testing
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
 


## Citations
If you find this submission is useful for you, please consider citing:

```
@inproceedings{hou2021fcl,
  title={Detecting Human-Object Interaction via Fabricated Compositional Learning},
  author={Hou, Zhi and Yu, Baosheng and Qiao, Yu and Peng, Xiaojiang and Tao, Dacheng},
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
  author={Hou, Zhi and Yu, Baosheng and Qiao, Yu and Peng, Xiaojiang and Tao, Dacheng},
  booktitle={CVPR},
  year={2021}
}
```