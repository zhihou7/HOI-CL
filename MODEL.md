
## Pre-trained model

### FCL
FCL Long-tailed Model: https://drive.google.com/file/d/144F7srsnVaXFa92dvsQtWm2Sm0b30jpi/view?usp=sharing

Zero shot (rare first): https://cloudstor.aarnet.edu.au/plus/s/CiVjoneqoiYD0Re

Zero-Shot (non-rare first): https://cloudstor.aarnet.edu.au/plus/s/Q1NSDfiPPMXZxBC



### ATL 
ATL model (HICO-DET) : https://cloudstor.aarnet.edu.au/plus/s/NfKOuJKV5bUWiIA

This is the ATL pre-trained model on HICO-DET. This model is fine-tuned with around 500000 iterations after we train th ATL model. We pick the the best checkpoint on HOI detection. Noticeably, the fine-tuning step does not improve the result of affordance recognition apparently. The model name is ATL_union_batch1_atl_l2_def4_epoch2_epic2_cosine5_s0_7_vloss2_rew2_aug5_3_x5new_coco_res101.

ATL model (HOI-COCO)(COCO): https://cloudstor.aarnet.edu.au/plus/s/zZfJM4ctylwAEiZ 

model name is ATL_union_multi_atl_ml5_l05_t5_def2_aug5_3_new_VCOCO_test_coco_CL_21

ATL model (HOI-COCO)(COCO, HICO): https://cloudstor.aarnet.edu.au/plus/s/zih9Vcdlwbpt92v

model name is ATL_union_multi_atl_ml5_l05_t5_def2_aug5_3_new_VCOCO_test_both_CL_21

### VCL 
Our model will converge at around iteration 500000 in HICO-DET. V-COCO will converge after 200000 iterations. We provide the model parameters that we trained as follows,

VCL on V-COCO: https://drive.google.com/file/d/1SzzMw6fS6fifZkpuar3B40dIl7YLNoYF/view?usp=sharing. I test the result is 47.82. The baseline also decreases compared to the reported result. The model in my reported result is deleted by accident. Empirically, hyper-parameters $lambda_1$ affects V-COCO more apparently.

VCL on HICO: https://drive.google.com/file/d/16unS3joUleoYlweX0iFxlU2cxG8csTQf/view?usp=sharing

model name is VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new

VCL on HICO(Res101): https://drive.google.com/file/d/1iiCywBR0gn6n5tPzOvOSmZw_abOmgg53/view?usp=sharing

model name is iCAN_R_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101. Here ``iCAN_R'' has no special meaning, just because the code is based on the repository of iCAN.

Here, we also provide zero-shot models:
- non-rare first (better than reported): https://cloudstor.aarnet.edu.au/plus/s/1CmocazcoNGtM4D
- rare first (a bit worse than reported): https://cloudstor.aarnet.edu.au/plus/s/TqhEN5IzPREMr5F
