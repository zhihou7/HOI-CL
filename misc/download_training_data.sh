#!/bin/bash

# Download training data
echo "Downloading training data..."
# python lib/ult/Download_data.py 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD Data/action_index.json
# python lib/ult/Download_data.py 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY Data/prior_mask.pkl
# python lib/ult/Download_data.py 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
# python lib/ult/Download_data.py 1Y9yRTntfThrKMJbqyMzVasua25GUucf4 Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl
# python lib/utl/Download_data.py 1QI1kcZJqI-ym6AGQ2swwp4CKb39uLf-4 Data/Test_HICO_res101_3x_FPN_hico.pkl
# python lib/utl/Download_data.py 15UXbsoverISJ9wNO-84uI4kQEbRjyRa8 Data/gt_annotations.pkl

# python lib/ult/Download_data.py 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ Data/Trainval_GT_HICO.pkl
# python lib/ult/Download_data.py 1YrsQUcBEF31cvqgCZYmX5j-ns2tgYXw7 Data/Trainval_GT_VCOCO.pkl
# python lib/ult/Download_data.py 1PPPya4M2poWB_QCoAheStEYn3rPMKIgR Data/Trainval_Neg_HICO.pkl
# python lib/ult/Download_data.py 1oGZfyhvArB2WHppgGVBXeYjPvgRk95N9 Data/Trainval_Neg_VCOCO.pkl
# python lib/ult/Download_data.py 1um07VNrfz03Oyytlp_4wbknuMOmNJ9mH Data/Trainval_GT_VCOCO_obj.pkl
# python lib/ult/Download_data.py 1SuOCeQXhTHT-Txk-XDP67uZveATC7ckC Data/Trainval_Neg_VCOCO_obj.pkl
# python lib/ult/Download_data.py 1BFJAniI4rZpq2KsZaxEoBB5sCgl0zBmZ Data/Trainval_GT_VCOCO_with_pose_obj.pkl
# python lib/ult/Download_data.py 1kKkUj1zyWh7-hEK3PumOi6TwUlsoMPkm Data/Trainval_Neg_VCOCO_with_pose_obj.pkl

gdown 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD -O Data/action_index.json
gdown 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY -O Data/prior_mask.pkl
gdown 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru -O Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
gdown 1Y9yRTntfThrKMJbqyMzVasua25GUucf4 -O Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl
gdown 1QI1kcZJqI-ym6AGQ2swwp4CKb39uLf-4 -O Data/Test_HICO_res101_3x_FPN_hico.pkl
gdown 15UXbsoverISJ9wNO-84uI4kQEbRjyRa8 -O Data/gt_annotations.pkl

gdown 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ -O Data/Trainval_GT_HICO.pkl
gdown 1YrsQUcBEF31cvqgCZYmX5j-ns2tgYXw7 -O Data/Trainval_GT_VCOCO.pkl
gdown 1PPPya4M2poWB_QCoAheStEYn3rPMKIgR -O Data/Trainval_Neg_HICO.pkl
gdown 1oGZfyhvArB2WHppgGVBXeYjPvgRk95N9 -O Data/Trainval_Neg_VCOCO.pkl
gdown 1um07VNrfz03Oyytlp_4wbknuMOmNJ9mH -O Data/Trainval_GT_VCOCO_obj.pkl
gdown 1SuOCeQXhTHT-Txk-XDP67uZveATC7ckC -O Data/Trainval_Neg_VCOCO_obj.pkl
gdown 1BFJAniI4rZpq2KsZaxEoBB5sCgl0zBmZ -O Data/Trainval_GT_VCOCO_with_pose_obj.pkl
gdown 1kKkUj1zyWh7-hEK3PumOi6TwUlsoMPkm -O Data/Trainval_Neg_VCOCO_with_pose_obj.pkl

# gdown 0B1_fAEgxdnvJR1N3c1FYRGo1S1U -O Weights/coco_900-1190k.tgz

#https://drive.google.com/file/d/1BFJAniI4rZpq2KsZaxEoBB5sCgl0zBmZ/view?usp=sharing
#https://drive.google.com/file/d/1kKkUj1zyWh7-hEK3PumOi6TwUlsoMPkm/view?usp=sharing