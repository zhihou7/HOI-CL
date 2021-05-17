#!/bin/bash

# Download training data
echo "Downloading training data..."
python lib/ult/Download_data.py 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD Data/action_index.json
python lib/ult/Download_data.py 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY Data/prior_mask.pkl
python lib/ult/Download_data.py 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
python lib/ult/Download_data.py 1Y9yRTntfThrKMJbqyMzVasua25GUucf4 Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl
python lib/utl/Download_data.py 1QI1kcZJqI-ym6AGQ2swwp4CKb39uLf-4 Data/Test_HICO_res101_3x_FPN_hico.pkl
python lib/utl/Download_data.py 15UXbsoverISJ9wNO-84uI4kQEbRjyRa8 Data/gt_annotations.pkl

python lib/ult/Download_data.py 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ Data/Trainval_GT_HICO.pkl
python lib/ult/Download_data.py 1YrsQUcBEF31cvqgCZYmX5j-ns2tgYXw7 Data/Trainval_GT_VCOCO.pkl
python lib/ult/Download_data.py 1PPPya4M2poWB_QCoAheStEYn3rPMKIgR Data/Trainval_Neg_HICO.pkl
python lib/ult/Download_data.py 1oGZfyhvArB2WHppgGVBXeYjPvgRk95N9 Data/Trainval_Neg_VCOCO.pkl
python lib/ult/Download_data.py 1um07VNrfz03Oyytlp_4wbknuMOmNJ9mH Data/Trainval_GT_VCOCO_obj.pkl
python lib/ult/Download_data.py 1SuOCeQXhTHT-Txk-XDP67uZveATC7ckC Data/Trainval_Neg_VCOCO_obj.pkl
python lib/ult/Download_data.py 1BFJAniI4rZpq2KsZaxEoBB5sCgl0zBmZ Data/Trainval_GT_VCOCO_with_pose_obj.pkl
python lib/ult/Download_data.py 1kKkUj1zyWh7-hEK3PumOi6TwUlsoMPkm Data/Trainval_Neg_VCOCO_with_pose_obj.pkl

#https://drive.google.com/file/d/1BFJAniI4rZpq2KsZaxEoBB5sCgl0zBmZ/view?usp=sharing
#https://drive.google.com/file/d/1kKkUj1zyWh7-hEK3PumOi6TwUlsoMPkm/view?usp=sharing