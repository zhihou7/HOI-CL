from scripts import get_id_convert_dicts
from scripts.analysis import obtain_config

import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'VCOCO_CL_21'
dataset_name = 'HICO'
import sys
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, gt_known_labels = obtain_config(dataset_name)
if dataset_name.__contains__('VCOCO'):
    id_obj = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
           6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
           12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
           19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
           26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
           33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
           39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
           47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
           54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
           61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
           68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
           75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    id_vb = {0: 'surf', 1: 'ski', 2: 'cut', 3: 'ride', 4: 'talk_on_phone',
             5: 'kick', 6: 'work_on_computer', 7: 'eat', 8: 'sit', 9: 'jump',
             10: 'lay', 11: 'drink', 12: 'carry', 13: 'throw', 14: 'look',
             15: 'hit', 16: 'snowboard', 17: 'read', 18: 'hold',
             19: 'skateboard', 20: 'catch'}
else:
    id_vb, id_obj, id_hoi, hoi_to_obj, hoi_to_verbs = get_id_convert_dicts()

verbs = [id_vb[i] for i in range(verb_class_num)]
objs = [id_obj[i] for i in range(obj_class_num)]

if dataset_name == 'VCOCO_CL_21':
    file_list = [
        'VCL_R_union_multi_ml5_l05_t5_VERB_def1_aug5_3_new_VCOCO_test_CL_21_affordance_3ATL_1.npy',
        'VCL_R_union_multi_ml5_l05_t5_VERB_def2_aug5_3_new_VCOCO_test_CL_21_affordance_9_HOI_iter_160000.ckpt.npy',
        'VCL_R_union_multi_ml5_l05_t5_VERB_def2_aug5_3_new_VCOCO_test_CL_21_affordance_AF713_9_1_HOI_iter_200000.ckpt.npy',
        'VCL_R_union_multi_semi_ml5_l05_t5_VERB_def2_aug5_3_new_VCOCO_test_CL_21_cocoall_affordance_9_AF713_1_HOI_iter_260000.ckpt.npy'
    ]
else:
    file_list = [
    'VCL_R_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_9ATL_2.npy',
    'VCL_R_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_9_HOI_iter_800000.ckpt.npy',
    'VCL_R_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_9_HOI_iter_3000000.ckpt.npy',
    ]

list_titles = [
    'Ground Truth',
    'Known Concepts',
    'Affordance Prediction',
    'Online Concept Discovery',
    '+ Self-Training',
    '+ Novel Objects'
]


if dataset_name == 'HICO':
    fig, axs = plt.subplots(3, 2, figsize=(8, 12), dpi=100)
    font_size = 18
    label_size = 18
else:
    fig, axs = plt.subplots(3, 2, figsize=(5, 3.3), dpi=100)
    font_size = 8
    label_size = 6
plt.rcParams.update({'font.size': font_size})
for i in range(len(file_list) + 2):
    x = i // 2
    y = i % 2
    name = 'gt'
    if i == 0:
        arr = gt_labels
    elif i == 1:
        arr = gt_known_labels
    else:
        arr = np.load('../../temp/{}'.format(file_list[i-2]))
        name = file_list[i-2]
    # arr /= arr.max()
    print(arr.min(), np.sum(arr == 0.), np.sum(arr > 1.), arr.shape, arr.sum(), arr.max(), name)

    arr = arr.reshape([verb_class_num, obj_class_num])
    arr = arr * ( 1 - gt_known_labels)
    if i == 1:
        arr = gt_known_labels
    # arr = arr.transpose()
    axs[x, y].set_title(list_titles[i], fontsize=font_size)
    axs[x, y].imshow(arr)
    axs[x, y].pcolormesh(arr)
    axs[x, y].label_outer()
    axs[x, y].tick_params('both', labelsize=label_size)
    # axs[x, y].set_yticklabels(verbs)
    # axs[x, y].set_xticklabels(objs)
    if y == 0:
        axs[x, y].set_ylabel('verb', fontsize=label_size)
    # axs[x, y].tick_params('y', fontsize=16)

if dataset_name == 'HICO':
    axs[-1, -1].axis('off')
    axs[2, 0].set_xlabel('object', fontsize=label_size)
    axs[1, 1].set_xlabel('object', fontsize=label_size)
else:
    axs[2, 0].set_xlabel('object', fontsize=label_size)
    axs[2, 1].set_xlabel('object', fontsize=label_size)
fig.tight_layout()

plt.savefig(dataset_name+'result_visualized.pdf')
plt.show()
exit()
