from scripts import get_id_convert_dicts
from scripts.analysis import obtain_config

import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'VCOCO_CL_21'
dataset_name = 'HICO'
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
    'Online Concept Discovery with Self-Training',
    '+ Novel objects'
]

obj_id = 4
confidence_threshold = 0.4
affordance_threshold = 0.3
# verb_id = 23
verb_id=26
verb_id=27
verb_id=34
verb_id=64
verb_id=83
verb_id=44
verb_id=91

selected_verbs = [83, 91, 34]
fig_column = 1
fig, axs = plt.subplots(len(selected_verbs) // fig_column, fig_column, figsize =(6, 13))

for f_idx, verb_id in enumerate(selected_verbs):
    i = -1
    concept_confidence = np.load('../../temp/{}'.format(file_list[i]))
    concept_confidence = concept_confidence.reshape([verb_class_num, obj_class_num])

    # arr_gt = np.load('../../temp/{}'.format(file_list[0]))
    # arr_gt = arr_gt.reshape([verb_class_num, obj_class_num])
    arr_known = gt_known_labels.reshape([verb_class_num, obj_class_num])
    arr_gt = gt_labels.reshape([verb_class_num, obj_class_num])

    obj_idx = np.argwhere(concept_confidence[verb_id] > confidence_threshold).reshape(-1) # find related objects
    obj_idx_known = np.argwhere(arr_known[verb_id] > confidence_threshold).reshape(-1) # confidence
    obj_idx_gt = np.argwhere(arr_gt[verb_id] > confidence_threshold).reshape(-1)
    # obj_idx_ = np.argwhere(arr_gt[verb_id] > 0.4).reshape(-1)

    values = concept_confidence[:, obj_idx]  # find affordance of related objects
    values_known = arr_known[:, obj_idx_known]
    values_gt = arr_gt[:, obj_idx_gt]

    relation_confidence = np.sum(values, axis=1).reshape(-1)  # shared affordance
    relation_confidence_known = np.sum(values_known, axis=1).reshape(-1)
    relation_confidence_gt = np.sum(values_gt, axis=1).reshape(-1)

    affordance_id_list = np.argwhere(relation_confidence > affordance_threshold * len(obj_idx))
    affordance_id_list_known = np.argwhere(relation_confidence_known > affordance_threshold * len(obj_idx_known))
    affordance_id_list_gt = np.argwhere(relation_confidence_gt > affordance_threshold * len(obj_idx_gt))

    affordance_id_list_known = affordance_id_list_known.reshape(-1).tolist()
    affordance_id_list = affordance_id_list.reshape(-1).tolist()
    affordance_id_list_gt = affordance_id_list_gt.reshape(-1).tolist()

    affordance_id_list.extend(affordance_id_list_known)
    # v_id_list.extend(v_id_list_gt)

    affordance_id_list = sorted(set(affordance_id_list))

    if 57 in affordance_id_list:
        affordance_id_list.remove(57)

    X = np.arange(len(affordance_id_list))


    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])

    barWidth = 0.3
    # import ipdb;ipdb.set_trace()
    hi = f_idx % fig_column
    wi = f_idx // fig_column
    if fig_column == 1:
        current_axs = axs[f_idx]
    else:
        current_axs = axs[hi, wi]
    current_axs.set_title(verbs[verb_id])
    current_axs.barh(y=X, width=[relation_confidence.tolist()[i] for i in affordance_id_list], height = barWidth, label='Self-Training')
    current_axs.barh(y=X+barWidth, width=[relation_confidence_known.tolist()[i] for i in affordance_id_list], height = barWidth, label='Known')
    # plt.barh(y=X+barWidth*2, width=[values_affordance_gt.tolist()[i] for i in v_id_list], height = barWidth, label='GT')
    current_axs.set_yticks([r + barWidth for r in range(len(affordance_id_list))],)
    current_axs.set_yticklabels([verbs[i] for i in affordance_id_list], fontdict=None, minor=False)

    current_axs.legend()

plt.savefig(dataset_name+'verb_{}.pdf'.format(''.join([verbs[verb_id] for verb_id in selected_verbs])))
plt.show()

exit()
concept_confidence = np.load('')

# verb_id means the verb category index, i.e. which verb we want to analyze
# concept_confidence is the confidence matrix M_c
# verb_class_num is the number of verb categories, while obj_class_num is the number of object categories
concept_confidence = concept_confidence.reshape([verb_class_num, obj_class_num])
obj_idx = np.argwhere(concept_confidence[verb_id] > confidence_threshold).reshape(-1) # find related objects

# we obtain all affordances relate to verb_id
values = concept_confidence[:, obj_idx]  # find affordance (i.e. verb) of related objects
relation_confidence = np.sum(values, axis=1).reshape(-1)  # relation_confidence represents the co-relation confidence

# affordance_threshold is the pre-defined threshold to select correlative verbs/affordances
affordance_id_list = np.argwhere(relation_confidence > affordance_threshold * len(obj_idx)) # verb_id list
