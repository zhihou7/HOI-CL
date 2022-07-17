#!/usr/bin/env python

import argparse

import _init_paths
from sklearn.metrics import average_precision_score

from networks.tools1 import get_convert_matrix
from scripts import get_id_convert_dicts, get_convert_matrix_coco3
from scripts import get_hoi_convert_dicts, get_id_dicts, get_id_convert_dicts
from ult.config import cfg
import os
import json

# dataset = 'gtobj365_coco'
# dataset = 'gtobj365'
# dataset = 'gtval2017'
DATA_DIR = './Data/'

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=259638, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_R_union_batch_semi_l2_def1_epoch2_epic2_cosine5_s1_7_gc_gall_embloss_vloss2_var_gan_dax_xrew45_randso2_aug5_3_x5new_coco_res101_2', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='Human threshold',
                        default='hico_train', type=str)
    parser.add_argument('--num', dest='num', default=100,
                        type=int)
    parser.add_argument('--pred_type', dest='pred_type', default=1,
                        type=int)
    parser.add_argument('--num_verbs', dest='num_verbs', default=100,
                        type=int)
    parser.add_argument('--incre_classes', dest='incre_classes',
                        help='Human threshold',
                        default=None, type=str)
    args = parser.parse_args()
    return args


def cal_ap(gt_labels, affordance_probs_new, mask):

    # ap = average_precision_score(gt_labels.reshape(-1), affordance_stat_tmp1.reshape(-1))
    # print(ap)
    # exit()
    #
    mask = mask.reshape([-1])
    gt_labels = gt_labels.reshape(-1).tolist()
    affordance_probs_new = affordance_probs_new.reshape(-1).tolist()
    assert len(mask) == len(gt_labels) == len(affordance_probs_new)
    gt_labels = [gt_labels[i] for i in range(len(mask)) if mask[i] == 0]
    affordance_probs_new = [affordance_probs_new[i] for i in range(len(mask)) if mask[i] == 0]

    ap = average_precision_score(gt_labels, affordance_probs_new)
    return ap


def obtain_config(file_name):
    if file_name.__contains__('VCOCO') and file_name.__contains__('CL_24'):
        num_classes = 222
        verb_class_num = 24
        obj_class_num = 80

        gt_label_file = open(DATA_DIR + 'vcoco_concepts.csv')
        import numpy as np

        gt_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        concept_gt_pairs = []
        for line in gt_label_file.readlines():
            arrs = line.split(' ')
            v = arrs[1]
            o = arrs[2]
            gt_labels[int(v)][int(o)] = 1.
            concept_gt_pairs.append((int(v), int(o)))
    elif file_name.__contains__('VCOCO') and file_name.__contains__('CL_21'):
        convert_24_21 = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 7,
         16: 14, 17: 15, 18: 15, 19: 16, 20: 17, 21: 18, 22: 19, 23: 20}
        num_classes = 222
        verb_class_num = 21
        obj_class_num = 80

        gt_label_file = open(DATA_DIR + 'vcoco_concepts.csv')
        import numpy as np

        gt_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        concept_gt_pairs = []
        for line in gt_label_file.readlines():
            arrs = line.split(' ')
            v = arrs[1]
            o = arrs[2]
            gt_labels[convert_24_21[int(v)]][int(o)] = 1.
            concept_gt_pairs.append((convert_24_21[int(v)], int(o)))
    else:
        num_classes = 600
        verb_class_num = 117
        obj_class_num = 80

        gt_label_file = open(DATA_DIR + 'HICO_concepts.csv')
        import numpy as np

        gt_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        concept_gt_pairs = []
        for line in gt_label_file.readlines():
            arrs = line.split(' ')
            v = arrs[1]
            o = arrs[2]
            gt_labels[int(v)][int(o)] = 1.
            concept_gt_pairs.append((int(v), int(o)))
    return num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs


def stat_concept_result(model_name):

    obj_id = []

    num_gt_hois = 0
    new_verb_feats = []
    if not model_name.__contains__('VCOCO'):
        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix()
        num_hoi_classes = 600
        no_interactions = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107, 111, 129, 146, 160, 170, 174, 186, 194, 198,
                           208, 214, 224, 232, 235, 239, 243, 247, 252, 257, 264, 273, 283, 290, 295, 305, 313, 325,
                           330, 336, 342, 348, 352, 356, 363, 368, 376, 383, 389, 393, 397, 407, 414, 418, 429, 434,
                           438, 445, 449, 453, 463, 474, 483, 488, 502, 506, 516, 528, 533, 538, 546, 550, 558, 562,
                           567, 576, 584, 588, 595, 600]
        no_interactions = [item - 1 for item in no_interactions]
    elif model_name.__contains__('CL_24'):
        hoi_to_obj, hoi_to_verbs, verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(
            verb_class_num=24)
        num_hoi_classes = 222
    else:
        hoi_to_obj, hoi_to_verbs, verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(
            verb_class_num=21)
        num_hoi_classes = 222

    import pickle
    num_gt_verbs = [67.0, 43.0, 101.0, 101.0, 108.0, 50.0, 101.0, 28.0, 342.0, 115.0, 49.0, 102.0, 26.0, 78.0, 101.0, 109.0, 106.0, 101.0, 101.0, 103.0, 120.0, 108.0, 3.0, 120.0, 101.0, 101.0, 107.0, 79.0, 101.0, 1.0, 132.0, 19.0, 101.0, 47.0, 102.0, 109.0, 1649.0, 52.0, 97.0, 129.0, 61.0, 217.0, 13.0, 213.0, 139.0, 101.0, 51.0, 103.0, 101.0, 101.0, 103.0, 30.0, 103.0, 6.0, 101.0, 32.0, 4.0, 0.0, 113.0, 82.0, 30.0, 10.0, 101.0, 24.0, 59.0, 151.0, 57.0, 104.0, 62.0, 38.0, 120.0, 101.0, 123.0, 161.0, 16.0, 108.0, 608.0, 101.0, 121.0, 101.0, 101.0, 68.0, 20.0, 101.0, 2.0, 108.0, 113.0, 285.0, 5.0, 44.0, 18.0, 7.0, 5.0, 215.0, 106.0, 69.0, 37.0, 25.0, 291.0, 108.0, 1.0, 101.0, 101.0, 101.0, 117.0, 72.0, 75.0, 101.0, 101.0, 101.0, 151.0, 149.0, 174.0, 1.0, 137.0, 177.0, 1.0]
    num_verb = 117
    if model_name.__contains__('VCOCO'):
        feats_verbs = pickle.load(
            open(cfg.LOCAL_DATA + '/feats1/' + '{}_train_HICO_verb_feats.pkl'.format(model_name),
                 'rb'))
        num_verb = 21 if model_name.__contains__('CL_21') else 24
    else:
        feats_verbs = pickle.load(
            open(cfg.LOCAL_DATA + '/feats_verb/' + '{}_{}_train_HICO_verb_feats.pkl'.format(model_name, args.num_verbs), 'rb'))

    acts = feats_verbs['A_list']
    import numpy as np
    num_gt_verbs = np.zeros(num_verb)
    for hoi in acts:
        verb_nums = np.asarray(np.matmul(hoi, verb_to_HO_matrix.transpose()) > 0, np.float32)
        num_gt_verbs += verb_nums
    zero_verbs = np.argwhere(num_gt_verbs == 0)
    obj_verb = {}
    for i in range(81):
        obj_verb[i] = []

    verb_to_HO_matrix_preds = verb_to_HO_matrix
    if model_name.__contains__('ICL'):
        # incremental continual learning
        os.environ['DATASET'] = 'HICO_res101_icl'
        from networks.HOI_Concept_Discovery import HOIICLNet

        incremental_class_pairs = [[]]
        if args.incre_classes is not None and os.path.exists(args.incre_classes):
            incremental_class_pairs = json.load(open(args.incre_classes))
        net = HOIICLNet(model_name=model_name, task_id=1, incremental_class_pairs=incremental_class_pairs)
        net.obj_to_HO_matrix = net.incre_obj_to_HOI
        verb_to_HO_matrix_preds = net.verb_to_HO_matrix_np
        pred_num_hoi_classes = net.sum_num_classes
    elif model_name.__contains__('VCOCO') and model_name.__contains__('CL_'):
        os.environ['DATASET'] = 'VCOCO1'
        from networks.HOI import DisentanglingNet

        pred_num_hoi_classes = 222
        net = DisentanglingNet(model_name=model_name)
    elif model_name.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import DisentanglingNet

        net = DisentanglingNet(model_name=model_name)
        pred_num_hoi_classes = net.num_classes
    else:
        os.environ['DATASET'] = 'VCOCO'
        from networks.HOI import DisentanglingNet

        net = DisentanglingNet(model_name=model_name)

    # exit()
    # for i in range(600):
    #     num_gt_verbs[hoi_to_verbs[i]] += num_gt_hois[i]
    # print(coco_to_hico_obj)

    import pickle
    if model_name.__contains__('VCOCO'):
        f_name = cfg.LOCAL_DATA + "/obj_hoi_map_new{}/{}_{}_obj_hoi_map_list.pkl".format(args.pred_type, args.dataset,
                                                                                          model_name)
    else:
        f_name = cfg.LOCAL_DATA + "/obj_hoi_map_new{}_{}/{}_{}_obj_hoi_map_list.pkl".format(args.pred_type if args.pred_type > 0 else 1, args.num_verbs, args.dataset, model_name)
    if args.dataset == 'gtobj365_coco' and not os.path.exists(f_name):

        obj_verb_map_list1 = pickle.load(open(cfg.LOCAL_DATA + "/obj_hoi_map_new_{}/{}_{}_obj_hoi_map_list.pkl".format(args.num_verbs,
            'gtobj365_coco_1', model_name), 'rb'))
        obj_verb_map_list2 = pickle.load(open(
            cfg.LOCAL_DATA + "/obj_hoi_map_new_{}/{}_{}_obj_hoi_map_list.pkl".format(args.num_verbs,
                'gtobj365_coco_2', model_name), 'rb'))

        obj_verb_map_list = obj_verb_map_list1 + obj_verb_map_list2
        assert len(obj_verb_map_list1) + len(obj_verb_map_list2) == len(obj_verb_map_list)
    else:
        try:
            obj_verb_map_list = pickle.load(open(f_name, 'rb'))
        except:
            print(f_name, 'fail')
            return 0
    print(f_name)
    verb_preds_stat = {}
    num_objs_per_verb = {}
    verb_preds_stat_list = {}
    for i in range(81):
        verb_preds_stat[i] = [0] * num_verb
        num_objs_per_verb[i] = 0
        verb_preds_stat_list[i] = [0] * pred_num_hoi_classes
    import numpy as np
    obj_verb_all_list = []
    for item in obj_verb_map_list:
        if args.pred_type == 0:
            verb_preds_stat[item[0]] += (np.matmul(np.asarray(item[1]), verb_to_HO_matrix_preds.transpose())/ verb_to_HO_matrix_preds.sum(axis=-1))
        else:
            verb_preds_stat[item[0]] += item[-1]
        # item[-1][57] = 0
        # if not (item[-1] <= num_gt_verbs).all():
        #     print(np.argwhere(item[-1] > num_gt_verbs))
        # import ipdb;
        # ipdb.set_trace()
        # assert (item[-1] <= num_gt_verbs).all(), (item[-1], num_gt_verbs)
        # verb_preds_stat_list[item[0]] += np.asarray(item[1], np.float32)
        num_objs_per_verb[item[0]] += 1
        # verb_probs = item[1] / num_gt_verbs
        # for v_id in zero_verbs:
        #     verb_probs[v_id] = 0
        # obj_verb_all_list.append([item[0], verb_probs])
    # print(verb_preds_stat[0])
    verb_preds_stat.pop(0)
    for k in verb_preds_stat:
        if num_objs_per_verb[k] > 0:
            verb_preds_stat[k] = verb_preds_stat[k] / (np.asarray(num_gt_verbs) * num_objs_per_verb[k])
        else:
            verb_preds_stat[k] = np.zeros_like(num_gt_verbs)
        # verb_preds_stat[k] = np.matmul(verb_preds_stat_list[k], verb_to_HO_matrix_preds.transpose()) / (np.asarray(num_gt_verbs)*num_objs_per_verb[k])
        # print(k, verb_preds_stat[k])
        # import ipdb;ipdb.set_trace()
        if not model_name.__contains__('VCOCO'):
            # verb_preds_stat[k][57] = 0
            for v_id in zero_verbs:
                verb_preds_stat[k][v_id] = 0
    existing_hoi_pairs = []
    base_hoi_pairs = []
    for i in range(pred_num_hoi_classes):
        o_id = np.argmax(net.obj_to_HO_matrix_np[:, i])
        v_id = np.argmax(net.verb_to_HO_matrix_np[:, i])
        existing_hoi_pairs.append((v_id, o_id))
        if i < net.base_classes:
            base_hoi_pairs.append((v_id, o_id))
    affordance_probs = np.zeros([80, num_verb], np.float32)
    hoi_to_obj, hoi_to_verbs, obj_to_hoi, coco_to_hico_obj, coco80_to_hico_obj, hico_to_coco_obj, id_vb, id_obj, id_hoi = get_hoi_convert_dicts()
    for k in verb_preds_stat:
        if not model_name.__contains__('VCOCO'):
            affordance_probs[coco80_to_hico_obj[k]] = verb_preds_stat[k]
        else:
            affordance_probs[k-1] = verb_preds_stat[k]
    affordance_probs_new = affordance_probs
    affordance_probs = affordance_probs.reshape(-1)
    sorted_index = affordance_probs.argsort()[::-1]
    orig_existing_list = []
    for i in range(num_hoi_classes):
        # verb_to_HO_matrix, obj_to_HO_matrix
        o_id = np.argmax(obj_to_HO_matrix[:, i])
        v_id = np.argmax(verb_to_HO_matrix[:, i])
        orig_existing_list.append((v_id, o_id))

    num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs = obtain_config(model_name)
    if model_name.__contains__('VCOCO'):
        hoi_to_obj, hoi_to_verbs, verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(verb_class_num=verb_class_num)
    else:
        id_vb, id_obj, id_hoi, hoi_to_obj, hoi_to_verbs = get_id_convert_dicts()



    hico_id_pairs = []
    zs_id_pairs = []
    for i in range(num_classes):
        hico_id_pairs.append((hoi_to_verbs[i], hoi_to_obj[i]))
    af_new = 0.
    mask = np.zeros([verb_class_num, obj_class_num], np.float32)
    for v, o in hico_id_pairs:
        mask[v][o] = 1.
    affordance_probs_new= affordance_probs_new.transpose()
    print('save to ', DATA_DIR + '/afford/' + model_name.replace('/', '_')+'ATL_{}'.format(args.pred_type)+ ".npy")
    np.save(DATA_DIR + '/afford/' + model_name.replace('/', '_')+'ATL_{}'.format(args.pred_type)+ ".npy", affordance_probs_new)
    # import ipdb;ipdb.set_trace()
    ap_new = cal_ap(gt_labels, affordance_probs_new, mask)
    ap_all = cal_ap(gt_labels, affordance_probs_new, np.zeros([verb_class_num, obj_class_num], np.float32))
    gt_labels_known = np.zeros([verb_class_num, obj_class_num], np.float32)
    for v, o in hico_id_pairs:
        gt_labels_known[v][o] = 1.
    ap_all_known = cal_ap(gt_labels_known, affordance_probs_new, np.zeros([verb_class_num, obj_class_num], np.float32))

    for v, o in hico_id_pairs:
        affordance_probs_new[v][o] = 1. + np.max(affordance_probs_new)
        gt_labels_known[v][o] = 1.
    ap_all_fix = cal_ap(gt_labels, affordance_probs_new, np.zeros([verb_class_num, obj_class_num], np.float32))
    print(ap_new, ap_all, ap_all_fix, ap_all_known)

    f = open(cfg.LOCAL_DATA + '/hico_concepts.txt', 'a')
    f.write('{} {}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('ATL n', model_name + args.dataset, ap_new, ap_all, ap_all_fix, ap_all_known))
    f.close()

    return ap_new

if __name__ == '__main__':
    args = parse_args()
    obj = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
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

    # from pycocotools.coco import COCO
    # coco_annotation_path = cfg.LOCAL_DATA + '/Data/v-coco/coco/annotations/instances_train2014.json'
    # coco = COCO(coco_annotation_path)
    # cats = coco.loadCats(coco.getCatIds())
    coco_annos_map = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}
    coco_id_map_90_2_80 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27:
    24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46
    , 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68,
    79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

    # [0, 4, 7, 8, 10, 12, 14, 15, 17, 21, 23, 25, 26, 36, 37, 38, 39, 41, 43, 45, 48, 49, 52, 54, 57, 58, 59, 62, 64, 65, 66, 67, 73, 75, 76, 85, 87, 89, 93, 94, 96, 98, 99, 104, 110, 111, 112, 114]

    # [20,       53,          182,                 171,          365,         220,     334,     352,      29,      216,    23,       183,          300,      225,          282]
    # ['glove', 'microphone', 'american football', 'strawberry', 'flashlight', 'tape', 'baozi', 'durian', 'boots', 'ship', 'flower', 'basketball', 'cheese', 'watermelon', 'camel']
    # obj365 = {224: 'scale', 220: 'tape', 217: 'chicken', 244: 'hurdle', 354: 'game board', 334: 'baozi', 360: 'target', 26: 'plants pot/vase', 209: 'toothbrush', 190: 'projector', 300: 'cheese', 166: 'candy', 352: 'durian', 279: 'dumbbell', 136: 'gas stove', 335: 'lion', 251: 'french fries', 27: 'bench', 83: 'power outlet', 58: 'faucet', 25: 'storage box', 330: 'crab', 237: 'helicopter', 362: 'chainsaw', 288: 'antelope', 280: 'hamimelon', 294: 'jellyfish', 200: 'kettle', 215: 'marker', 204: 'clutch', 283: 'lettuce', 138: 'toilet', 115: 'oven', 170: 'baseball', 85: 'drum', 88: 'hanger', 236: 'toaster', 22: 'bracelet', 261: 'cherry', 159: 'tissue ', 225: 'watermelon', 183: 'basketball', 128: 'cleaning products', 123: 'tent', 188: 'fire hydrant', 81: 'truck', 304: 'rice cooker', 331: 'microscope', 262: 'tablet', 73: 'stuffed animal', 228: 'golf ball', 247: 'CD', 273: 'eggplant', 44: 'bowl', 12: 'desk', 351: 'eagle', 43: 'slippers', 252: 'horn', 40: 'carpet', 234: 'notepaper', 232: 'peach', 346: 'saw', 144: 'surfboard', 210: 'facial cleanser', 265: 'corn', 169: 'folder', 214: 'violin', 64: 'watch', 10: 'glasses', 124: 'shampoo/shower gel', 131: 'pizza', 357: 'asparagus', 295: 'mushroom', 322: 'steak', 178: 'suitcase', 347: 'table tennis  paddle', 211: 'mango', 29: 'boots', 56: 'necklace', 327: 'noodles', 272: 'volleyball', 141: 'baseball bat', 264: 'nuts', 139: 'stroller', 155: 'pumpkin', 171: 'strawberry', 181: 'pear', 111: 'luggage', 54: 'sandals', 150: 'liquid soap', 13: 'handbag', 365: 'flashlight', 291: 'trombone', 116: 'remote', 140: 'shovel', 180: 'ladder', 74: 'cake', 292: 'pomegranate', 84: 'clock', 162: 'vent', 104: 'cymbal', 364: 'iron', 348: 'okra', 359: 'pasta', 126: 'lantern', 269: 'broom', 192: 'fire extinguisher', 177: 'snowboard', 277: 'rice', 245: 'swing', 82: 'cow', 63: 'van', 305: 'tuba', 15: 'book', 249: 'swan', 5: 'lamp', 303: 'race car', 213: 'egg', 253: 'avocado', 92: 'guitar', 246: 'radio', 2: 'sneakers', 342: 'eraser', 320: 'measuring cup', 312: 'sushi', 212: 'deer', 318: 'parrot', 168: 'scissors', 102: 'balloon', 317: 'tortoise/turtle', 285: 'meat balls', 148: 'cat', 315: 'electric drill', 341: 'comb', 191: 'sausage', 223: 'bar soap', 201: 'hamburger', 174: 'pepper', 227: 'router/modem', 316: 'spring rolls', 182: 'american football', 299: 'egg tart', 278: 'tape measure/ruler', 109: 'banana', 146: 'gun', 187: 'billiards', 11: 'picture/frame', 118: 'paper towel', 87: 'bus', 284: 'goldfish', 133: 'computer box', 21: 'potted plant', 216: 'ship', 356: 'ambulance', 99: 'dog', 286: 'medal', 298: 'butterfly', 308: 'hair dryer', 268: 'globe', 355: 'french horn', 275: 'board eraser', 94: 'tea pot', 106: 'telephone', 328: 'mop', 137: 'broccoli', 311: 'dolphin', 3: 'chair', 4: 'hat', 96: 'tripod', 51: 'traffic light', 208: 'hot dog', 90: 'pot/pan', 9: 'car', 30: 'dining table', 306: 'crosswalk sign', 121: 'tomato', 45: 'barrel/bucket', 161: 'washing machine', 337: 'polar bear', 49: 'tie', 350: 'monkey', 238: 'green beans', 203: 'cucumber', 163: 'cookies', 47: 'suv', 239: 'brush', 160: 'carrot', 165: 'tennis racket', 17: 'helmet', 66: 'sink', 36: 'stool', 23: 'flower', 157: 'radiator', 260: 'fishing rod', 147: 'Life saver', 338: 'lighter', 60: 'bread', 326: 'radish', 1: 'human', 93: 'traffic cone', 78: 'knife', 179: 'grapes', 79: 'cellphone', 274: 'trophy', 313: 'urinal', 8: 'cup', 185: 'paint brush', 105: 'mouse', 113: 'soccer', 164: 'cutting/chopping board', 221: 'wheelchair', 156: 'Accordion/keyboard/piano', 189: 'goose', 336: 'red cabbage', 16: 'plate', 254: 'saxophone', 77: 'laptop', 194: 'facial mask', 218: 'onion', 75: 'motorbike/motorcycle', 55: 'canned', 363: 'lobster', 135: 'toiletries', 242: 'earphone', 33: 'flag', 333: 'Bread/bun', 255: 'trumpet', 248: 'parking meter', 250: 'garlic', 143: 'skateboard', 198: 'pie', 332: 'barbell', 329: 'yak', 281: 'stapler', 130: 'tangerine', 151: 'zebra', 70: 'traffic sign', 6: 'bottle', 361: 'hotair balloon', 129: 'sailboat', 325: 'llama', 101: 'blackboard/whiteboard', 175: 'coffee machine', 319: 'flute', 345: 'pencil case', 219: 'ice cream', 65: 'combine with bowl', 132: 'kite', 53: 'microphone', 86: 'fork', 358: 'hoverboard', 205: 'blender', 167: 'skating and skiing shoes', 89: 'nightstand', 287: 'toothpaste', 323: 'poker card', 98: 'fan', 108: 'orange', 196: 'chopsticks', 302: 'pig', 176: 'bathtub', 20: 'glove', 202: 'golf club', 119: 'refrigerator', 290: 'rickshaw', 72: 'candle', 57: 'mirror', 142: 'microwave', 158: 'converter', 110: 'airplane', 149: 'lemon', 125: 'head phone', 235: 'tricycle', 259: 'bear', 37: 'backpack', 69: 'apple', 114: 'trolley', 206: 'tong', 307: 'papaya', 233: 'cello', 282: 'camel', 324: 'binoculars', 226: 'cabbage', 31: 'umbrella', 241: 'cigar', 301: 'pomelo', 7: 'cabinet/shelf', 95: 'keyboard', 67: 'horse', 152: 'duck', 117: 'combine with glove', 229: 'pine apple', 184: 'potato', 103: 'air conditioner', 270: 'pliers', 231: 'fire truck', 97: 'hockey stick', 134: 'elephant', 153: 'sports car', 48: 'toy', 339: 'mangosteen', 353: 'rabbit', 59: 'bicycle', 154: 'giraffe', 267: 'screwdriver', 100: 'spoon', 91: 'sheep', 266: 'key', 28: 'wine glass', 297: 'treadmill', 193: 'extension cord', 289: 'shrimp', 62: 'ring', 32: 'boat', 263: 'green vegetables', 46: 'coffee table', 343: 'pitaya', 321: 'shark', 41: 'basket', 76: 'wild bird', 240: 'carriage', 207: 'slide', 68: 'fish', 199: 'frisbee', 271: 'hammer', 186: 'printer', 222: 'plum', 42: 'towel/napkin', 71: 'camera', 34: 'speaker', 107: 'pickup truck', 61: 'high heels', 172: 'bow tie', 173: 'pigeon', 293: 'coconut', 122: 'machinery vehicle', 38: 'sofa', 50: 'bed', 195: 'tennis ball', 276: 'dates', 14: 'street lights', 80: 'paddle', 296: 'calculator', 349: 'starfish', 310: 'chips', 120: 'train', 258: 'kiwi fruit', 39: 'belt', 24: 'monitor', 112: 'skis', 18: 'leather shoes', 256: 'sandwich', 197: 'Electronic stove and gas stove', 243: 'penguin', 145: 'surveillance camera', 257: 'cue', 344: 'scallop', 309: 'green onion', 340: 'seal', 230: 'crane', 314: 'donkey', 52: 'pen/pencil', 127: 'donut', 19: 'pillow', 35: 'trash bin/can'}
    obj365_set_list = { 20: [8, 9, 37, 51, 68, 115], # glove
                        53: [9, 37, 51, 68], # microphone
                        182: [3, 9, 10, 37, 45, 51, 68, 105], # american football
                        171:[8, 24, 37, 51, 57], # strawberry
                        365: [8, 37, 51, 68], # flashlight
                        220: [8, 37, 51, 68], # tape
                        334: [8, 24, 37, 51, 68], # baozi
                        352: [8, 37, 51, 68], # durian
                        29: [8, 37, 51, 68, 115,], # boots
                        216: [1, 5], # ship
                        23: [8, 37, 39, 51, 68], # flower
                        183: [3, 37, 45, 51, 68, 105, ], # basketball
                        # 300: [8, 16, 24, 37, 51, 55, 68], # cheese
                        # 225: [8, 16, 24, 37, 51], # watermelon
                        # 282: [27, 37, 77, 88, 111, 112, 113], # camel
                        # 335: [27, 37, 77, 88, 111, 112, 113] # lion
                        #51
    }
    if args.model.__contains__('*'):
        print('line', args.model)
        import re

        r = re.compile(args.model)

        import glob


        tmp = glob.glob(cfg.LOCAL_DATA + "/obj_hoi_map_new1_{}/*".format(args.num_verbs))
        print(cfg.LOCAL_DATA + "/obj_hoi_map_new1_{}/".format(args.num_verbs))
        tmp.sort(key=os.path.getmtime, reverse=False)
        tmp = [item.split('/')[-1] for item in tmp]
        file_arr = list(filter(r.match, tmp))

        file_arr = [item for item in file_arr if not item.__contains__('VCOCO')]
        print(file_arr)
        for i, model_name in enumerate(file_arr):
            print(model_name)
            if model_name.startswith(args.dataset):
                try:
                    model_name = model_name.replace(args.dataset+'_', '')
                    model_name = model_name.replace('_obj_hoi_map_list.pkl', '')
                    # stat_concept_result(model_name)
                    import multiprocessing
                    p = multiprocessing.Process(target=stat_concept_result, args=(model_name, ))
                    p.start()
                    p.join()
                except Exception as e:
                    print(e)
    else:
        stat_concept_result(args.model)
    # import ipdb; ipdb.set_trace()
    # ddd = np.zeros(80, 117)
    # for i in range(0,80):
    #     ddd[i] = np.sum([item[1] for item in obj_verb_all_list if item[0] == hico_to_coco_obj[i]+1], axis=0) / num_objs_per_verb[hico_to_coco_obj[i]+1]
    # ddd = ddd.reshape(-1)
    # ddd = ddd.argsort()[::-1]
    # for idx in sorted_index:
    #     if affordance_probs[idx] == 0:
    #         continue
    #     o_id = idx // 117
    #     v_id = (idx % 117)
    #     if (v_id, o_id) in existing_hoi_pairs:
    #         print('yes', v_id, o_id, affordance_probs[idx], id_vb[v_id], id_obj[o_id])
    #     elif (v_id, o_id) in orig_existing_list:
    #         print('orig', v_id, o_id, affordance_probs[idx], id_vb[v_id], id_obj[o_id])
    #     else:
    #         print(v_id, o_id, affordance_probs[idx], id_vb[v_id], id_obj[o_id])
