from scripts import get_id_convert_dicts, get_convert_matrix_coco3
from scripts import _init_paths
from scripts.analysis import obtain_config
from ult.ult import get_zero_shot_type, get_unseen_index
from sklearn.metrics import average_precision_score


def cal_ap(gt_labels, affordance_probs_new, mask):

    # ap = average_precision_score(gt_labels.reshape(-1), affordance_stat_tmp1.reshape(-1))
    # print(ap)
    # exit()
    #
    mask = mask.reshape([-1])
    gt_labels = gt_labels.reshape(-1).tolist()
    affordance_probs_new = affordance_probs_new.reshape(-1).tolist()
    assert len(mask) == len(gt_labels) == len(affordance_probs_new), (len(mask), len(gt_labels), len(affordance_probs_new))
    gt_labels = [gt_labels[i] for i in range(len(mask)) if mask[i] == 0]
    affordance_probs_new = [affordance_probs_new[i] for i in range(len(mask)) if mask[i] == 0]

    ap = average_precision_score(gt_labels, affordance_probs_new)
    return ap


def stat_concept_result(file_name, gt_labels, concept_gt_pairs, num_classes=600, verb_class_num=117, obj_class_num=80):
    if not file_name.__contains__('xrandom'):
        model_name = file_name.split('Weights/')[-1]
        reader = tf.train.NewCheckpointReader(file_name)
        affordance_stat = reader.get_tensor('affordance_stat')
        affordance_count = reader.get_tensor('affordance_count')
    else:
        model_name = file_name
        import numpy as np
        affordance_stat = np.random.random([verb_class_num, obj_class_num])
    # assert affordance_stat.max() <= 1., file_name
    if affordance_stat.max() >= 1.:
        print('fail=======', file_name, affordance_stat.max())
    import numpy as np
    if file_name.__contains__('VCOCO'):
        hoi_to_obj, hoi_to_verbs, verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(verb_class_num=verb_class_num)
    else:
        id_vb, id_obj, id_hoi, hoi_to_obj, hoi_to_verbs = get_id_convert_dicts()
    zero_shot_type = get_zero_shot_type(file_name)
    unseen_idx = get_unseen_index(zero_shot_type)
    hico_id_pairs = []
    zs_id_pairs = []
    for i in range(num_classes):
        hico_id_pairs.append((hoi_to_verbs[i], hoi_to_obj[i]))
        if i in unseen_idx:
            zs_id_pairs.append((hoi_to_verbs[i], hoi_to_obj[i]))
    affordance_stat_tmp = affordance_stat.reshape(-1)
    mask = np.zeros([verb_class_num, obj_class_num], np.float32)
    for v, o in hico_id_pairs:
        mask[v][o] = 1.

    ap_new = cal_ap(gt_labels, affordance_stat_tmp, mask)
    if not os.path.exists(DATA_DIR + '/afford/'):
        os.makedirs(DATA_DIR + '/afford/')
    np.save(DATA_DIR + '/afford/'+model_name.replace('/', '_')+".npy", affordance_stat_tmp)

    ap_all = cal_ap(gt_labels, affordance_stat_tmp, np.zeros([verb_class_num, obj_class_num], np.float32))
    print(affordance_stat_tmp.max(), np.max(affordance_stat_tmp))
    affordance_stat_tmp = affordance_stat_tmp.reshape([verb_class_num, obj_class_num])
    max_v = np.max(affordance_stat_tmp)
    gt_labels_known = np.zeros([verb_class_num, obj_class_num], np.float32)
    for v, o in hico_id_pairs:
        gt_labels_known[v][o] = 1.
    ap_all_know = cal_ap(gt_labels_known, affordance_stat_tmp, np.zeros([verb_class_num, obj_class_num], np.float32))

    gt_labels_zs = np.zeros([verb_class_num, obj_class_num], np.float32)
    for v, o in zs_id_pairs:
        gt_labels_zs[v][o] = 1.
    ap_all_zs = cal_ap(gt_labels_zs, affordance_stat_tmp, np.zeros([verb_class_num, obj_class_num], np.float32))

    for v, o in hico_id_pairs:
        affordance_stat_tmp[v][o] = 100. + max_v
    # print(affordance_stat_tmp.max())
    ap_all_fix = cal_ap(gt_labels, affordance_stat_tmp, np.zeros([verb_class_num, obj_class_num], np.float32))
    # print(affordance_stat_tmp.max())
    f = open(DATA_DIR + '/hico_concepts.txt', 'a')
    print_result = '{} {}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n'.format('analysis', model_name, ap_new, ap_all, ap_all_fix, ap_all_know, ap_all_zs)
    f.write(print_result)
    f.close()
    print('unknown: {}, ap_all_known'.format(ap_new, ap_all_know))

    return ap_new, ap_all_know


if __name__ == "__main__":
    import tensorflow as tf

    import os

    DATA_DIR = './'
    import sys

    file_name = DATA_DIR + '/Weights/VCL_R_union_batch_large2_ml5_def1_ICL_zs7_vloss2_l2_rew2_aug5_3_x5new_res101_affordance_4/HOI_iter_120000.ckpt'
    if len(sys.argv) > 1:
        file_name = sys.argv[1]

    # ap_new_list = []
    # ap_known_list = []
    # for i in range(10):
    #     num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, _ = obtain_config('xrandom')
    #     ap_new, ap_all_know = stat_concept_result('xrandom', gt_labels, concept_gt_pairs, num_classes, verb_class_num, obj_class_num)
    #     ap_new_list.append(ap_new)
    #     ap_known_list.append(ap_all_know)
    # print(sum(ap_new_list)/len(ap_new_list), sum(ap_known_list)/len(ap_known_list))
    # ap_new_list = []
    # ap_known_list = []
    # for i in range(10):
    #     num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, _ = obtain_config(
    #         'xrandom_VCOCO_CL_21')
    #     ap_new, ap_all_know = stat_concept_result('xrandom_VCOCO_CL_21', gt_labels, concept_gt_pairs, num_classes,
    #                                               verb_class_num,
    #                                               obj_class_num)
    #     ap_new_list.append(ap_new)
    #     ap_known_list.append(ap_all_know)
    # print(sum(ap_new_list) / len(ap_new_list), sum(ap_known_list) / len(ap_known_list))
    # exit()
    num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, gt_known_labels = obtain_config(file_name)

    if file_name.__contains__('*'):
        print('line', file_name)
        import re

        r = re.compile(file_name)

        import glob

        tmp = glob.glob(DATA_DIR + '/Weights/*')
        tmp.sort(key=os.path.getmtime, reverse=False)
        # tmp = tmp[:40]
        # tmp = tmp[-40:]
        # print(tmp)
        # tmp = list(set(os.listdir(DATA_DIR + '/Weights/')))
        tmp = [item.split('/')[-1] for item in tmp]
        model_arr = list(filter(r.match, tmp))
        # model_arr = sorted(model_arr)

        # print(cfg.LOCAL_DATA)
        # model_arr = list(filter(r.match, list(os.listdir(cfg.LOCAL_DATA + '/Weights/'))))
        # if not file_name.__contains__('VCOCO'):
        #     model_arr = [item for item in model_arr if not item.__contains__('VCOCO')]
        num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, _ = obtain_config('xrandom')
        stat_concept_result('xrandom', gt_labels, concept_gt_pairs, num_classes, verb_class_num, obj_class_num)

        num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, _ = obtain_config('xrandom_VCOCO_CL_21')
        stat_concept_result('xrandom_VCOCO_CL_21', gt_labels, concept_gt_pairs, num_classes, verb_class_num, obj_class_num)
        for i, model in enumerate(model_arr):
            if model.__contains__('zs4'):
                stride = 5000
            elif model.__contains__('zs2'):
                stride = 40000
            else:
                stride = 5000
            for index in list(range(stride, 6000001, stride)) + list(range(43273, 6000001, 43273)):
                import os

                # print(cfg.LOCAL_DATA + '/Weights/' + model + '/HOI_iter_' + str(index) + '.ckpt.index')
                weight_name = DATA_DIR + '/Weights/' + model + '/HOI_iter_' + str(index) + '.ckpt'
                if os.path.exists(weight_name + '.index'):
                    print(weight_name)
                    num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, _ = obtain_config(weight_name)
                    stat_concept_result(weight_name, gt_labels, concept_gt_pairs, num_classes, verb_class_num, obj_class_num)
                    continue
    else:
        stat_concept_result(file_name, gt_labels, concept_gt_pairs, num_classes, verb_class_num, obj_class_num)
    # import ipdb;ipdb.set_trace()
    # os.system('python scripts/analysis/update_concepts_results.py')