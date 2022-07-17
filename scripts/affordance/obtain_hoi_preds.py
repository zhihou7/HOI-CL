# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os

os.environ['DATASET'] = 'HICO'

import tensorflow as tf
import numpy as np
import argparse
import pickle

from ult.timer import Timer
from ult.config import cfg
from networks.tools import get_convert_matrix as get_cooccurence_matrix
from networks.tools import get_convert_matrix_coco3 as get_cooccurence_matrix_coco3

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=160000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='ATL_union_multi_atl_ml5_l05_t5_def2_aug5_new_VCOCO_coco_CL_21', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0., type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0., type=float)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset type: gthico',
                        default='gthico', type=str)
    parser.add_argument('--no-save', action='store_true', dest='no_save')
    parser.add_argument('--num', dest='num', default=100,
                        type=int) # HOI-COCO has less than 10000 verb features.
    parser.add_argument('--num_verbs', dest='num_verbs', default=100,
                        type=int)
    parser.add_argument('--pred_type', dest='pred_type', default=0,
                        type=int)
    parser.add_argument('--incre_classes', dest='incre_classes',
                        help='Human threshold',
                        default=None, type=str)
    args = parser.parse_args()
    return args

def switch_checkpoint_path(model_checkpoint_path):
    head = model_checkpoint_path.split('Weights')[0]
    model_checkpoint_path = model_checkpoint_path.replace(head, cfg.LOCAL_DATA +'/')
    return model_checkpoint_path


if __name__ == '__main__':

    args = parse_args()
    print(args)
    # test detections result

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    import os
    if not os.path.exists(weight + '.index'):
        weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight )
    # init session

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    if not os.path.exists(cfg.LOCAL_DATA + '/obj_hoi_map_new'):
        os.makedirs(cfg.LOCAL_DATA + '/obj_hoi_map_new')
    if not args.model.__contains__('VCOCO'):
        obj_hoi_map_file = cfg.LOCAL_DATA + '/obj_hoi_map_new/{}_{}_obj_hoi_map_list.pkl'.format(args.dataset, args.model)
    else:
        obj_hoi_map_file = cfg.LOCAL_DATA + '/obj_hoi_map_new/{}_{}_obj_hoi_map_list.pkl'.format(args.dataset,
                                                                                                  args.model)
    # if os.path.exists(obj_hoi_map_file):
    #     print('exist', obj_hoi_map_file)
    #     exit(0)

    if not os.path.exists(cfg.LOCAL_DATA + '/feats'):
        os.makedirs(cfg.LOCAL_DATA + '/feats')
    print('Pre-trained weights loaded.')

    if not args.model.__contains__('VCOCO'):
        feats = pickle.load(open(cfg.LOCAL_DATA + '/feats/'+'{}_train_HOI_verb_feats.pkl'.format(args.model), 'rb'))
    else:
        feats = pickle.load(open(cfg.LOCAL_DATA + '/feats/' + '{}_train_HOI_verb_feats.pkl'.format(args.model), 'rb'))
    verb_feats = feats['V_list']
    action_list = feats['A_list']

    num_hoi_classes = 600
    if not args.model.__contains__('VCOCO'):
        verb_to_HO_matrix, obj_to_HO_matrix = get_cooccurence_matrix()
        num_hoi_classes = 600
        # choose verbs.
    else:
        # verb_label_list = []
        verb_to_HO_matrix, obj_to_HO_matrix = get_cooccurence_matrix_coco3(21, 80)
        # for i in range(len(verb_feats)):
            # hoi = action_list[i]
            # verb_labels = np.matmul(hoi, verb_to_HO_matrix.transpose())
            # verb_label_list.append(np.argwhere(verb_labels).reshape(-1).tolist())
        num_hoi_classes = 222
    # obj_feats = feats['O_list']
    coco_type = ''
    real_dataset_name = args.dataset if not args.dataset.startswith('gtobj365_coco') else 'gtobj365_coco'
    coco_type = '_' + real_dataset_name

    detector_type = 'res101'
    if not args.model.__contains__('res101'):
        detector_type = 'res50'
    print(cfg.LOCAL_DATA + '/{}_{}_HOI{}_obj_feats.pkl'.format(args.model, detector_type, coco_type))
    obj_feats_list = pickle.load(open(cfg.LOCAL_DATA + '/feats/{}_{}_HOI{}_obj_feats.pkl'.format(args.model, detector_type, coco_type), 'rb'))

    # this is for speeding up the evaluation on object365.
    if args.dataset == 'gtobj365_coco_1':
        print('equal')
        obj_feats_list = obj_feats_list[:len(obj_feats_list)//2]
    elif args.dataset == 'gtobj365_coco_2':
        obj_feats_list = obj_feats_list[len(obj_feats_list)//2:]

    # if args.dataset.startswith('fres50_1_'):
    #     #for si in range(10):
    #     si = int(args.dataset[len('fres50_1_'):])
    #     if si == 9:
    #         obj_feats_list = obj_feats_list[si * len(obj_feats_list) // 10:]
    #     else:
    #         obj_feats_list = obj_feats_list[si * len(obj_feats_list) // 10: (si + 1) * len(obj_feats_list) // 10]
    # [num, new_o_box, new_o_cls, new_o_score, o_feats[-1], _image_id]
    print(len(verb_feats))
    # assert args.num < len(verb_feats)
    b_size = 10000 if args.num >= 10000 else args.num # This command is useless because the two dataset HICO-DEt and HOI-COCO have less 10000 verbs.
    b_size = len(verb_feats) if b_size >= len(verb_feats) else b_size

    tfconfig = tf.ConfigProto(device_count={"CPU": 12},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.2
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)


    # Generate_HICO_detection(output_file, HICO_dir)
    if args.model.__contains__('ICL'):
        # incremental continual learning
        os.environ['DATASET'] = 'HICO_res101_icl'
        from networks.HOI_Concept_Discovery import HOIICLNet

        incremental_class_pairs = [[]]
        if args.incre_classes is not None and os.path.exists(args.incre_classes):
            import json
            incremental_class_pairs = json.load(open(args.incre_classes))
        net = HOIICLNet(model_name=args.model, task_id=1, incremental_class_pairs=incremental_class_pairs)
        net.obj_to_HO_matrix = net.incre_obj_to_HOI
        verb_to_HO_matrix_preds = net.verb_to_HO_matrix_np
        pred_num_hoi_classes = net.sum_num_classes
    elif args.model.__contains__('VERB'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import DisentanglingNet

        net = DisentanglingNet(model_name=args.model)
        pred_num_hoi_classes = net.verb_num_classes
    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    elif args.model.__contains__('VCOCO'):
        os.environ['DATASET'] = 'VCOCO1'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)

    real_dataset_name = args.dataset if not args.dataset.startswith('gtobj365_coco') else 'gtobj365_coco'
    output_file = cfg.LOCAL_DATA + '/feats/' + str(args.iteration) + '_' + args.model + '_{}_{}.pkl'.format(args.dataset, args.num)
    print(output_file)
    # if os.path.exists(output_file):
    #     exit()
    net.create_architecture(False)

    # assert not net.model_name.__contains__('_base')

    # o_feats_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    # v_feats_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    # o_cls = tf.placeholder(dtype=tf.int32, shape=[])
    # o_boxes = tf.placeholder(dtype=tf.float32, shape=[4])
    # o_score = tf.placeholder(dtype=tf.float32, shape=[])


    def feature_generator():
        for j in range(len(obj_feats_list)):
            # object_thres
            if obj_feats_list[j][3] < args.object_thres:
                continue
            if args.dataset == 'gtobj365' and int(obj_feats_list[j][2]) not in [20, 53, 182, 171, 365, 220, 334, 352,
                                                                                29, 216, 23, 183, 300, 225, 282, 335]:
                continue
            if args.dataset == 'gtobj365':
                assert int(obj_feats_list[j][2]) in [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183, 300, 225,
                                                     282, 335], obj_feats_list[j][2]

            # for jj in range(num_hoi_classes):
            #     obj_list_map[jj] = 0
            for i in range(0, min(args.num, len(verb_feats)), b_size):
                o_feats = [obj_feats_list[j][4]]
                o_feats = np.tile(o_feats, [b_size, 1])
                _t['im_detect'].tic()
                v_feats = np.asarray(verb_feats[i:i + b_size])

                new_o_box = obj_feats_list[j][1]
                new_o_cls = obj_feats_list[j][2]
                new_o_score = obj_feats_list[j][3]
                _image_id = obj_feats_list[j][-1]
                hoi_gt_action = np.asarray(action_list[i:i + b_size])
                yield o_feats, v_feats, new_o_box, new_o_cls, new_o_score, _image_id, hoi_gt_action


    # o_feats_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    # v_feats_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    dataset = tf.data.Dataset.from_generator(feature_generator,
                                             output_types=(tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32),
                                             output_shapes=(
                                             tf.TensorShape([None, 2048]),
                                             tf.TensorShape([None, 2048]),
                                             tf.TensorShape([4]),
                                             tf.TensorShape([]),
                                             tf.TensorShape([]),
                                             tf.TensorShape([]),
                                             tf.TensorShape([None, num_hoi_classes]),
                                                 )
                                             )
    dataset = dataset.prefetch(100)

    iterator = dataset.make_one_shot_iterator()
    o_feats_ph, v_feats_ph, o_box, o_cls, o_score, image_id, hoi_gt_action = iterator.get_next()

    fc7_vo = net.head_to_tail_ho(o_feats_ph, v_feats_ph, None, None, False, 'fc_HO')
    net.region_classification_ho(fc7_vo, False, tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                 'classification', nameprefix='merge_')

    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)

    detection = {}

    last_img_id = -1
    count = 0
    _t = {'im_detect': Timer(), 'misc': Timer()}
    _t['im_detect'].tic()

    obj_list_map_list = []
    pred_name = "merge_cls_prob_verbs"
    if 'merge_cls_prob_verbs_f' in net.predictions:
        pred_name = 'merge_cls_prob_verbs_f'
    if args.model.__contains__('VERB'):
        pred_name = 'merge_cls_prob_verbs_VERB'
    while True:

        _t['im_detect'].tic()
        try:
            merge_probs, new_o_box, new_o_cls, new_o_score, _image_id, hoi_gt_action_list = sess.run(
                [net.predictions[pred_name], o_box, o_cls, o_score, image_id, hoi_gt_action])
        except tf.errors.OutOfRangeError:
            print('END')
            break

        # print(merge_probs.shape)
        # new_merge_probs = np.mean(merge_probs, axis=0)
        # print(merge_probs.shape, new_merge_probs.shape)
        # print([hoi_to_verbs[h] for h in np.where(new_merge_probs > 0.5)[0].tolist()], new_o_cls)
        if args.pred_type == 0 or  args.pred_type == 1:
            hoi_preds = np.asarray(merge_probs > 0.5, np.float32)
            if hoi_preds.shape[-1] == pred_num_hoi_classes and not args.model.__contains__('VERB'):
                verb_preds = np.matmul(hoi_preds, verb_to_HO_matrix_preds.transpose())
                verb_preds = np.asarray(verb_preds > 0, np.float32)
            else:
                # 117
                verb_preds = hoi_preds
            verb_gt_labels = np.matmul(hoi_gt_action_list, verb_to_HO_matrix.transpose())
            right_verbs = np.sum(np.multiply(verb_preds, verb_gt_labels), axis=1) > 0.

            hois_right = np.sum(hoi_preds[right_verbs], axis=0)
            # print(right_verbs.shape, hois_right.shape, hoi_preds[np.argwhere(right_verbs)].shape)
            # assert len(hois_right) == pred_num_hoi_classes, (hois_right.shape, pred_num_hoi_classes, hoi_gt_action_list.shape)
            # print(hois_right)
            obj_list_map = hois_right.astype(np.int32)
            verb_list = np.sum(verb_preds.astype(np.int32), axis=0)
        elif args.pred_type == 2:
            hoi_preds = np.asarray(merge_probs > 0.5, np.float32)
            if merge_probs.shape[-1] == pred_num_hoi_classes and not args.model.__contains__('VERB'):
                verb_preds = np.matmul(merge_probs, verb_to_HO_matrix_preds.transpose()) / np.sum(verb_to_HO_matrix_preds, axis=1)
                # verb_preds = np.asarray(verb_preds > 0, np.float32)
            else:
                # 117
                verb_preds = merge_probs
            verb_gt_labels = (np.matmul(hoi_gt_action_list, verb_to_HO_matrix.transpose()) > 0).astype(np.float32)
            verb_preds = np.multiply(verb_preds, verb_gt_labels)
            # right_verbs = np.sum(np.multiply(verb_preds, verb_gt_labels), axis=1) > 0.

            # hois_right = np.sum(hoi_preds[right_verbs], axis=0)
            # print(right_verbs.shape, hois_right.shape, hoi_preds[np.argwhere(right_verbs)].shape)
            # assert len(hois_right) == pred_num_hoi_classes, (hois_right.shape, pred_num_hoi_classes, hoi_gt_action_list.shape)
            # print(hois_right)
            obj_list_map = [0]
            verb_list = np.sum(verb_preds, axis=0)
        # assert verb_gt_labels.
        _t['im_detect'].toc()
        count += 1

        # print(i, j)
        print('\rmodel: {} {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, args.dataset, count, len(obj_feats_list), _image_id,
                                                                       _t['im_detect'].average_time), end='',
              flush=True)
        # print(obj_list_map)
        # print(new_o_score)
        if args.dataset == 'gtobj365':
            assert int(new_o_cls) in [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183, 300, 225, 282, 335], new_o_cls
        obj_list_map_list.append([int(new_o_cls), obj_list_map, new_o_score, _image_id, verb_list])
        # if count > 1000:
        #     break
    pickle.dump(obj_list_map_list, open(obj_hoi_map_file, 'wb'))
    # TODO remove
    # if not args.no_save:
    #     pickle.dump(detection, open(output_file, "wb"))
    #     print(output_file)
    sess.close()
