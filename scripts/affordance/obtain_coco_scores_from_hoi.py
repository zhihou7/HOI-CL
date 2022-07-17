# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


os.environ['DATASET'] = 'HICO'

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb

from ult.timer import Timer
from ult.config import cfg
from networks.tools1 import get_convert_matrix, get_convert_matrix_coco3

# from models.test_HICO_pose_pattern_all_wise_pair import test_net, obtain_test_dataset, test_net_data_api1
from models.test_HICO import test_net, obtain_test_dataset, obtain_test_dataset1, test_net_data_api1, \
    obtain_test_dataset_with_coco, test_net_data_api1_for_coco


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=259638, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_R_union_batch_semi_l2_def1_epoch2_epic2_cosine5_s1_7_gc_gall_embloss_vloss2_var_gan_dax_xrew45_randso2_aug5_3_x5new_coco_res101_2', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0., type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    parser.add_argument('--debug', dest='debug',
                        help='Human threshold',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='Human threshold',
                        default='cocosub', type=str)
    parser.add_argument('--type', dest='test_type',
                        help='Human threshold',
                        default='res101', type=str)
    parser.add_argument('--no-save', action='store_true', dest='no_save')
    parser.add_argument('--num', dest='num', default=100,
                        type=int)
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
    from sys import version_info

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    import os
    if not os.path.exists(weight + '.index'):
        weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight )
    # init session

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    tfconfig = tf.ConfigProto(device_count={"CPU": 12},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.2
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # net = ResNet50(model_name=args.model)
    # net.create_architecture(False)
    #
    #
    # saver = tf.train.Saver()
    # saver.restore(sess, weight)
    #
    # print('Pre-trained weights loaded.')
    #
    # test_net(sess, net, Test_RCNN, output_file, args.object_thres, args.human_thres)
    # sess.close()

    # Generate_HICO_detection(output_file, HICO_dir)
    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import DisentanglingNet
        net = DisentanglingNet(model_name=args.model)
    elif args.model.__contains__('VCOCO'):
        os.environ['DATASET'] = 'VCOCO1'
        from networks.HOI import DisentanglingNet
        net = DisentanglingNet(model_name=args.model)
    else:
        from networks.HOI import DisentanglingNet
        net = DisentanglingNet(model_name=args.model)
        # if not args.model.__contains__('semi'):
        #     args.test_type = 'default'
        #     if args.model.__contains__('_pa'):
        #         args.test_type = 'pose'

    real_dataset_name = args.dataset if not args.dataset.startswith('gtobj365_coco') else 'gtobj365_coco'
    output_file = cfg.LOCAL_DATA + '/feats/' + str(args.iteration) + '_' + args.model + '_{}_{}.pkl'.format(args.dataset, args.num)
    print(output_file)
    # if os.path.exists(output_file):
    #     exit()
    net.create_architecture(False)

    # assert not net.model_name.__contains__('_base')

    o_feats_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    v_feats_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    # o_cls = tf.placeholder(dtype=tf.int32, shape=[])
    # o_boxes = tf.placeholder(dtype=tf.float32, shape=[4])
    # o_score = tf.placeholder(dtype=tf.float32, shape=[])

    fc7_vo = net.head_to_tail_ho(o_feats_ph, v_feats_ph, None, None, False, 'fc_HO')
    # fc7_vo = tf.Print(fc7_vo, [tf.shape(fc7_vo), tf.reduce_mean(fc7_vo)], 'sdfsdfasdfasdf')

    net.region_classification_ho(fc7_vo, True, tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                      'classification', nameprefix='merge_')

    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    detection = {}
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0
    _t['im_detect'].tic()
    feats = pickle.load(open(cfg.LOCAL_DATA + '/feats/'+'{}_train_HICO_verb_feats.pkl'.format(args.model), 'rb'))
    verb_feats = feats['V_list']
    action_list = feats['A_list']

    num_hoi_classes = 600
    if not args.model.__contains__('VCOCO'):
        new_verb_feats = []
        verb_label_list = []
        new_action_list = []
        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix()

        no_interactions = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107,
                           111, 129, 146, 160, 170, 174, 186, 194, 198,
                           208, 214, 224, 232, 235, 239, 243, 247, 252, 257,
                           264, 273, 283, 290, 295, 305, 313, 325, 330, 336,
                           342, 348, 352, 356, 363, 368, 376, 383, 389, 393,
                           397, 407, 414, 418, 429, 434, 438, 445, 449, 453,
                           463, 474, 483, 488, 502, 506, 516, 528, 533, 538,
                           546, 550, 558, 562, 567, 576, 584, 588, 595, 600]

        no_interactions = set([item - 1 for item in no_interactions])
        for i in range(len(verb_feats)):
            hoi = action_list[i]
            verb_labels = np.matmul(hoi, verb_to_HO_matrix.transpose())
            has_no_inter = len(set(np.argwhere(hoi).reshape(-1).tolist()).intersection(no_interactions)) > 0
            if has_no_inter:
                continue
            verb_label_list.append(np.argwhere(verb_labels).reshape(-1).tolist())
            new_verb_feats.append(verb_feats[i])
            new_action_list.append(action_list[i])

        action_list = new_action_list
        verb_feats = new_verb_feats
        num_hoi_classes = 600
    else:
        verb_label_list = []
        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(21, 80)
        for i in range(len(verb_feats)):
            hoi = action_list[i]
            verb_labels = np.matmul(hoi, verb_to_HO_matrix.transpose())
            verb_label_list.append(np.argwhere(verb_labels).reshape(-1).tolist())
        num_hoi_classes = 222
    # obj_feats = feats['O_list']
    coco_type = ''
    if args.dataset == 'coco_2017_val':
        coco_type = '_coco101val2017'
    else:
        real_dataset_name = args.dataset if not args.dataset.startswith('gtobj365_coco') else 'gtobj365_coco'
        coco_type = '_' + real_dataset_name

    detector_type = 'res101'
    if not args.model.__contains__('res101'):
        detector_type = 'res50'
    print(cfg.LOCAL_DATA + '/{}_{}_HOI{}_coco_feats.pkl'.format(args.model, detector_type, coco_type))
    obj_feats_list = pickle.load(open(cfg.LOCAL_DATA + '/feats/{}_{}_HOI{}_coco_feats.pkl'.format(args.model, detector_type, coco_type), 'rb'))

    if args.dataset == 'gtobj365_coco_1':
        print('equal')
        obj_feats_list = obj_feats_list[:len(obj_feats_list)//2]
    elif args.dataset == 'gtobj365_coco_2':
        obj_feats_list = obj_feats_list[len(obj_feats_list)//2:]
    # [num, new_o_box, new_o_cls, new_o_score, o_feats[-1], _image_id]
    print(len(verb_feats))
    # assert args.num < len(verb_feats)
    b_size = 10000 if args.num >= 10000 else args.num
    b_size = len(verb_feats) if b_size >= len(verb_feats) else b_size
    obj_list_map_list = []
    for j in range(len(obj_feats_list)):
        # object_thres
        if obj_feats_list[j][3] < args.object_thres:
            continue
        if args.dataset == 'gtobj365' and int(obj_feats_list[j][2]) not in [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183, 300, 225, 282, 335]:
            continue
        if args.dataset == 'gtobj365':
            assert int(obj_feats_list[j][2]) in [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183, 300, 225, 282, 335], obj_feats_list[j][2]
        obj_list_map = [0]*num_hoi_classes

        # for jj in range(num_hoi_classes):
        #     obj_list_map[jj] = 0
        for i in range(0, min(args.num, len(verb_feats)), b_size):
            o_feats = [obj_feats_list[j][4]]
            o_feats = np.tile(o_feats, [b_size, 1])
            _t['im_detect'].tic()
            v_feats = np.asarray(verb_feats[i:i + b_size])
            v_label_list = verb_label_list[i:i+b_size]
            # print(i, o_feats.shape, v_feats.shape)
            merge_probs, = sess.run(
                [net.predictions["merge_cls_prob_verbs"]],
                feed_dict={o_feats_ph: o_feats,
                           v_feats_ph: v_feats})
            new_o_box = obj_feats_list[j][1]
            new_o_cls = obj_feats_list[j][2]
            new_o_score = obj_feats_list[j][3]
            _image_id = obj_feats_list[j][-1]
            # print(merge_probs.shape)
            new_merge_probs = np.mean(merge_probs, axis=0)
            # print(merge_probs.shape, new_merge_probs.shape)
            # print([hoi_to_verbs[h] for h in np.where(new_merge_probs > 0.5)[0].tolist()], new_o_cls)
            hoi_preds = np.asarray(merge_probs > 0.5, np.float32)
            # verb_preds = np.matmul(hoi_preds, verb_to_HO_matrix.transpose())
            # verb_preds = np.asarray(verb_preds > 0, np.float32)

            # hoi_gt_action_list = np.asarray(action_list[i:i + b_size])
            # verb_gt_labels = np.matmul(hoi_gt_action_list, verb_to_HO_matrix.transpose())
            # right_verbs = np.sum(np.multiply(verb_preds, verb_gt_labels), axis=1) > 0.

            hois_right = np.sum(hoi_preds, axis=0)
            # print(right_verbs.shape, hois_right.shape, hoi_preds[np.argwhere(right_verbs)].shape)
            assert len(hois_right) == num_hoi_classes, hois_right
            # print(hois_right)
            obj_list_map += hois_right.astype(np.int32)
            # for pi in range(len(merge_probs)):
            #     v_label_gt = v_label_list[pi]
            #     hoi_tmp = np.asarray(merge_probs[pi] > 0.5, np.float32)
            #
            #     verb_labels = np.matmul(hoi_tmp, verb_to_HO_matrix.transpose())
            #     pred_verbs = np.argwhere(verb_labels).reshape(-1).tolist()
            #
            #     for hi in np.where(merge_probs[pi] > 0.5)[0].tolist():
            #         obj_list_map[hi] += 1
                # import ipdb;ipdb.set_trace()

            # if not os.path.exists(
            #         '/opt/data/private/Code/detectron2/datasets/coco/val2017/' + (str(_image_id)).zfill(12) + '.jpg'):
            #     print('wrong ', _image_id)
            if not args.no_save:
                temp = [[len(merge_probs), new_o_box, new_o_cls, 0, 0, new_o_score, new_merge_probs]]
                if _image_id in detection:
                    if last_img_id == _image_id and tuple(detection[_image_id][-1][1].tolist()) == tuple(temp[0][1].tolist()):
                        detection[_image_id][-1][-1] = (detection[_image_id][-1][-1] * detection[_image_id][-1][0] + temp[0][
                            0] * temp[0][-1]) / (temp[0][0] + detection[_image_id][-1][0])
                        detection[_image_id][-1][0] = detection[_image_id][-1][0] + temp[0][0]
                    else:
                        detection[_image_id].extend(temp)
                    last_img_id = _image_id
                else:
                    detection[_image_id] = temp
                    last_img_id = _image_id

            _t['im_detect'].toc()
            count += 1

            # print(i, j)
            print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, len(obj_feats_list), _image_id,
                                                                           _t['im_detect'].average_time), end='',
                  flush=True)
        # print(obj_list_map)
        if args.dataset == 'gtobj365':
            assert int(obj_feats_list[j][2]) in [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183, 300, 225, 282, 335], obj_feats_list[j][2]
        obj_list_map_list.append([int(obj_feats_list[j][2]), obj_list_map])
        if j == 1000:
            pickle.dump(obj_list_map_list, open(cfg.LOCAL_DATA + '/obj_hoi_map2/{}_{}_obj_hoi_map_list.pkl'.format(args.dataset, net.model_name), 'wb'))
    pickle.dump(obj_list_map_list, open(cfg.LOCAL_DATA + '/obj_hoi_map2/{}_{}_obj_hoi_map_list.pkl'.format(args.dataset, net.model_name), 'wb'))
    # TODO remove
    if not args.no_save:
        pickle.dump(detection, open(output_file, "wb"))
        print(output_file)
    sess.close()
