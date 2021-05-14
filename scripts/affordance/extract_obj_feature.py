# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os

import tensorflow as tf
import argparse

from ult.timer import Timer
from ult.config import cfg
from models.test_HICO import obtain_test_dataset_with_obj
import numpy as np
tf.set_random_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=259638, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='ATL_union_multi_atl_ml5_l05_t5_def2_aug5_new_VCOCO_coco_CL_21', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0., type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    parser.add_argument('--type', dest='test_type',
                        help='Dataset type: gthico',
                        default='gthico', type=str)
    parser.add_argument('--num', dest='num',
                        type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)
    # test detections result

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    import os
    if not os.path.exists(weight + '.index'):
        weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print('weight:', weight)
    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight )
    detector_type = 'res101'
    if args.model.__contains__('VCOCO'):
        detector_type = 'res50'
    test_type_prefix = ''
    if len(args.test_type) > 0:
        test_type_prefix = '_'+args.test_type
    output_file = cfg.LOCAL_DATA + '/feats/' + args.model + '_' + detector_type + '_' + 'HOI{}_obj_feats.pkl'.format(test_type_prefix)
    print('output:', output_file)
    # if os.path.exists(output_file):
    #     exit()

    # init session

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    tfconfig = tf.ConfigProto(device_count={"CPU": 12},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tfconfig)

    # Generate_HICO_detection(output_file, HICO_dir)
    if args.model.__contains__('VCOCO'):
        os.environ['DATASET'] = 'VCOCO1'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    elif args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)

    stride = 200

    test_type = 'default'
    test_type = args.test_type


    image, blobs, image_id = obtain_test_dataset_with_obj(args.object_thres, args.human_thres,
                                                          stride=stride, test_type=test_type, hoi_nums=1, model_name=args.model)

    tmp_labels = tf.one_hot(tf.reshape(tf.cast(blobs['O_cls'], tf.int32), shape=[-1, ]), 80, dtype=tf.float32)
    tmp_ho_class_from_obj = tf.cast(tf.matmul(tmp_labels, net.obj_to_HO_matrix) > 0, tf.float32)
    # action_ho = blobs['O_cls']

    net.set_ph(image, image_id, num_pos=blobs['H_num'], Human_augmented=blobs['H_boxes'], Object_augmented=blobs['O_boxes'],
               sp=blobs['sp'])
    net.set_add_ph(pos1_idx=blobs['H_num'] // 2)
    # net.init_verbs_objs_cls()
    net.create_architecture(False)
    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(v.name)

    # assert not net.model_name.__contains__('_base')

    # verb_feats = net.intermediate['fc7_verbs']
    # O_features = [net.intermediate['fc7_O'][:net.pos1_idx],
    #               net.intermediate['fc7_O'][net.pos1_idx:net.get_compose_num_stop()]]
    # V_features = [verb_feats[:net.pos1_idx],
    #               verb_feats[net.pos1_idx:]]


    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)
    # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(var.name, var.eval(sess).mean())
    detection = []
    count = 0
    print('Pre-trained weights loaded.')
    _t = {'im_detect': Timer(), 'misc': Timer()}
    while True:
        _t['im_detect'].tic()
        try:
            # f_obj_score, f_obj_cls, _o_box, _o_cls, _h_score, _o_score, _image_id = [0],[0],[0],[0],[0],[0],1
            # o_feats, _o_box = sess.run([net.test_visualize['fc7_O_feats'], blobs['O_boxes']])

            o_feats, _o_box, _o_cls, _h_score, _o_score, _image_id = sess.run(
                [net.test_visualize['fc7_O_feats'],
                 blobs['O_boxes'], blobs['O_cls'], blobs['H_score'], blobs['O_score'],
                 image_id])
            _o_box = _o_box[:, 1:]
        except tf.errors.OutOfRangeError:
            print('END')
            break
        _t['im_detect'].toc()
        # start_len = len(_o_box)//2
        # temp = [[_h_box[i+start_len], _o_box[i+start_len], _o_cls[i], 0, _h_score[i], _o_score[i], merge_probs[i]] for i in range(len(_h_box)//2)]
        # temp = [[0, _o_box[i+start_len], _o_cls[i], 0, 0, _o_score[i], merge_probs[i]] for i in range(len(_h_box)//2)]
        new_o_cls = _o_cls[-1]
        if test_type == 'gtobj365':
            # print(new_o_cls)
            assert new_o_cls in [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183, 300, 225, 282, 335]
        new_o_score = _o_score[-1]
        new_o_box = _o_box[-1]
        num = len(o_feats)
        # print([num, new_o_box, new_o_cls, new_o_score, o_feats[-1], _image_id], '\n')
        detection.append([num, new_o_box, new_o_cls, new_o_score, o_feats[-1], _image_id])

        count += 1
        # if count > 10:
        #     break

        print('\rmodel: {} im_detect: {:d}  {:d}, {:.3f}s'.format(net.model_name, count, _image_id,_t['im_detect'].average_time), end='', flush=True)

    import pickle
    detector_type = 'res101'
    # if args.model.__contains__('VCOCO'):
    #     detector_type = 'default'
    # if len(args.test_type) > 0:
    #     args.test_type = '_'+args.test_type
    pickle.dump(detection, open(output_file, 'wb'))
    sess.close()
