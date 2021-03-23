# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# ---------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import _init_paths
from ult.ult import obtain_data, obtain_test_data

import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb

from ult.config import cfg
from ult.Generate_HICO_detection import Generate_HICO_detection
from ult.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=80000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_multi_base_rew_aug5_3_x5new', type=str)
    parser.add_argument('--type', dest='type',
                        help='Object threshold',
                        default='train', type=str)



    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print ('iter = ' + str(args.iteration) + ', path = ' + weight )
  
    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'

    HICO_dir = cfg.LOCAL_DATA + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    tfconfig = tf.ConfigProto(device_count={"CPU": 16},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    # init session
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)


    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)


    if args.type == 'train':
        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_data(
            Pos_augment=0,
            Neg_select=0,
            augment_type=-1,
            pattern_type=False)
        net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
    else:
        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_test_data(
            Pos_augment=0,
            Neg_select=0,
            augment_type=-1,
            with_pose=False)
        net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
    net.create_architecture(False)

    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    detection = {}

    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0
    img_id_list = []
    O_list = []
    V_list = []
    A_list = []
    result = {}
    result['img_id_list'] = []
    result['O_list'] = []
    result['V_list'] = []
    result['A_list'] = []
    while True:
        _t['im_detect'].tic()

        try:
            _image, _image_id, fc7_O, fc7_verbs, actions = sess.run(
                [image, image_id,
                 net.test_visualize['fc7_O_feats'],
                 net.test_visualize['fc7_verbs_feats'],
                 action_HO,
                ])
        except tf.errors.OutOfRangeError:
            print('END')
            break
        _t['im_detect'].toc()
        count += 1
        # print(fc7_O.shape, actions.shape)
        result['O_list'].extend(fc7_O)
        result['V_list'].extend(fc7_verbs)
        result['img_id_list'].append([_image_id]*len(fc7_O))
        result['A_list'].extend(actions)
        print('im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(count, 9658, _image_id, _t['im_detect'].average_time))

    print(len(result['A_list']))
    import pickle
    pickle.dump(result, open(cfg.LOCAL_DATA + '/' + args.model +'_'+args.type+'_'+'HICO_HO_feats.pkl', 'wb'))
    sess.close()
#