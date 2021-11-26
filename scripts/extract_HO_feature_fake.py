# --------------------------------------------------------
# Tensorflow 
# Licensed under The MIT License [see LICENSE for details]
# Written by zhihou
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import _init_paths
from ult.ult import obtain_data, obtain_test_data

import tensorflow as tf
import numpy as np
import argparse

from ult.config import cfg
from ult.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=80000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='iCAN_R_union_multi_base_rew_aug5_3_x5new', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.3, type=float)
    parser.add_argument('--type', dest='type',
                        help='Object threshold',
                        default='train', type=str)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight ) 
  
    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'

    HICO_dir = cfg.LOCAL_DATA + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    tfconfig = tf.ConfigProto(device_count={"CPU": 16},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    # init session
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth=True

    # else:
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.15
    sess = tf.Session(config=tfconfig)


    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import DisentanglingNet
        net = DisentanglingNet(model_name=args.model)
    else:
        from networks.HOI import DisentanglingNet
        net = DisentanglingNet(model_name=args.model)

    os.environ['FEATS'] = 'TRUE'
    if args.type == 'train':
        large_neg_for_ho = False
        if args.model.endswith('_aug5_new') or args.model.endswith('_aug6_new'):
            large_neg_for_ho = True
        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_data(
            Pos_augment=0, Neg_select=0, augment_type=-1, pattern_type=False)
        net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
    else:

        large_neg_for_ho = False
        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_test_data(
            Pos_augment=0,
            Neg_select=0,
            augment_type=-1,
            large_neg_for_ho=large_neg_for_ho)
        net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
    net.init_verbs_objs_cls()
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
    # result['fake_O_list'] = []
    # result['fake_O_all_list'] = []
    while True:
        _t['im_detect'].tic()

        try:
            _image, _image_id, fc7_O, fc7_verbs, fc7_O_fake, fc7_O_fake_all, actions = sess.run(
                [image, image_id,
                 net.test_visualize['fc7_O_feats'],
                 net.test_visualize['fc7_verbs_feats'],
                 # net.test_visualize['fc7_fake_O_feats'],
                 tf.constant([0.]),
                 tf.constant([0.]),
                 action_HO,
                ])
        except tf.errors.OutOfRangeError:
            print('END')
            break
        _t['im_detect'].toc()
        count += 1
        if args.model.__contains__('var_gan_gen') or args.model.__contains__('varv_gan_gen'):
            assert len(fc7_O) == len(fc7_verbs) == len(actions) == len(fc7_O_fake), ( len(fc7_O) ,len(fc7_verbs) ,len(actions) ,len(fc7_O_fake))
        else:
            assert len(fc7_O) == len(fc7_verbs) == len(actions), ( len(fc7_O) ,len(fc7_verbs) ,len(actions))

        # print(fc7_O.shape, actions.shape, fc7_O_fake.shape)
        result['O_list'].extend(fc7_O)
        result['V_list'].extend(fc7_verbs)
        # result['fake_O_list'].extend(fc7_O_fake)
        # result['fake_O_all_list'].extend(fc7_O_fake_all)
        result['img_id_list'].append([_image_id]*len(fc7_O))
        result['A_list'].extend(actions)
        print('im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(count, 9658, _image_id, _t['im_detect'].average_time))

    print(len(result['A_list']))
    import pickle
    pickle.dump(result, open(cfg.LOCAL_DATA+ '/feats/' + args.model +'_'+args.type+'_'+'HICO_HO_feats.pkl', 'wb'))
    sess.close()
#