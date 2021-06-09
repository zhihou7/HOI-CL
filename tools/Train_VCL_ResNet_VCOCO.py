# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import ipdb

from models.train_Solver_VCOCO_MultiGPU import VCOCOSolverWrapperMultiGPU
from ult.config import cfg
from ult.ult import obtain_coco_data1


def parse_args():
    parser = argparse.ArgumentParser(description='Train VCL on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=500000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_multi_ml1_l05_t3_rew_aug5_3_new_VCOCO_test', type=str)
    # VCL_union_multi_base_l05_t2_rew_aug5_3_new_VCOCO_test
    # VCL_union_multi_ml1_l05_t3_rew_aug5_3_new_VCOCO_test
    parser.add_argument('--Restore_flag', dest='Restore_flag',
            help='Number of Res5 blocks',
            default=5, type=int)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one. (By jittering the object detections)',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=30, type=int)
    args = parser.parse_args()
    return args

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

devices_list = '0'
for i in range(1, len(get_available_gpus())):
    devices_list = devices_list + ',' + str(i)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = devices_list

if __name__ == '__main__':
    import os

    args = parse_args()
    try:
        import skimage
        if skimage.__version__ != '0.14.2':
            print("ALERT!!!: The version of skimage might affect the running speed largely. I'm not sure. I use 0.14.2")
    except :
        print('no skimage=================')
        pass
    np.random.seed(cfg.RNG_SEED)
    if args.model.__contains__('res101'):
        weight    = cfg.ROOT_DIR + '/Weights/res101_faster_rcnn_iter_1190000.ckpt'
    else:
        weight    = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    assert args.model.__contains__('_t3_') or args.model.__contains__('_t2_'), 'you must choice t2 or t3 in VCL. we use the t3 strategy which seems like more effective in our experiment. ' \
                                                                               't3 means we use a differect classifier with 238 targets. ' \
                                                                               't2 means we use a similar classifier to the backbone. ' \
                                                                               'Noticeably, VCOCO only contains 26 verbs. We do not evaluate t2 and t3 carefully'
    # output directory where the logs are saved
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

    import os
    os.environ['DATASET'] = 'VCOCO'
    from networks.HOI import HOI
    augment_type = 0
    if args.model.__contains__('_aug5'):
        augment_type = 4
    elif args.model.__contains__('_aug6'):
        augment_type = 5
    network = HOI(args.model)
    image, image_id, num_pos, blobs = obtain_coco_data1(args.Pos_augment, args.Neg_select)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    tfconfig = tf.ConfigProto(device_count={"CPU": 16},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8)
    # tfconfig = tf.ConfigProto()
    tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = VCOCOSolverWrapperMultiGPU(sess, network, output_dir, tb_dir,
                                        args.Restore_flag, weight)
        print(blobs)
        sw.set_coco_data(image, image_id, num_pos, blobs)
        print('Solving..., Pos augment = ' + str(args.Pos_augment) + ', Neg augment = ' + str(
            args.Neg_select) + ', Restore_flag = ' + str(args.Restore_flag))
        sw.train_model(sess, args.max_iters)
        print('done solving')
