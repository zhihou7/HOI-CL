# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
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


from ult.config import cfg
from models.train_Solver_VCOCO import train_net
from ult.ult import obtain_coco_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train VCL on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=200000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_l2_rew_aug5_3_x5new_VCOCO', type=str)
    # VCL_union_l2_rew_aug5_3_x5new_VCOCO
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


if __name__ == '__main__':

    args = parse_args()

    Trainval_GT       = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO.pkl', "rb" ), encoding='latin1' )
    Trainval_N        = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO.pkl', "rb" ), encoding='latin1' )

    np.random.seed(cfg.RNG_SEED)
    weight    = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    # output directory where the logs are saved
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.LOCAL_DATA + '/Weights/' + args.model + '/'
    if args.model.__contains__('unique_weights'):
        args.Restore_flag = 6
    iCAN_Early_flag = 0
    import os
    os.environ['DATASET'] = 'VCOCO'
    from networks.HOI import HOI
    augment_type = 0
    if args.model.__contains__('_aug2'):
        augment_type = 1
    elif args.model.__contains__('_aug3'):
        augment_type = 2
    elif args.model.__contains__('_aug4'):
        augment_type = 3
    elif args.model.__contains__('_aug5'):
        augment_type = 4
    elif args.model.__contains__('_aug6'):
        augment_type = 5
    net = HOI(args.model)
    image, image_id, num_pos, blobs = obtain_coco_data(args.Pos_augment, args.Neg_select)
    net.set_ph(image, image_id, num_pos, blobs['sp'], blobs['H_boxes'],
               blobs['O_boxes'], blobs['gt_class_H'], blobs['gt_class_HO'], blobs['gt_class_sp'],
               blobs['Mask_HO'], blobs['Mask_H'], blobs['Mask_sp'], blobs['gt_class_C'])

    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment, args.Neg_select, iCAN_Early_flag, args.Restore_flag, weight, max_iters=args.max_iters)
