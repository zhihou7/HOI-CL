# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
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
import ipdb


from ult.config import cfg
from models.train_Solver_HICO import train_net
from ult.ult import obtain_data, get_zero_shot_type, get_augment_type


def parse_args():
    parser = argparse.ArgumentParser(description='Train VCL on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=1500010, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_l2_rew_aug5_3_x5new_res101', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one. (By jittering the object detections)',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
            help='How many ResNet blocks are there?',
            default=5, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)
    args.model = args.model.strip()

    Trainval_GT = None
    Trainval_N = None

    np.random.seed(cfg.RNG_SEED)
    if args.model.__contains__('res101'):
        weight    = cfg.ROOT_DIR + '/Weights/res101_faster_rcnn_iter_1190000.ckpt'
    else:
        weight    = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    print(weight)
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.LOCAL_DATA + '/Weights/' + args.model + '/'
    if args.Restore_flag == 5:
        if os.path.exists(output_dir+'checkpoint'):
            args.Restore_flag = -1
        elif args.model.__contains__('unique_weights'):
            args.Restore_flag = 6

    augment_type = get_augment_type(args.model)

    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)

    with_pose = False
    if args.model.__contains__('pose'):
        with_pose = True

    coco = False
    zero_shot_type = get_zero_shot_type(args.model)
    large_neg_for_ho = False
    if args.model.endswith('_aug5_new') or args.model.endswith('_aug6_new'):
        large_neg_for_ho = True
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_data(Pos_augment=args.Pos_augment,
                                                                                                       Neg_select=args.Neg_select,
                                                                                                       augment_type=augment_type,
                                                                                                       with_pose = with_pose,
                                                                                                       zero_shot_type=zero_shot_type,
                                                                                                       )
    print('coco', coco)

    net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)


    from models.train_Solver_HICO import train_net

    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment, args.Neg_select, args.Restore_flag, weight, max_iters=args.max_iters)
    