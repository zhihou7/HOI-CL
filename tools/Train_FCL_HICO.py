


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

import os

os.environ["KMP_BLOCKTIME"] = str(0)
os.environ["KMP_SETTINGS"] = str(1)
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(8)

import tensorflow as tf
import numpy as np
import argparse

from ult.config import cfg
from ult.ult import obtain_data, get_zero_shot_type, get_augment_type


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=600000, type=int)
    parser.add_argument('--model', dest='model',
             help='Select model: '
                  'open long-tailed FCL: FCL_union_l2_zs_s0_vloss2_varl_gan_dax_rands_rew_aug5_x5new_res101,'
                  'open long-tailed baseline: FCL_base_union_l2_zs_s0_vloss2_rew_aug5_x5new_res101'
                        'rare-first zero-shot:FCL_union_l2_zsrare_s0_vloss2_varl_gan_dax_rands_rew_aug5_x5new_res101'
                        'non-rare-first zero-shot:FCL_union_l2_zsnrare_s0_vloss2_varl_gan_dax_rands_rew_aug5_x5new_res101'
                        'you can also use FCL_union_l2_zs_s0_vloss2_var_gan_dax_rands_rew_aug5_x5new_res101',
            default='FCL_union_l2_zs_s0_vloss2_varl_gan_dax_rands_rew_aug5_x5new_res101', type=str)
    # This is our baseline FCL_base_union_l2_zs_s0_vloss2_rew_aug5_xnew_res101

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

    # args.max_iters
    # it is better change the max_iters for zero-shot HOI detection since the training data is small when we fine-tune the network.

    Trainval_GT = None
    Trainval_N = None
    np.random.seed(cfg.RNG_SEED)

    if args.model.__contains__('res101'):
        weight    = cfg.ROOT_DIR + '/Weights/res101_faster_rcnn_iter_1190000.ckpt'
    else:
    # we will provide the resnet50 model in the released version
        weight    = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    print(weight)
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'
    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

    augment_type = get_augment_type(args.model)
    start_epoch = 0

    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)

    pattern_type = 0

    neg_type_ratio = 0
    zero_shot_type = get_zero_shot_type(args.model)
    human_adj = None
    obj_adj = None

    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_data(
        Pos_augment=args.Pos_augment, Neg_select=args.Neg_select, augment_type=augment_type, pattern_type=pattern_type,
        zero_shot_type=zero_shot_type, epoch=start_epoch)

    net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)

    if args.model.__contains__('gan'):
        from models.train_Solver_HICO_FCL import train_net
    else:
        from models.train_Solver_HICO import train_net
    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment, args.Neg_select, args.Restore_flag, weight, max_iters=args.max_iters)
    
