# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
# import ipdb
import logging



logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from ult.config import cfg
from models.train_Solver_HICO_MultiBatch import SolverWrapperMultiBatch

from ult.ult import obtain_data, get_zero_shot_type, get_augment_type, obtain_data2, \
    obtain_batch_data_semi1, get_epoch_iters, obtain_data2_large



def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=1200000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model,ATL: ATL_union_batch1_semi_l2_def4_vloss2_rew2_aug5_3_x5new_coco_res101,'
                 'Baseline: ATL_union_batch1_semi_l2_def4_vloss2_rew2_aug5_3_x5new_rehico_res101',
            default='ATL_union_batch1_semi_l2_def4_vloss2_rew2_aug5_3_x5new_coco_res101', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one. (By jittering the object detections)',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
            help='How many ResNet blocks are there?',
            default=5, type=int)
    parser.add_argument('--incre_classes', dest='incre_classes',
                        help='Human threshold',
                        default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    args.model = args.model.strip()
    # if args.model.__contains__('epic') and not args.model.__contains__('_s0_'):
    #     args.max_iters = 600000
    # if args.model.__contains__('semi'):
    #     args.max_iters = 1500000
    if args.model.__contains__('zs7') and args.max_iters > 400000:
        args.max_iters = 400000
    if args.model.__contains__('zs4') and not args.Restore_flag == -1:
        args.max_iters = 300000
    # if args.model.__contains__('epic') and args.model.__contains__('cosine5'):
    #     args.max_iters = get_epoch_iters(args.model) * 10 + 5
    Trainval_GT = None
    Trainval_N = None

    np.random.seed(cfg.RNG_SEED)
    tf.random.set_random_seed(0)
    if args.model.__contains__('res101'):
        weight    = cfg.ROOT_DIR + '/Weights/res101_faster_rcnn_iter_1190000.ckpt'
    else:
        weight    = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    # output directory where the logs are saved
    tb_dir     = cfg.LOCAL_DATA + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'
    start_epoch = 0
    if args.Restore_flag == 5:
        if os.path.exists(output_dir+'checkpoint'):
            args.Restore_flag = -1
        elif args.model.__contains__('cosine') and not args.model.__contains__('s0'):
            # This is for fine-tuning
            args.Restore_flag = -7
        elif args.model.__contains__('unique_weights'):
            args.Restore_flag = 6
    if args.model.__contains__('unique_weights'):
        args.Restore_flag = 6

    if args.Restore_flag == -1:
        ckpt = tf.train.get_checkpoint_state(output_dir)
        print(output_dir, ckpt.model_checkpoint_path)
        init_step = ckpt.model_checkpoint_path.split('/')[- 1].split('_')[- 1]
        init_step = int(init_step.replace('.ckpt', ''))
        start_epoch = init_step // get_epoch_iters(args.model)
    augment_type = get_augment_type(args.model)

    if args.model.__contains__('ICL'):
        # incremental continual learning
        # This is not used in the paper. We use this code for unknown zero-shot evaluation
        if args.model.__contains__('res101'):
            os.environ['DATASET'] = 'HICO_res101_icl'
        else:
            os.environ['DATASET'] = 'HICO_icl'
        from networks.HOI_Concept_Discovery import HOIICLNet
        import json
        incremental_class_pairs = [[]]
        if args.incre_classes is not None and os.path.exists(args.incre_classes):
            incremental_class_pairs = json.load(open(args.incre_classes))
        net = HOIICLNet(model_name=args.model, task_id=1, incremental_class_pairs=incremental_class_pairs)
    elif args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)

    pattern_type = 0
    zero_shot_type = get_zero_shot_type(args.model)
    large_neg_for_ho = False
    assert args.model.__contains__('batch')
    logger.info("large neg: %".format(large_neg_for_ho))
    isalign = False
    if args.model.__contains__('_ALA_'):
        isalign = True # This does not affect the result. In our experiment, we do not use this.
    neg_type_ratio = 0
    if args.model.__contains__('atl'):
        print('semi ====', args.model)
        if args.model.__contains__('vcoco'):
            semi_type = 'vcoco'
        elif args.model.__contains__('coco3'):
            semi_type = 'coco3'
        elif args.model.__contains__('coco1'):
            semi_type = 'coco1' # COCO2014

        elif args.model.__contains__('coco'):
            semi_type = 'coco'
        elif args.model.__contains__('rehico'):
            semi_type = 'rehico'
        elif args.model.__contains__('bothzs'):
            semi_type = 'bothzs' # use only zero-shot object images
        elif args.model.__contains__('both1'):
            semi_type = 'both1'
        elif args.model.__contains__('both'):
            semi_type = 'both'
        else:
            semi_type = 'default'
        bnum = 1
        if args.model.__contains__('batch1'):
            bnum = 2
        if args.model.__contains__('batch2'):
            bnum = 3
        if args.model.__contains__('batch3'):
            bnum = 4

        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, pos1_idx = obtain_batch_data_semi1(
            Pos_augment=args.Pos_augment,
            Neg_select=args.Neg_select,
            augment_type=augment_type, model_name=args.model, pattern_type=pattern_type, zero_shot_type=zero_shot_type, isalign=isalign,
            epoch=start_epoch, semi_type=semi_type, bnum=bnum, neg_type_ratio=neg_type_ratio)
    elif args.model.__contains__('large'):
        bnum = 4
        if args.model.__contains__('large2'):
            bnum = 2
        elif args.model.__contains__('large1'):
            bnum = 3
        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, pos1_idx = obtain_data2_large(
            Pos_augment=args.Pos_augment,
            Neg_select=args.Neg_select,
            augment_type=augment_type,
            model_name=args.model,
            pattern_type=pattern_type, zero_shot_type=zero_shot_type, bnum=bnum, neg_type_ratio=neg_type_ratio)
    else:
        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, pos1_idx = obtain_data2(
            Pos_augment=args.Pos_augment,
            Neg_select=args.Neg_select,
            augment_type=augment_type,
            model_name=args.model,
            pattern_type=pattern_type,
            zero_shot_type=zero_shot_type,
        neg_type_ratio=neg_type_ratio)
    net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
    net.set_add_ph(pos1_idx)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    tfconfig = tf.ConfigProto(device_count={"CPU": 8},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8)
    # tfconfig = tf.ConfigProto()
    tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        if args.model.__contains__('gan'):
            from models.train_Solver_HICO_MultiBatch_FCL import SolverWrapperMultiBatchFCL
            sw = SolverWrapperMultiBatchFCL(sess, net, output_dir, tb_dir,
                                       args.Restore_flag, weight)
        else:
            sw = SolverWrapperMultiBatch(sess, net, output_dir, tb_dir,
                                        args.Restore_flag, weight)
        sw.set_data(image, image_id, num_pos, Human_augmented, Object_augmented, net.gt_class_HO, sp)
        print('Solving..., Pos augment = ' + str(args.Pos_augment) + ', Neg augment = ' + str(
            args.Neg_select) + ', Restore_flag = ' + str(args.Restore_flag))
        sw.train_model(sess, args.max_iters)
        print('done solving')