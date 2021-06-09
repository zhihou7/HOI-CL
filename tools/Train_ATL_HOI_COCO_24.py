# --------------------------------------------------------
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse

from models.train_Solver_VCOCO_HOI import VCOCOSolverWrapperCL
from models.train_Solver_VCOCO_MultiGPU import VCOCOSolverWrapperMultiGPU
from ult.config import cfg
from ult.ult import obtain_coco_data, obtain_coco_data_hoicoco_24, obtain_coco_data_atl


def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=500000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='ATL_union_multi_atl_ml5_l05_t4_def2_aug5_new_VCOCO_coco_CL_24', type=str)
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
    import os
    os.environ['DATASET'] = 'HICO'
    os.environ["KMP_BLOCKTIME"] = str(0)
    os.environ["KMP_SETTINGS"] = str(1)
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    os.environ["OMP_NUM_THREADS"] = str(8)

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)
    weight    = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    # output directory where the logs are saved
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'
    if os.path.exists(output_dir + 'checkpoint'):
        args.Restore_flag = -1
    import os
    os.environ['DATASET'] = 'VCOCO1'
    from networks.HOI import HOI
    augment_type = 4
    network = HOI(args.model)
    image, image_id, num_pos, blobs = obtain_coco_data_hoicoco_24(args.Pos_augment, args.Neg_select)
    if args.model.__contains__('atl'):
        atl_type = 1
    if args.model.__contains__('_hico_'):
        atl_type = 2
    elif args.model.__contains__('_both_'):
        atl_type = 3
    elif args.model.__contains__('_vcoco_'):
        atl_type = 5
    elif args.model.__contains__('_coco_'):
        atl_type = 9
        image, image_id, num_pos, blobs = obtain_coco_data_atl(args.Pos_augment, args.Neg_select, atl_type, vcoco_type=24)
    else:
        image, image_id, num_pos, blobs = obtain_coco_data_hoicoco_24(args.Pos_augment, args.Neg_select)

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
        sw = VCOCOSolverWrapperCL(sess, network, output_dir, tb_dir,
                                    args.Restore_flag, weight)
        print(blobs)
        sw.set_coco_data(image, image_id, num_pos, blobs)
        print('Solving..., Pos augment = ' + str(args.Pos_augment) + ', Neg augment = ' + str(
            args.Neg_select) + ', Restore_flag = ' + str(args.Restore_flag))
        sw.train_model(sess, args.max_iters)
        print('done solving')
