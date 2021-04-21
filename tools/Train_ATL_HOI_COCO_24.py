# --------------------------------------------------------
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
from models.train_Solver_HICO_MultiGPU import SolverWrapperMultiGPU
from ult.ult import obtain_coco_data, obtain_coco_data_hoicoco_24, obtain_coco_data3_hoicoco_24_atl


def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=500000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='ATL_union_multi_atl_ml5_l05_t5_def2_aug5_new_VCOCO_coco_CL_24', type=str)
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
    output_dir = cfg.LOCAL_DATA + '/Weights/' + args.model + '/'
    if os.path.exists(output_dir + 'checkpoint'):
        args.Restore_flag = -1
    import os
    os.environ['DATASET'] = 'VCOCO'
    from networks.HOI import DisentanglingNet
    augment_type =4
    network = DisentanglingNet(args.model)
    image, image_id, num_pos, blobs = obtain_coco_data_hoicoco_24(args.Pos_augment, args.Neg_select)
    if args.model.__contains__('atl'):
        image, image_id, num_pos, blobs = obtain_coco_data_hoicoco_24_atl(args.Pos_augment, args.Neg_select)
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
        sw = VCOCOSolverWrapperMultiGPU(sess, network, output_dir, tb_dir,
                                    args.Restore_flag, weight)
        print(blobs)
        sw.set_coco_data(image, image_id, num_pos, blobs)
        print('Solving..., Pos augment = ' + str(args.Pos_augment) + ', Neg augment = ' + str(
            args.Neg_select) + ', Restore_flag = ' + str(args.Restore_flag))
        sw.train_model(sess, args.max_iters)
        print('done solving')
