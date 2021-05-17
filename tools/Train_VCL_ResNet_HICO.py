# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os



os.environ['DATASET'] = 'HICO'
import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import ipdb
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from ult.config import cfg
from models.train_Solver_HICO_MultiGPU import SolverWrapperMultiGPU

from ult.ult import obtain_data, obtain_data_vcl_hico, get_zero_shot_type, get_augment_type

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

devices_list = '0'
for i in range(1, len(get_available_gpus())):
    devices_list = devices_list + ',' + str(i)

os.environ["CUDA_VISIBLE_DEVICES"] = devices_list
logger.info(os.environ["CUDA_VISIBLE_DEVICES"])

def parse_args():
    parser = argparse.ArgumentParser(description='Train VCL on HICO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101', type=str)
    # VCL_union_multi_base_l2_rew2_aug5_3_x5new_res101
    # VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new
    # posesp for pose
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
    try:
        import skimage
        if skimage.__version__ != '0.14.2':
            print("ALERT!!!: The version of skimage might affect the running speed largely. I'm not sure. I use 0.14.2")
    except :
        print('no skimage=================')
        pass
    args = parse_args()
    args.model = args.model.strip()
    Trainval_GT = None
    Trainval_N = None

    np.random.seed(cfg.RNG_SEED)
    tf.random.set_random_seed(0)
    if args.model.__contains__('res101'):
        weight    = cfg.ROOT_DIR + '/Weights/res101_faster_rcnn_iter_1190000.ckpt'
    else:
        weight    = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    # output directory where the logs are saved
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'
    if os.path.exists(output_dir+'checkpoint'):
        args.Restore_flag = -1

    if args.model.__contains__('unique_weights'):
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
    # if args.model.__contains__('pose'):
    #     with_pose = True


    zero_shot_type = get_zero_shot_type(args.model)
    assert args.model.__contains__('multi')

    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_data_vcl_hico(
        Pos_augment=args.Pos_augment,
        Neg_select=args.Neg_select,
        augment_type=augment_type, with_pose=with_pose, zero_shot_type=zero_shot_type)
    print(Human_augmented[0])
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
        sw = SolverWrapperMultiGPU(sess, net, output_dir, tb_dir,
                                        args.Restore_flag, weight)
        sw.set_data(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
        print('Solving..., Pos augment = ' + str(args.Pos_augment) + ', Neg augment = ' + str(
            args.Neg_select) + ', Restore_flag = ' + str(args.Restore_flag))
        sw.train_model(sess, args.max_iters)
        print('done solving')