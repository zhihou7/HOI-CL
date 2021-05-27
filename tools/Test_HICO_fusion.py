# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ['DATASET'] = 'HICO'
os.environ["KMP_BLOCKTIME"] = str(0)
os.environ["KMP_SETTINGS"] = str(1)
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(8)
import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb

from ult.config import cfg
from models.test_HICO import test_net_data_api_fuse_obj, obtain_test_dataset_fcl, obtain_test_dataset_fusion

obj_2_hico = {4: 0, 47: 1, 24: 2, 46: 3, 34: 4, 35: 5, 21: 6, 59: 7, 13: 8, 1: 9, 14: 10, 8: 11, 73: 12, 39: 13, 45: 14, 50: 15, 5: 16, 55: 17, 2: 18, 51: 19, 15: 20, 67: 21, 56: 22, 74: 23, 57: 24, 19: 25, 41: 26, 60: 27, 16: 28, 54: 29, 20: 30, 10: 31, 42: 32, 29: 33, 23: 34, 78: 35, 26: 36, 17: 37, 52: 38, 66: 39, 33: 40, 43: 41, 63: 42, 68: 43, 3: 44, 64: 45, 49: 46, 69: 47, 12: 48, 0: 49, 53: 50, 58: 51, 72: 52, 65: 53, 48: 54, 76: 55, 18: 56, 71: 57, 36: 58, 30: 59, 31: 60, 44: 61, 32: 62, 11: 63, 28: 64, 37: 65, 77: 66, 38: 67, 27: 68, 70: 69, 61: 70, 79: 71, 9: 72, 6: 73, 7: 74, 62: 75, 25: 76, 75: 77, 40: 78, 22: 79}
coco_2_hico_obj_matrix = np.zeros([80, 80])
for i in range(80):
    coco_2_hico_obj_matrix[i][obj_2_hico[i]] = 1

def parse_args():
    parser = argparse.ArgumentParser(description='Test an model on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
                        help='Specify which weight to load',
                        default=600000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='FCL_union_l2_zs_s0_vloss2_varl_gan_dax_rands_rew_aug5_x5new_res101', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.1, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.3, type=float)
    parser.add_argument('--debug', dest='debug',
                        help='for fusing object prediction',
                        default=1, type=int)
    parser.add_argument('--type', dest='test_type',
                        help='vcl, drg, gt, coco101, coco50 ...',
                        default='vcl', type=str)
    parser.add_argument('--not_h_threhold', dest='not_h_threhold',
                        help='not_h_threhold',
                        action='store_true')
    args = parser.parse_args()
    return args


def switch_checkpoint_path(model_checkpoint_path):
    head = model_checkpoint_path.split('Weights')[0]
    model_checkpoint_path = model_checkpoint_path.replace(head, cfg.LOCAL_DATA + '/')
    return model_checkpoint_path

if __name__ == '__main__':

    args = parse_args()
    print(args)

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    import os
    if not os.path.exists(weight + '.index'):
        weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print('weight:', weight)
    print('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(
        args.iteration) + ', path = ' + weight)
    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '_tin.pkl'
    if os.path.exists(output_file):
        os.remove(output_file)
    # init session

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    tfconfig = tf.ConfigProto(device_count={"CPU": 12},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import DisentanglingNet
        net = DisentanglingNet(model_name=args.model)
    else:
        from networks.HOI import DisentanglingNet
        net = DisentanglingNet(model_name=args.model)
    stride = 200

    pattern_type = 0
    image, blobs, image_id = obtain_test_dataset_fusion(args.object_thres, args.human_thres,
                                                     stride=stride, test_type=args.test_type)
    print(blobs, image)


    # action_ho = blobs['O_cls']
    net.set_ph(image, image_id, num_pos=blobs['H_num'], Human_augmented=blobs['H_boxes'],
               Object_augmented=blobs['O_boxes'],
               action_HO=None, sp=blobs['sp'],
               )
    # net.init_verbs_objs_cls()
    net.create_architecture(False)
    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    test_net_data_api_fuse_obj(sess, net, output_file, blobs['H_boxes'][:, 1:], blobs['O_boxes'][:, 1:],
                       blobs['O_cls'], blobs['H_score'], blobs['O_score'], blobs['O_all_score'], image_id, args.debug)
    sess.close()