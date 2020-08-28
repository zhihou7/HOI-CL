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
import json
import ipdb


from ult.config import cfg
from models.test_VCOCO import test_net
from ult.vsrl_eval import VCOCOeval


def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on VCOCO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=300000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_multi_ml1_l05_t3_rew_aug5_3_new_VCOCO_test', type=str)
    parser.add_argument('--prior_flag', dest='prior_flag',
            help='whether use prior_flag',
            default=3, type=int)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.4, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)


    args = parser.parse_args()
    return args

#     apply_prior   prior_mask
# 0        -             -          
# 1        Y             - 
# 2        -             Y
# 3        Y             Y


if __name__ == '__main__':

    args = parse_args()
  
    Test_RCNN      = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl', "rb" ) , encoding='latin1')
    prior_mask     = pickle.load( open( cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb" ) , encoding='latin1')
    Action_dic     = json.load(   open( cfg.DATA_DIR + '/' + 'action_index.json'))
    Action_dic_inv = {y:x for x,y in Action_dic.items()}

    vcocoeval      = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json', cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json', cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
    import ipdb
    vcocoeval._get_vcocodb()
    weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight ) 
  
    output_file = cfg.ROOT_DIR + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'

    # init session
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    tfconfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
    # tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)


    iCAN_Early_flag = 0
    import os
    os.environ['DATASET'] = 'VCOCO'
    from networks.HOI import HOI
    net = HOI(args.model)
    net.create_architecture(False)
    
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    
    test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_file, args.object_thres, args.human_thres, args.prior_flag)
    sess.close()

    vcocoeval._do_eval(output_file, ovr_thresh=0.5)  
