# --------------------------------------------------------
# Tensorflow ATL
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

from ult.vsrl_eval1 import VCOCOeval
from ult.config import cfg
from models.test_VCOCO import test_net, obtain_data, test_net_data_api_21


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='iteration',
                        help='Specify which weight to load',
                        default=300000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='ATL_union_multi_atl_ml5_l05_t5_def2_aug5_new_VCOCO_coco_CL_21', type=str)
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

    Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl', "rb"),
                            encoding='latin1')
    prior_mask = pickle.load(open(cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb"), encoding='latin1')
    Action_dic = json.load(open(cfg.DATA_DIR + '/' + 'action_index.json'))
    Action_dic_inv = {y: x for x, y in Action_dic.items()}


    weight = cfg.LOCAL_DATA0 + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(
        args.iteration) + ', path = ' + weight)

    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'

    import os
    tfconfig = tf.ConfigProto(device_count={"CPU": 8},
                              inter_op_parallelism_threads=4,
                              intra_op_parallelism_threads=4,
                              allow_soft_placement=True)
    # init session
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.1
    # tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # if args.model == 'iCAN_ResNet50_VCOCO':
    #     from networks.iCAN_ResNet50_VCOCO import ResNet50
    #     net = ResNet50(args.model)
    # elif args.model == 'iCAN_ResNet50_VCOCO_Early':
    #     from networks.iCAN_ResNet50_VCOCO_Early import ResNet50
    #     net = ResNet50(args.model)
    # else:
    iCAN_Early_flag = 0
    import os

    os.environ['DATASET'] = 'VCOCO1'
    from networks.HOI import DisentanglingNet

    net = DisentanglingNet(args.model)

    print('Pre-trained weights loaded.')

    image, blobs, image_id = obtain_data(Test_RCNN, prior_mask, Action_dic_inv, output_file, args.object_thres,
                                         args.human_thres, args.prior_flag)
    print('blobs:', blobs)
    print(image)

    net.set_ph(image, image_id, num_pos=blobs['H_num'], sp=blobs['sp'], Human_augmented=blobs['H_boxes'],
               Object_augmented=blobs['O_boxes'],
               gt_cls_H=None,
               gt_cls_HO=None, gt_cls_sp=None, Mask_HO=None, Mask_H=None, Mask_sp=None, gt_compose = None)

    net.create_architecture(False)

    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    test_net_data_api_21(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_file,
                         args.object_thres, args.human_thres, args.prior_flag, blobs, image_id, image)

    sess.close()
    vcocoeval = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json',
                          cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json',
                          cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
    vcocoeval._do_eval(output_file, ovr_thresh=0.5)
    os.remove(output_file)
