#!/home/zhihou/anaconda3/envs/tf/bin/python
# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# ---------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import os

from ult.config import cfg




def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=1800000, type=int)

    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_ResNet50_HICO', type=str)
    parser.add_argument('--forever', dest='forever', action='store_true', default=False)
    parser.add_argument('--fuse_type', dest='fuse_type', default='spv')
    # parser.add_argument('--object_thres', dest='object_thres',
    #                     help='Object threshold',
    #                     default=0.8, type=float)
    # parser.add_argument('--human_thres', dest='human_thres',
    #                     help='Human threshold',
    #                     default=0.6, type=float)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    max_iteration = 3000001
    args = parse_args()

    weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt.index'

    if os.path.exists(weight):
        output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '_tin.pkl'
        os.system('python tools/Test_VCL_ResNet_HICO.py --model ' + args.model + ' --num_iteration ' + str(args.iteration))
        if args.model.__contains__('R_V'):
            os.system(
                'cd Data/ho-rcnn/;python ../../scripts/postprocess_test.py --model ' + args.model + ' --num_iteration ' + str(
                    args.iteration) + ' --fuse_type v')
        else:
            os.system(
                'cd Data/ho-rcnn/;python ../../scripts/postprocess_test.py --model ' + args.model + ' --num_iteration ' + str(
                    args.iteration) + ' --fuse_type spv')
    else:
        raise Exception('weight do not exist')