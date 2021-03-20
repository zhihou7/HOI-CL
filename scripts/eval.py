
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=500000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='FCL_resnet101_union_l2_zs_s0_vloss2_varl_gan_dax_rands_aug5_xnew', type=str)
    parser.add_argument('--fuse_type', dest='fuse_type', default='spv')
    # parser.add_argument('--object_thres', dest='object_thres',
    #                     help='Object threshold',
    #                     default=0.1, type=float)
    # parser.add_argument('--human_thres', dest='human_thres',
    #                     help='Human threshold',
    #                     default=0.3, type=float)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    max_iteration = 3000001
    args = parse_args()
    # global pool
    # if pool is None:
    print(args)
    iteration = args.iteration
    iter_list = []
    stride = 10000

    fuse_type = args.fuse_type

    cur_list = []
    cl = []

    os.system(
        'python tools/Test_FCL_HICO.py --human_thres 0.3 --object_thres 0.1 --model '+args.model+' --num_iteration '+str(iteration)+' --type vcl')
    os.system(
        'cd Data/ho-rcnn/;python ../../scripts/evaluate.py --model ' + args.model + ' --num_iteration ' + str(
            iteration) + ' --fuse_type spv')

