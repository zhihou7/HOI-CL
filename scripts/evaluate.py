
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pickle import UnpicklingError

import _init_paths
import numpy as np
import argparse
import pickle
import json
import ipdb
import os
import re

from ult.config import cfg
from ult.Generate_HICO_detection import Generate_HICO_detection




def parse_args():
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=500000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='FCL_resnet101_union_l2_zs_s0_vloss2_varl_gan_dax_rands_aug5_xnew', type=str)
    parser.add_argument('--fuse_type', dest='fuse_type', default='spv')
    parser.add_argument('--type', dest='test_type',
                        help='Human threshold',
                        default='vcl', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    DATA_DIR = '../../Data/'
    max_iteration = 3000001
    args = parse_args()
    from multiprocessing import Pool
    process_num = 2 if args.fuse_type == 'spv' else 2
    # global pool
    # if pool is None:
    # pool = Pool(processes=process_num)
    pool = None
    # print(args)
    iteration = args.iteration
    fuse_type = args.fuse_type
    model = args.model

    cur_list = []
    weight = cfg.ROOT_DIR + '/Weights/' + model + '/HOI_iter_' + str(iteration) + '.ckpt'


    import os
    output_file = DATA_DIR + '/Results/' + str(iteration) + '_' + model + '.pkl'

    print ('iter = ' + str(iteration) + ', path = ' + weight  + ', fuse_type = ' + str(fuse_type))
    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(iteration) + '_' + model + '_'+str(fuse_type) +'/'
    if not os.path.exists(os.environ['HOME'] + '/Results/HICO/'):
        os.makedirs(os.environ['HOME'] + '/Results/HICO/')
    # print(output_file, HICO_dir)
    try:
        Generate_HICO_detection(output_file, HICO_dir, fuse_type, pool)
    except EOFError as e:
        print('%s EOF' % (output_file))
        exit()
    except UnpicklingError as e:
        exit()
    print ('iter = ' + str(iteration) + ', path = ' + weight  + ', fuse_type = ' + str(fuse_type))

    prefix = model + '_' + fuse_type
    dst = 'output/ho_1_s/hico_det_test2015/rcnn_caffenet_pconv_ip_' + prefix + '_iter_' + str(iteration)
    exp_name = 'rcnn_caffenet_ho_pconv_ip1_s_' + prefix + '_iter_' + str(iteration)
    prefix = 'rcnn_caffenet_pconv_ip_' + prefix
    result_file = 'result.csv' # store the results
    iteration = iteration
    os.system('matlab -nodesktop -nosplash -r "Generate_detection(\'{}\', \'{}\'); quit"'.format(dst, HICO_dir))
    print(HICO_dir)
    os.system('matlab -nodesktop -nosplash -r "eval_one_def(\'{}\', \'{}\', \'{}\', \'{}\'); quit"'.format(exp_name, prefix, result_file,
                                                                                   str(iteration)))
    os.system('matlab -nodesktop -nosplash -r "eval_one_ko(\'{}\', \'{}\', \'{}\', \'{}\'); quit"'.format(exp_name, prefix, result_file,
                                                                                   str(iteration)))
    f = open(result_file, 'a')
    f.write(' {}\n'.format(args.test_type))
    f.close()

    import os
    print("Remove temp files", HICO_dir)
    filelist = [f for f in os.listdir(HICO_dir)]
    for f in filelist:
        if os.path.exists(os.path.join(HICO_dir, f)):
            os.remove(os.path.join(HICO_dir, f))
    os.removedirs(HICO_dir)

    remove_dir = dst
    print(dst)
    filelist = [f for f in os.listdir(remove_dir)]
    for f in filelist:
        if os.path.exists(os.path.join(remove_dir, f)):
            os.remove(os.path.join(remove_dir, f))
    os.removedirs(remove_dir)
    # os.remove(output_file)
