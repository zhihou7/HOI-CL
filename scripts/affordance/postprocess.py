
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from ult.Generate_HICO_detection import Generate_HICO_detection
from ult.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=1800000, type=int)
    parser.add_argument('--iter_start', dest='iter_start',
                        help='Specify which weight to load',
                        default=20000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='iCAN_ResNet50_HICO', type=str)
    parser.add_argument('--fuse_type', dest='fuse_type', default='spv')
    parser.add_argument('--type', dest='test_type',
                        help='Human threshold',
                        default='vcl', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    DATA_DIR = cfg.LOCAL_DATA
    max_iteration = 3000001
    args = parse_args()
    process_num = 2 if args.fuse_type == 'spv' else 2

    pool = None
    # print(args)
    iteration = args.iteration
    fuse_type = args.fuse_type
    model = args.model
    suffix = '_tin' # because we use the strategy from TIN during inference

    cur_list = []
    weight = cfg.ROOT_DIR + '/Weights/' + model + '/HOI_iter_' + str(iteration) + '.ckpt'


    output_file =DATA_DIR + '/Results/' + str(iteration) + '_' + model + suffix + '.pkl'
    import os
    import time
    print ('iter = ' + str(iteration) + ', path = ' + weight  + ', fuse_type = ' + str(fuse_type))

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(iteration) + '_' + model + '_'+str(fuse_type) +'/'
    if not os.path.exists(os.environ['HOME'] + '/Results/HICO/'):
        os.makedirs(os.environ['HOME'] + '/Results/HICO/')

    Generate_HICO_detection(output_file, HICO_dir, fuse_type, pool)
    print ('iter = ' + str(iteration) + ', path = ' + weight  + ', fuse_type = ' + str(fuse_type))

    prefix = model + '_' + fuse_type
    dst = 'output/ho_1_s/hico_det_test2015/rcnn_caffenet_pconv_ip_' + prefix + '_iter_' + str(iteration)
    exp_name = 'rcnn_caffenet_ho_pconv_ip1_s_' + prefix + '_iter_' + str(iteration)
    prefix = 'rcnn_caffenet_pconv_ip_' + prefix
    result_file = ''
    os.system('matlab -nodesktop -nosplash -r "Generate_detection(\'{}\', \'{}\'); quit"'.format(dst, HICO_dir))
    os.system('matlab -nodesktop -nosplash -r "eval_def(\'{}\', \'{}\', \'{}\', \'{}\'); quit"'.format(exp_name, prefix, result_file,
                                                                                   str(iteration)))
    os.system('matlab -nodesktop -nosplash -r "eval_ko(\'{}\', \'{}\', \'{}\', \'{}\'); quit"'.format(exp_name, prefix, result_file,
                                                                                   str(iteration)))
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
