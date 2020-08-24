#!/home/zhihou/anaconda3/envs/tf/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pickle import UnpicklingError

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb
import os

from ult.config import cfg
from ult.Generate_HICO_detection import Generate_HICO_detection




def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=1800000, type=int)
    parser.add_argument('--iter_start', dest='iter_start',
                        help='Specify which weight to load',
                        default=20000, type=int)
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

import signal, psutil


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    # print('kill', children)
    for process in children:
        os.system('kill -9 '+str(process.pid))
        # process.send_signal(signal.SIGKILL)

if __name__ == '__main__':
    print("""I use Octave to evaluate the performace cause I did not install matlab in my machine""")
    # DATA_DIR = '/run/user/1000/gvfs/ftp:host=172.26.1.56/'
    DATA_DIR = cfg.LOCAL_DATA
    max_iteration = 3000001
    args = parse_args()
    from multiprocessing import Pool
    process_num = 2 if args.fuse_type == 'spv' else 2
    if not os.path.exists(cfg.LOCAL_DATA + '/csv'):
        os.makedirs(cfg.LOCAL_DATA + '/csv')
    # global pool
    # if pool is None:
    # pool = Pool(processes=process_num)
    pool = None
    # print(args)
    iteration = args.iteration
    iter_list = []
    stride = 5000
    if args.iter_start < 0:
        iter_list = [iteration]
    else:
        iter_list = range(args.iter_start, max_iteration, stride)

    import re
    if args.model.__contains__('*'):
        # print('line', args.model)

        r = re.compile(args.model)

        import glob
        tmp = glob.glob(DATA_DIR + '/Weights/*')
        tmp.sort(key=os.path.getmtime)
        tmp = tmp[-20:]
        # tmp = list(set(os.listdir(DATA_DIR + '/Weights/')))
        tmp = [item.split('/')[-1] for item in tmp]
        model_arr = list(filter(r.match, tmp))
        model_arr = sorted(model_arr)
        model_arr = [item.strip() for item in model_arr]
        # print([item for item in enumerate(model_arr)])
    else:
        model_arr = args.model.split(',')

    filesss = os.listdir(cfg.LOCAL_DATA + '/Results/')
    r = re.compile(args.model)
    filesss = list(filter(r.match, filesss))
    filesss = sorted(filesss)

    iteration = args.iteration
    fuse_type = args.fuse_type
    model = args.model
    for suffix in [ '_tin']:

        if model.startswith('VCL_V_') and args.fuse_type != 'v':
            continue
        cur_list = []
        result_file_name = DATA_DIR + '/csv/' + model + '_' + str(fuse_type) + suffix+'.csv'
        weight = cfg.ROOT_DIR + '/Weights/' + model + '/HOI_iter_' + str(iteration) + '.ckpt'

        import os
        output_file = cfg.LOCAL_DATA + '/Results/' + str(iteration) + '_' + model + suffix + '.pkl'
        if not os.path.exists(output_file) or os.path.getsize(output_file) < 1000:
            print("not exists", output_file)
            continue
        f = open(result_file_name, 'a')
        f.write('%6d ' % iteration)
        f.close()
        import time
        time.sleep(20)
        print ('iter = ' + str(iteration) + ', path = ' + weight  + ', fuse_type = ' + str(fuse_type))

        HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(iteration) + '_' + model + '_'+str(fuse_type) +'/'
        if not os.path.exists(os.environ['HOME'] + '/Results/HICO/'):
            os.makedirs(os.environ['HOME'] + '/Results/HICO/')
        # print(output_file, HICO_dir)
        from oct2py import Oct2Py

        oct = Oct2Py()
        try:
            Generate_HICO_detection(output_file, HICO_dir, fuse_type, pool)
        except EOFError as e:
            print('%s EOF' % (output_file))
            break
        except UnpicklingError as e:
            print("%s UnpicklingError" %(output_file))
            f = open(result_file_name, 'r')
            lines = f.readlines()
            w = open(result_file_name, 'a')
            for item in lines[:-1]:
                w.write(item)
            w.close()
            break
        print ('iter = ' + str(iteration) + ', path = ' + weight  + ', fuse_type = ' + str(fuse_type))
        # import os
        prefix = model  +'_'+fuse_type
        dst = 'output/ho_1_s/hico_det_test2015/rcnn_caffenet_pconv_ip_'+prefix + '_iter_' + str(iteration)
        oct.eval('addPaths')

        oct.push('dst', dst)
        oct.push('dect_dir', HICO_dir)
        oct.eval('OctGenerate_detection')

        oct.push('exp_name', 'rcnn_caffenet_ho_pconv_ip1_s_'+prefix + '_iter_' + str(iteration))
        oct.push('exp_dir', 'ho_1_s')
        oct.push('prefix', 'rcnn_caffenet_pconv_ip_'+prefix)
        oct.push('result_file', result_file_name)
        oct.push('format', 'obj')
        oct.push('score_blob', 'n/a')
        oct.push('image_set', 'test2015')
        oct.push('iter', iteration)

        oct.push('eval_mode', 'def')
        oct.eval('oct_eval_one')

        f = open(result_file_name, 'a')
        f.write('\n')
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
        # kill_child_processes(os.getpid())