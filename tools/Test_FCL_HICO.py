# --------------------------------------------------------

# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os

os.environ["KMP_BLOCKTIME"] = str(0)
os.environ["KMP_SETTINGS"] = str(1)
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(8)

import tensorflow as tf
import numpy as np
import argparse

from ult.config import cfg
from models.test_HICO import obtain_test_dataset_fcl, test_net_data_fcl

def parse_args():
    parser = argparse.ArgumentParser(description='Test an FCL on HICO')
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
                        help='debug, if not debug, ignore.',
                        default=0, type=int)
    parser.add_argument('--type', dest='test_type',
                        help='vcl, drg, gt, coco101, coco50 ...',
                        default='vcl', type=str)
    # vcl, drg, gt, coco101, coco50 ...
    args = parser.parse_args()
    return args

def switch_checkpoint_path(model_checkpoint_path):
    head = model_checkpoint_path.split('Weights')[0]
    model_checkpoint_path = model_checkpoint_path.replace(head, cfg.LOCAL_DATA +'/')
    return model_checkpoint_path

if __name__ == '__main__':

    args = parse_args()
    print(args)

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    import os
    if not os.path.exists(weight + '.index'):
        weight = cfg.LOCAL_DATA + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print('weight:', weight)
    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight )
    if not os.path.exists(cfg.LOCAL_DATA + '/Results/'):
        os.makedirs(cfg.LOCAL_DATA + '/Results/')
    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'
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
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)

    stride = 200

    pattern_type = 0
    image, blobs, image_id = obtain_test_dataset_fcl(args.object_thres, args.human_thres,
                                                     stride=stride, test_type=args.test_type, model_name=args.model,
                                                     pattern_type=pattern_type)
    print(blobs, image)

    net.set_ph(image, image_id, num_pos=blobs['H_num'], Human_augmented=blobs['H_boxes'], Object_augmented=blobs['O_boxes'],
               action_HO=None, sp=blobs['sp'],
               )
    # net.set_add_ph()
    net.create_architecture(False)
    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    test_net_data_fcl(sess, net, output_file, blobs['H_boxes'][:, 1:], blobs['O_boxes'][:, 1:],
                      blobs['O_cls'], blobs['H_score'], blobs['O_score'], image_id)
    sess.close()