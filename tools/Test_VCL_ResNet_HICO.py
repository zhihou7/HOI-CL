# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ['DATASET'] = 'HICO'
import _init_paths
import tensorflow as tf
import argparse


from ult.config import cfg
from models.test_HICO import obtain_test_dataset1, test_net_data_api1

def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.3, type=float) 
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    # TODO For better object detector, the object_thres and human_thres should also be changed accordingly.
    #  e.g. in our fine-tuned detector, object_thres and human_thres is 0.1 and 0.3 respectively.
    parser.add_argument('--debug', dest='debug',
                        help='Human threshold',
                        default=0, type=int)
    parser.add_argument('--type', dest='test_type',
                        help='Human threshold',
                        default='res101', type=str)
    args = parser.parse_args()

    return args

def switch_checkpoint_path(model_checkpoint_path):
    head = model_checkpoint_path.split('Weights')[0]
    model_checkpoint_path = model_checkpoint_path.replace(head, cfg.LOCAL_DATA +'/')
    return model_checkpoint_path

if __name__ == '__main__':

    args = parse_args()
    print(args)
    # test detections result
    from sys import version_info

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    import os

    print('weight:', weight)
    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight )
    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '_tin.pkl'
    # init session

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    tfconfig = tf.ConfigProto(device_count={"CPU": 12},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    # init session

    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=tfconfig)



    # Generate_HICO_detection(output_file, HICO_dir)
    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    stride = 200


    image, blobs, image_id = obtain_test_dataset1(args.object_thres, args.human_thres,
                                                stride=stride, test_type=args.test_type, model_name=args.model)


    tmp_labels = tf.one_hot(tf.reshape(tf.cast(blobs['O_cls'], tf.int32), shape=[-1, ]), 80, dtype=tf.float32)
    tmp_ho_class_from_obj = tf.cast(tf.matmul(tmp_labels, net.obj_to_HO_matrix) > 0, tf.float32)
    # action_ho = blobs['O_cls']

    net.set_ph(image, image_id, num_pos=blobs['H_num'], Human_augmented=blobs['H_boxes'], Object_augmented=blobs['O_boxes'],
               action_HO=tmp_ho_class_from_obj, sp=blobs['sp'])
    # net.init_verbs_objs_cls()
    net.create_architecture(False)
    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(v.name)

    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    test_net_data_api1(sess, net, output_file, blobs['H_boxes'][:, 1:], blobs['O_boxes'][:, 1:],
                       blobs['O_cls'], blobs['H_score'], blobs['O_score'], image_id)
    sess.close()