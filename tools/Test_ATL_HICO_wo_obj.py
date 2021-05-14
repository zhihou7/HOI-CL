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
import tensorflow as tf
import argparse

from ult.config import cfg
from models.test_HICO import obtain_test_dataset_wo_obj, test_net_data_api_wo_obj


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
                        help='Specify which weight to load',
                        default=1800000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='iCAN_ResNet50_HICO', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.1, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.3, type=float)
    parser.add_argument('--debug', dest='debug',
                        help='Human threshold',
                        default=0, type=int)
    parser.add_argument('--type', dest='test_type',
                        help='Human threshold',
                        default='vcl', type=str)
    parser.add_argument('--not_h_threhold', dest='not_h_threhold',
                        help='not_h_threhold',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)
    # test detections result

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
    # init session
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)

    # net = ResNet50(model_name=args.model)
    # net.create_architecture(False)
    #
    #
    # saver = tf.train.Saver()
    # saver.restore(sess, weight)
    #
    # print('Pre-trained weights loaded.')
    #
    # test_net(sess, net, Test_RCNN, output_file, args.object_thres, args.human_thres)
    # sess.close()

    # Generate_HICO_detection(output_file, HICO_dir)
    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI

        net = HOI(model_name=args.model)
    stride = 200

    image, blobs, image_id = obtain_test_dataset_wo_obj(args.object_thres, args.human_thres, test_type=args.test_type,
                                                        has_human_threhold=not args.not_h_threhold,
                                                        stride=stride)
    image = image[0:1]
    print(blobs, image)

    tmp_labels = tf.one_hot(tf.reshape(tf.cast(blobs['O_cls'], tf.int32), shape=[-1, ]), 80, dtype=tf.float32)
    tmp_ho_class_from_obj = tf.cast(tf.matmul(tmp_labels, net.obj_to_HO_matrix) > 0, tf.float32)
    # action_ho = blobs['O_cls']

    net.set_ph(image, image_id, num_pos=blobs['H_num'], Human_augmented=blobs['H_boxes'],
               Object_augmented=blobs['O_boxes'],
               action_HO=None, sp=blobs['sp'],)
    # net.set_add_ph()
    # net.init_verbs_objs_cls()
    net.create_architecture(False)
    saver = tf.train.Saver()
    print(weight)
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    test_net_data_api_wo_obj(sess, net, output_file, blobs['H_boxes'][:, 1:], blobs['O_boxes'][:, 1:],
                       blobs['O_cls'], blobs['H_score'], blobs['O_score'], None, image_id, args.debug)
    sess.close()

