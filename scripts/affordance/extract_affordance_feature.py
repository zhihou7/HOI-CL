# --------------------------------------------------------

# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import argparse
import os

import numpy as np
import tensorflow as tf

from ult.config import cfg
from ult.timer import Timer
from ult.ult import obtain_data, obtain_test_data, obtain_coco_data2
tf.set_random_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=160000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='ATL_union_multi_atl_ml5_l05_t5_def2_aug5_new_VCOCO_coco_CL_21', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.3, type=float)
    parser.add_argument('--type', dest='type',
                        help='Object threshold',
                        default='train', type=str)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)

    args = parser.parse_args()
    return args


def save_img(img, target_size, name):
    import skimage.io as sio
    import skimage.transform as transform
    img = np.squeeze(img, axis=-1)
    img = transform.resize(img, target_size, order=0)
    print(img.shape)
    sio.imsave(cfg.IMAGE_TEMP+'/'+name+'.jpg', img)

if __name__ == '__main__':

    args = parse_args()

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight ) 
  
    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'

    HICO_dir = cfg.LOCAL_DATA + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    tfconfig = tf.ConfigProto(device_count={"CPU": 16},
                              inter_op_parallelism_threads=8,
                              intra_op_parallelism_threads=8,
                              allow_soft_placement=True)
    # init session
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=tfconfig)


    if args.model.__contains__('res101'):
        os.environ['DATASET'] = 'HICO_res101'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    elif args.model.__contains__('VCOCO'):
        os.environ['DATASET'] = 'VCOCO1'
        from networks.HOI import HOI
        net = HOI(model_name=args.model)
    else:
        from networks.HOI import HOI
        net = HOI(model_name=args.model)


    if args.type == 'train':
        if not args.model.__contains__('VCOCO'):
            large_neg_for_ho = False
            image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_data(
                Pos_augment=0, Neg_select=0, augment_type=-1, pattern_type=False)
            net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
        else:
            image, image_id, num_pos, blobs = obtain_coco_data2(0, 0, augment_type=-1, type=1)
            action_HO = blobs['gt_class_C']
            net.set_ph(image, image_id, num_pos, blobs['sp'], blobs['H_boxes'],
                       blobs['O_boxes'], blobs['gt_class_H'], blobs['gt_class_HO'], blobs['gt_class_sp'],
                       blobs['Mask_HO'], blobs['Mask_H'], blobs['Mask_sp'], blobs['gt_class_C'])
    else:
        image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_test_data(
            Pos_augment=0,
            Neg_select=0,
            augment_type=-1,
            large_neg_for_ho=False)
        net.set_ph(image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp)
    net.create_architecture(False)

    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    detection = {}

    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0
    img_id_list = []
    O_list = []
    V_list = []
    A_list = []
    result = {}
    result['img_id_list'] = []
    result['O_list'] = []
    result['V_list'] = []
    result['A_list'] = []
    while True:
        _t['im_detect'].tic()

        try:
            _image, _image_id, fc7_O, fc7_verbs, actions  = sess.run(
                [image, image_id,
                 net.test_visualize['fc7_O_feats'],
                 net.test_visualize['fc7_verbs_feats'],
                 action_HO,
                ])
            # print(fc7_verbs.shape, actions.shape, _image_id)
            # print(t_Human_augmented, t_Object_augmented)
            # print('---')
        except tf.errors.OutOfRangeError:
            print('END')
            break
        _t['im_detect'].toc()
        count += 1
        # print(fc7_O.shape, actions.shape)
        # result['O_list'].extend(fc7_O)
        # print(_image_id, fc7_verbs)
        result['V_list'].extend(fc7_verbs)
        result['img_id_list'].append([_image_id]*len(fc7_O))
        result['A_list'].extend(actions)
        print('im_detect: {:d}/{:d}  {:d}, {:.3f}s\r'.format(count, 9658, _image_id, _t['im_detect'].average_time))
        # if count > 10:
        #     exit()
    print(len(result['A_list']))
    import pickle
    if not os.path.exists(cfg.LOCAL_DATA+ '/feats/'):
        os.makedirs(cfg.LOCAL_DATA+ '/feats/')
    pickle.dump(result, open(cfg.LOCAL_DATA+ '/feats/' +args.model +'_'+args.type+'_'+'HOI_verb_feats.pkl', 'wb'))
    sess.close()
#