# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from iCAN and TIN
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.tools import get_convert_matrix
from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp

import cv2
import pickle
import numpy as np
import glob

import tensorflow as tf

from networks.tools import get_convert_matrix as get_cooccurence_matrix
# This strategy is based on TIN and is effective
human_num_thres = 4
object_num_thres = 4


def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape


def get_coco_blob(image_id, type='coco'):
    if type == 'hico' or type == 'gthico':
        im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(
            8) + '.jpg'
    elif type == 'hico_train':
        im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(
            8) + '.jpg'
    elif type == 'gtobj365_coco' or type == 'gtobj365':
        im_file = cfg.LOCAL_DATA + '/dataset/Objects365/Images/val/val/obj365_val_' + (str(image_id)).zfill(
            12) + '.jpg'
    elif type == 'vcoco_train':
        im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(
            12) + '.jpg'
    else:
        im_file  = cfg.LOCAL_DATA + '/dataset/coco/images/val2017/'+(str(image_id)).zfill(12) + '.jpg'
    # print(im_file)
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection):

    # save image information
    This_image = []

    im_orig, im_shape = get_blob(image_id)

    blobs = {}
    blobs['H_num']       = 1

    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human

            blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object

                    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                    blobs['sp']      = Get_next_sp(Human_out[2], Object[2]).reshape(1, 64, 64, 2)
                    mask = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
                    obj_box = blobs['O_boxes'][0][1:].astype(np.int32)
                    # print(obj_box)
                    # print(obj_box, blobs['O_boxes'])
                    mask[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                    blobs['O_mask'] = mask
                    print(image_id, blobs); exit()
                    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
                    prediction_HO, pH, pO, pSp, pVerbs = net.obtain_all_preds(sess, im_orig, blobs)
                    # print("DEBUG:", type(prediction_HO), len(prediction_HO), prediction_HO[0].shape, prediction_HO[0][0].shape)

                    temp = []
                    temp.append(Human_out[2])           # Human box
                    temp.append(Object[2])              # Object box
                    temp.append(Object[4])              # Object class
                    temp.append(prediction_HO[0])     # Score
                    temp.append(Human_out[5])           # Human score
                    temp.append(Object[5])              # Object score
                    temp.append(pH[0])                  # 6
                    temp.append(pO[0])
                    temp.append(pSp[0])
                    temp.append(pVerbs[0])
                    This_image.append(temp)

    detection[image_id] = This_image


def test_net(sess, net, Test_RCNN, output_dir, object_thres, human_thres):


    np.random.seed(cfg.RNG_SEED)
    detection = {}
    count = 0

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):

        _t['im_detect'].tic()

        image_id   = int(line[-9:-4])

        im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:d} {:.3f}s'.format(count + 1, 9658, image_id, _t['im_detect'].average_time))
        count += 1
        # if count > 10:  # TODO remove
        #     pickle.dump(detection, open('test_orig.pkl', 'wb'))
        #     return

    pickle.dump( detection, open( output_dir, "wb" ) )


def test_net_data_api1(sess, net, output_dir, h_box, o_box, o_cls, h_score, o_score, image_id, convert_matrix = None):
    detection = {}
    verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix()
    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    ones = np.ones([1, 600], np.float32)
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0

    fuse_res = tf.constant(0)

    obj_scores = tf.constant(0)
    objid = tf.constant(0)
    obj_scores = tf.constant(0)
    sp_preds = net.predictions["cls_prob_sp"] if 'cls_prob_sp' in net.predictions else h_box
    hoi_preds = net.predictions["cls_prob_hoi"] if 'cls_prob_hoi' in net.predictions else h_box
    if net.model_name.__contains__('ICL'):
        if 'cls_prob_sp' in net.predictions:
            # 480 -> 600
            sp_preds = tf.matmul(sp_preds, net.zs_convert_matrix_base, transpose_b=True)
            zs_hois = tf.reduce_sum(net.zs_convert_matrix_base, axis=1) > 0
            zs_hois = tf.cast(zs_hois, tf.float32)
            zs_hois = 1 - zs_hois
            sp_preds = tf.add(zs_hois, sp_preds)
        if net.model_name.__contains__('affordance'):
            # affordance_stat
            # reader = tf.train.NewCheckpointReader(file_name)
            # affordance_stat = reader.get_tensor('affordance_stat')
            # affordance_count = reader.get_tensor('affordance_count')
            # import ipdb;ipdb.set_trace()
            if net.model_name.__contains__('VERB'):
                verb_preds = net.predictions["cls_prob_verbs_VERB"]
                hoi_preds = tf.matmul(verb_preds, tf.constant(convert_matrix))
            else:
                # import ipdb;ipdb.set_trace()
                hoi_preds_orig = hoi_preds
                verb_preds = tf.matmul(hoi_preds, net.verb_to_HO_matrix, transpose_b=True) / tf.reduce_sum(net.verb_to_HO_matrix, axis=-1)
                hoi_preds_unknown = tf.matmul(verb_preds, tf.constant(convert_matrix))
                hoi_preds = tf.matmul(hoi_preds, net.zs_convert_matrix, transpose_b=True)  # 600
                known_hois = tf.matmul(tf.ones_like(hoi_preds_orig, tf.float32), net.zs_convert_matrix, transpose_b=True) # 480 -> 600: 0 for non
                unknown_hois = 1. - known_hois
                hoi_preds = hoi_preds + unknown_hois * hoi_preds_unknown

            # to 600.
            pass

        else:
            hoi_preds = tf.matmul(hoi_preds, net.zs_convert_matrix, transpose_b=True)

    fuse_res = tf.constant(0)
    if "cls_prob_sp" in net.predictions:
        fuse_res = tf.multiply(sp_preds, hoi_preds)
    else:
        fuse_res = net.predictions["cls_prob_verbs"]

    _t['im_detect'].tic()
    while True:
        _t['im_detect'].tic()

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:

            pH, pO, pSp, pVerbs, pSpHO, pFuse, f_obj_score, f_obj_cls, _h_box, _o_box, _o_cls, _h_score, _o_score, _image_id = sess.run([
                net.predictions["cls_prob_H"] if 'cls_prob_H' in net.predictions else h_box,
                net.predictions["cls_prob_O"] if 'cls_prob_O' in net.predictions else h_box,
                sp_preds,
                hoi_preds,
                net.predictions["cls_prob_spverbs"] if 'cls_prob_spverbs' in net.predictions else h_box,
                fuse_res if 'cls_prob_sp' in net.predictions else h_box, obj_scores, objid,
                                        h_box, o_box, o_cls, h_score, o_score, image_id])
            # print(pFuse.shape, f_obj_score.shape, f_obj_cls.shape)
        except InvalidArgumentError as e:
            # cls_prob_HO = np.zeros(shape=[blobs['sp'].shape[0], self.num_classes])
            raise e
        except tf.errors.OutOfRangeError:
            print('END')
            break

        temp = [[_h_box[i], _o_box[i], _o_cls[i], 0, _h_score[i], _o_score[i], pH[i], pO[i], pSp[i], pVerbs[i], pSpHO[i]] for i in range(len(_h_box))]

        # detection[_image_id] = temp
        if _image_id in detection:
            detection[_image_id].extend(temp)
        else:
            detection[_image_id] = temp

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 10566, _image_id, _t['im_detect'].average_time), end='', flush=True)

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    pickle.dump(detection, open(output_dir, "wb"))
    del detection
    import gc
    gc.collect()




def obtain_test_dataset_fusion(object_thres, human_thres, dataset_name='test2015', test_type='default', has_human_threhold=True, stride = 200):
    print('================================================================================')
    print(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg', glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'))
    from sys import version_info
    if dataset_name == 'test2015':
        if version_info.major == 3:
            # Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose
            Test_RCNN = obtain_obj_boxes(test_type)
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        # for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
        for image_id in Test_RCNN:
            i += 1
            # if i > 30: # TODO remove
            #     break
            # image_id = int(line[-9:-4])
            # save image information
            im_orig, im_shape = get_blob(image_id)
            mask_all = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_mask'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            blobs['O_all_score'] = []
            blobs['pose_box'] = []

            human_out_list = sorted([H for H in Test_RCNN[image_id] if H[1] == 'Human'], key=lambda x: x[5], reverse=True)
            obj_out_list = sorted([H for H in Test_RCNN[image_id]], key=lambda x: x[5], reverse=True)

            # for Human_out in Test_RCNN[image_id]:
            for Human_out in human_out_list:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in obj_out_list:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            mask = np.zeros(shape=(im_shape[0]// 16, im_shape[1]// 16, 1), dtype=np.float32)
                            obj_box = obj_box[1:].astype(np.int32)
                            mask_all[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                            blobs['O_mask'].append(mask)
                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])
                            if len(Object) >= 7 :
                                blobs['O_all_score'].append(Object[6])
                            else:
                                blobs['O_all_score'].append(np.ones([80], np.float32))

            if blobs['H_num'] == 0 and has_human_threhold:
                print('\rDealing with zero-sample test Image ' + str(image_id), end='', flush=True)

                list_human_included = []
                list_object_included = []
                Human_out_list = []
                Object_list = []

                test_pair_all = obj_out_list
                length = len(test_pair_all)

                flag_continue_searching = 1

                while (len(list_human_included) < human_num_thres) or (
                        len(list_object_included) < object_num_thres):
                    h_max = [-1, -1.0]
                    o_max = [-1, -1.0]
                    flag_continue_searching = 0
                    for i in range(length):
                        if test_pair_all[i][1] == 'Human':
                            if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(
                                    list_human_included) < human_num_thres:
                                h_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1
                        else:
                            if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(
                                    list_object_included) < object_num_thres:
                                o_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1

                    if flag_continue_searching == 0:
                        break

                    list_human_included.append(h_max[0])
                    list_object_included.append(o_max[0])

                    Human_out_list.append(test_pair_all[h_max[0]])
                    Object_list.append(test_pair_all[o_max[0]])

                for Human_out in Human_out_list:
                    for Object in Object_list:

                        blobs['H_num'] += 1
                        blobs['H_boxes'].append(
                            np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                        obj_box = np.array(
                            [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                        blobs['O_boxes'].append(obj_box)
                        blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                        mask = np.zeros(shape=(im_shape[0]// 16, im_shape[1]// 16, 1), dtype=np.float32)
                        obj_box = obj_box[1:].astype(np.int32)
                        # print(obj_box)
                        # print(obj_box, blobs['O_boxes'])
                        # mask[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                        mask_all[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                        # from skimage import transform
                        # mask = transform.resize(mask, [im_shape[0] // 16, im_shape[1] // 16, 1], order=0,
                        #                         preserve_range=True)
                        blobs['O_mask'].append(mask)
                        blobs['O_cls'].append(Object[4])
                        blobs['H_score'].append(Human_out[5])
                        blobs['O_score'].append(Object[5])
                        # blobs['O_all_score'].append(Object[6])
                        if len(Object) >= 7:
                            blobs['O_all_score'].append(Object[6])
                        else:
                            blobs['O_all_score'].append(np.ones([80], np.float32))

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue

            im_mask = np.multiply(im_orig, mask_all)
            im_orig = np.concatenate([im_orig, im_mask], axis=0)
            start = 0
            # stride = 200
            while start < blobs['H_num']:
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]


                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride
                # print('b_temp' , im_orig.shape, image_id, end=' ')
                # for k, v in blobs.items():
                #     if not k == 'H_num':
                #         blobs[k] = np.asarray(v)
                #         print(k, blobs[k].shape, end=' ')
                # print('\n')
                yield im_orig, b_temp, image_id
            # yield im_orig, blobs, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32, 'O_mask': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32, 'O_all_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([2, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_mask': tf.TensorShape([None, None, None, 1]), # useless
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None]),
                                                             'O_all_score': tf.TensorShape([None, 80])},
                                                            tf.TensorShape([]))
                                             )

    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000) # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id




def obtain_test_dataset_fcl(object_thres, human_thres, dataset_name='test2015', test_type='vcl',
                            has_human_threhold=True, stride = 200, model_name='', pattern_type=0):
    print('================================================================================')
    print(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg', glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'))
    from sys import version_info
    if dataset_name == 'test2015':
        print(test_type, version_info.major)
        if version_info.major == 3:
            # Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose
            Test_RCNN = obtain_obj_boxes(test_type)
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    assert pattern_type == 0, "we remove the pose pattern, if you want to add new pattern type, just remove this line"
    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        # for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
        for image_id in Test_RCNN:
            i += 1
            # if i > 30: # TODO remove
            #     break
            im_orig, im_shape = get_blob(image_id)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human
                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            assert Object[4] > 0, (Object[4])
                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0 and has_human_threhold:
                # copy from previous work (TIN). This is useless for better object detector.
                # This also illustrates the importance of fine-tuned object detector!
                print('\rDealing with zero-sample test Image ' + str(image_id), end='', flush=True)

                list_human_included = []
                list_object_included = []
                Human_out_list = []
                Object_list = []

                test_pair_all = Test_RCNN[image_id]
                length = len(test_pair_all)


                while (len(list_human_included) < human_num_thres) or (
                        len(list_object_included) < object_num_thres):
                    h_max = [-1, -1.0]
                    o_max = [-1, -1.0]
                    flag_continue_searching = 0
                    for i in range(length):
                        if test_pair_all[i][1] == 'Human':
                            if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(
                                    list_human_included) < human_num_thres:
                                h_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1
                        else:
                            if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(
                                    list_object_included) < object_num_thres:
                                o_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1

                    if flag_continue_searching == 0:
                        break

                    list_human_included.append(h_max[0])
                    list_object_included.append(o_max[0])

                    Human_out_list.append(test_pair_all[h_max[0]])
                    Object_list.append(test_pair_all[o_max[0]])

                for Human_out in Human_out_list:
                    for Object in Object_list:

                        blobs['H_num'] += 1
                        blobs['H_boxes'].append(
                            np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                        obj_box = np.array(
                            [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                        blobs['O_boxes'].append(obj_box)
                        blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                        blobs['O_cls'].append(Object[4])
                        blobs['H_score'].append(Human_out[5])
                        blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue

            start = 0
            # stride = 200
            while start < blobs['H_num']:
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]


                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride
                yield im_orig, b_temp, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32,}, tf.int32,),
                                             output_shapes=(tf.TensorShape([1, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None])},
                                                            tf.TensorShape([]))
                                             )

    dataset = dataset.prefetch(100)
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id


def test_net_data_api_fuse_obj(sess, net, output_dir, h_box, o_box, o_cls, h_score, o_score, o_all_score, image_id, debug_type = 0):
    detection = {}
    verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix()
    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    ones = np.ones([1, 600], np.float32)
    _t = {'im_detect': Timer(), 'misc': Timer()}
    label_trans_map = {0: 4, 1: 47, 2: 24, 3: 46, 4: 34, 5: 35, 6: 21, 7: 59, 8: 13, 9: 1, 10: 14, 11: 8, 12: 73,
                       13: 39, 14: 45, 15: 50, 16: 5, 17: 55, 18: 2, 19: 51, 20: 15, 21: 67, 22: 56, 23: 74, 24: 57,
                       25: 19, 26: 41, 27: 60, 28: 16, 29: 54, 30: 20, 31: 10, 32: 42, 33: 29, 34: 23, 35: 78, 36: 26,
                       37: 17, 38: 52, 39: 66, 40: 33, 41: 43, 42: 63, 43: 68, 44: 3, 45: 64, 46: 49, 47: 69, 48: 12,
                       49: 0, 50: 53, 51: 58, 52: 72, 53: 65, 54: 48, 55: 76, 56: 18, 57: 71, 58: 36, 59: 30, 60: 31,
                       61: 44, 62: 32, 63: 11, 64: 28, 65: 37, 66: 77, 67: 38, 68: 27, 69: 70, 70: 61, 71: 79, 72: 9,
                       73: 6, 74: 7, 75: 62, 76: 25, 77: 75, 78: 40, 79: 22}
    hoi_to_coco_obj = np.zeros([80, 80], np.float32)
    for k in label_trans_map:
        hoi_to_coco_obj[k][label_trans_map[k]] = 1.
    count = 0
    if "cls_prob_sp" in net.predictions:
        fuse_res = tf.multiply(net.predictions["cls_prob_sp"], net.predictions["cls_prob_verbs"])
    else:
        fuse_res = net.predictions["cls_prob_verbs"]

    obj_scores = tf.matmul(fuse_res, obj_to_HO_matrix, transpose_b=True) / tf.matmul(ones, obj_to_HO_matrix,
                                                                                     transpose_b=True)
    obj_scores = tf.matmul(obj_scores, hoi_to_coco_obj)
    if debug_type == 0:
        obj_scores = obj_scores * 0.5 + o_all_score * 0.5
    elif debug_type == 1:
        obj_scores = obj_scores * 0.3 + o_all_score * 0.7
    objid = tf.argmax(obj_scores, axis=-1)
    obj_scores = tf.reduce_max(obj_scores, axis=-1)
    _t['im_detect'].tic()
    while True:
        _t['im_detect'].tic()

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:
            pH, pO, pSp, pVerbs, pSpHO, pFuse, f_obj_score, f_obj_cls, _h_box, _o_box, _o_cls, _h_score, _o_score, _image_id = sess.run([
                net.predictions["cls_prob_H"] if 'cls_prob_H' in net.predictions else h_box,
                net.predictions["cls_prob_O"] if 'cls_prob_O' in net.predictions else h_box,
                net.predictions["cls_prob_sp"] if 'cls_prob_sp' in net.predictions else h_box,
                net.predictions["cls_prob_verbs"] if 'cls_prob_verbs' in net.predictions else h_box,
                net.predictions["cls_prob_spverbs"] if 'cls_prob_spverbs' in net.predictions else h_box,
                fuse_res if 'cls_prob_sp' in net.predictions else h_box, obj_scores, objid,
                                        h_box, o_box, o_cls, h_score, o_score, image_id])
            # print(pFuse.shape, f_obj_score.shape, f_obj_cls.shape)
        except InvalidArgumentError as e:
            # cls_prob_HO = np.zeros(shape=[blobs['sp'].shape[0], self.num_classes])
            if net.model_name.__contains__('lamb'):
                print('InvalidArgumentError', sess.run([image_id]))
                continue
            else:
                raise e
        except tf.errors.OutOfRangeError:
            print('END')
            break
        # import ipdb;ipdb.set_trace()
        temp = [[_h_box[i], _o_box[i], _o_cls[i], 0, _h_score[i], _o_score[i], 0, 0, pSp[i], pVerbs[i], 0, pFuse[i], f_obj_score[i], f_obj_cls[i]] for i in range(len(_h_box))]

        # detection[_image_id] = temp
        if _image_id in detection:
            detection[_image_id].extend(temp)
        else:
            detection[_image_id] = temp

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 10566, _image_id, _t['im_detect'].average_time), end='', flush=True)
        # if net.model_name.__contains__('_debugrl_') or 'DEBUG_NET' in os.environ:
        #     if count >= 1:
        #         print(temp)
        #         break

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    pickle.dump(detection, open(output_dir, "wb"))
    del detection
    import gc
    gc.collect()



def test_net_data_fcl(sess, net, output_dir, h_box, o_box, o_cls, h_score, o_score, image_id):
    detection = {}
    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0


    _t['im_detect'].tic()
    while True:
        _t['im_detect'].tic()

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:
            pH, pO, pSp, pVerbs, _h_box, _o_box, _o_cls, _h_score, _o_score, _image_id = sess.run([
                net.predictions["cls_prob_H"] if 'cls_prob_H' in net.predictions else h_box, # from previous work
                net.predictions["cls_prob_O"] if 'cls_prob_O' in net.predictions else h_box,
                net.predictions["cls_prob_sp"] if 'cls_prob_sp' in net.predictions else h_box,
                net.predictions["cls_prob_hoi"] if 'cls_prob_hoi' in net.predictions else h_box,
                h_box, o_box, o_cls, h_score, o_score, image_id])
        except InvalidArgumentError as e:
            # cls_prob_HO = np.zeros(shape=[blobs['sp'].shape[0], self.num_classes])
            if net.model_name.__contains__('lamb'):
                print('InvalidArgumentError', sess.run([image_id]))
                continue
            else:
                raise e
        except tf.errors.OutOfRangeError:
            print('END')
            break
        # additional predictions are for ablated study.
        temp = [[_h_box[i], _o_box[i], _o_cls[i], 0, _h_score[i], _o_score[i], pH[i], pO[i], pSp[i], pVerbs[i], 0] for i in range(len(_h_box))]

        # detection[_image_id] = temp
        if _image_id in detection:
            detection[_image_id].extend(temp)
        else:
            detection[_image_id] = temp

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 10566, _image_id, _t['im_detect'].average_time), end='', flush=True)
        # if net.model_name.__contains__('_debugrl_') or 'DEBUG_NET' in os.environ:
        #     if count >= 1:
        #         print(temp)
        #         break

    pickle.dump(detection, open(output_dir, "wb"))
    del detection
    import gc
    gc.collect()


def obtain_test_dataset_with_obj(object_thres, human_thres, dataset_name='test2015', test_type='vcl',
                                 has_human_threhold=True, stride = 200, hoi_nums=1, model_name=''):
    print('================================================================================')
    from sys import version_info
    if version_info.major == 3:
        # Test_RCNN = obtain_obj_boxes('train') # this is useless for object
        if test_type == 'gtval2017':
            Test_RCNN_coco = pickle.load(open(cfg.LOCAL_DATA + '/Test_GT_VCOCO_COCO_VCOCO_coco.pkl', "rb"))
        elif test_type == 'gtobj365_coco':
            # 116841
            Test_RCNN_coco = pickle.load(open(cfg.LOCAL_DATA + '/Test_GT_VCOCO_COCO_VCOCO_obj365_coco.pkl', "rb"))
        elif test_type == 'gtobj365':
            Test_RCNN_coco = pickle.load(open(cfg.LOCAL_DATA + '/Test_GT_VCOCO_COCO_VCOCO_obj365.pkl', "rb"))
        elif test_type == 'gthico':
            # we simply use the Ground truth HICO-DET test. This might contain repeats objects.
            # However this do not affect the comparison between ATL and Baseline
            Test_RCNN_coco = pickle.load(open(cfg.LOCAL_DATA + '/Test_GT_HICO_COCO_HICO.pkl', "rb"))
        else:
            raise Exception('no test_type {}'.format(test_type))
        #
        # print('sdf val all')
    print('end load', test_type)

    # keys = list(Test_RCNN.keys())
    print(len(Test_RCNN_coco.keys()), 'length')
    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        # for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
        for coco_image_id in Test_RCNN_coco:
            for Object in Test_RCNN_coco[coco_image_id]:
                if not (np.max(Object[5]) > object_thres):
                    continue
                if test_type == 'gtobj365' and int(Object[4]) not in [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183, 300, 225, 282, 335]:
                    # 29: [8, 37, 51, 68, 115, ],  # boots
                    # 216: [1, 5],  # ship
                    # 23: [8, 37, 39, 51, 68],  # flower
                    # 183: [3, 37, 45, 51, 68, 105, ],  # basketball
                    # 300: [8, 16, 24, 37, 51, 55, 68],  # cheese
                    # 225: [8, 16, 24, 37, 51],  # watermelon
                    # 282: [27, 37, 77, 88, 111, 112, 113],  # camel
                    # 335: [27, 37, 77, 88, 111, 112, 113]  # lion
                    continue
                coco_im_orig, coco_im_shape = get_coco_blob(coco_image_id, test_type)
                # coco_im_orig, coco_im_shape = get_blob(coco_image_id)
                blobs = {}

                blobs['H_num'] = 0
                blobs['H_boxes'] = []
                blobs['O_boxes'] = []
                blobs['sp'] = []

                blobs['O_cls'] = []
                blobs['H_score'] = []
                blobs['O_score'] = []

                blobs['H_num'] += 2
                blobs['H_boxes'].append(np.array([0, 0, 0, 0, 0]))

                obj_box = np.array(
                    [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                blobs['O_boxes'].append(obj_box)
                # print(len(Human_out), len(Human_out[6]))

                blobs['sp'].append(np.zeros([64, 64, 2], np.float32))
                blobs['O_cls'].append(Object[4])
                blobs['H_score'].append(0) # we do not use this here
                blobs['O_score'].append(Object[5])

                yield coco_im_orig, blobs, coco_image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([1, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None])},
                                                            tf.TensorShape([]))
                                             )
    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000) # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id


def test_net_data_api_wo_obj(sess, net, output_dir, h_box, o_box, o_cls, h_score, o_score, o_all_score, image_id, debug_type = 0):
    detection = {}
    verb_to_HO_matrix, obj_to_HO_matrix = get_cooccurence_matrix()
    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    ones = np.ones([1, 600], np.float32)
    _t = {'im_detect': Timer(), 'misc': Timer()}
    # convert HICO object to COCO objects.
    label_trans_map = {0: 4, 1: 47, 2: 24, 3: 46, 4: 34, 5: 35, 6: 21, 7: 59, 8: 13, 9: 1, 10: 14, 11: 8, 12: 73,
                       13: 39, 14: 45, 15: 50, 16: 5, 17: 55, 18: 2, 19: 51, 20: 15, 21: 67, 22: 56, 23: 74, 24: 57,
                       25: 19, 26: 41, 27: 60, 28: 16, 29: 54, 30: 20, 31: 10, 32: 42, 33: 29, 34: 23, 35: 78, 36: 26,
                       37: 17, 38: 52, 39: 66, 40: 33, 41: 43, 42: 63, 43: 68, 44: 3, 45: 64, 46: 49, 47: 69, 48: 12,
                       49: 0, 50: 53, 51: 58, 52: 72, 53: 65, 54: 48, 55: 76, 56: 18, 57: 71, 58: 36, 59: 30, 60: 31,
                       61: 44, 62: 32, 63: 11, 64: 28, 65: 37, 66: 77, 67: 38, 68: 27, 69: 70, 70: 61, 71: 79, 72: 9,
                       73: 6, 74: 7, 75: 62, 76: 25, 77: 75, 78: 40, 79: 22}
    hoi_to_coco_obj = np.zeros([80, 80], np.float32)
    for k in label_trans_map:
        hoi_to_coco_obj[k][label_trans_map[k]] = 1.
    last_img_id = -1
    count = 0
    if "cls_prob_sp" in net.predictions:
        fuse_res = tf.multiply(net.predictions["cls_prob_sp"], net.predictions["cls_prob_verbs"])
    else:
        fuse_res = net.predictions["cls_prob_verbs"]

    obj_scores = tf.matmul(fuse_res, obj_to_HO_matrix, transpose_b=True) / tf.matmul(ones, obj_to_HO_matrix,
                                                                                     transpose_b=True)
    obj_scores = tf.matmul(obj_scores, hoi_to_coco_obj)

    # obj_scores = obj_scores *  o_all_score
    objid = tf.argmax(obj_scores, axis=-1)
    obj_scores = tf.reduce_max(obj_scores, axis=-1)
    # obj_scores = tf.constant(1.0)
    _t['im_detect'].tic()
    while True:
        _t['im_detect'].tic()

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:
            pH, pO, pSp, pVerbs, pSpHO, pFuse, f_obj_score, f_obj_cls, _h_box, _o_box, _o_cls, _h_score, _o_score, _image_id = sess.run([
                net.predictions["cls_prob_H"] if 'cls_prob_H' in net.predictions else h_box,
                net.predictions["cls_prob_O"] if 'cls_prob_O' in net.predictions else h_box,
                net.predictions["cls_prob_sp"] if 'cls_prob_sp' in net.predictions else h_box,
                net.predictions["cls_prob_verbs"] if 'cls_prob_verbs' in net.predictions else h_box,
                net.predictions["cls_prob_spverbs"] if 'cls_prob_spverbs' in net.predictions else h_box,
                fuse_res if 'cls_prob_sp' in net.predictions else h_box, obj_scores, objid,
                                        h_box, o_box, o_cls, h_score, o_score, image_id])
        except InvalidArgumentError as e:
            if net.model_name.__contains__('lamb'):
                print('InvalidArgumentError', sess.run([image_id]))
                continue
            else:
                raise e
        except tf.errors.OutOfRangeError:
            print('END')
            break
        temp = [[_h_box[i], _o_box[i], _o_cls[i], 0, _h_score[i], _o_score[i], 0, 0, pSp[i], pVerbs[i], 0, pFuse[i], f_obj_score[i], f_obj_cls[i]] for i in range(len(_h_box))]

        # detection[_image_id] = temp
        if _image_id in detection:
            detection[_image_id].extend(temp)
        else:
            detection[_image_id] = temp

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 10566, _image_id, _t['im_detect'].average_time), end='', flush=True)
        if net.model_name.__contains__('_debugrl_'):
            if count >= 20:
                break

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    pickle.dump(detection, open(output_dir, "wb"))
    del detection
    import gc
    gc.collect()



def obtain_test_dataset(object_thres, human_thres, dataset_name='test2015'):
    from sys import version_info
    if dataset_name == 'test2015':
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"),
                                    encoding='latin1')
            Test_RCNN = obtain_obj_boxes('default')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
            i += 1
            # if i > 30: # TODO remove
            #     break
            image_id = int(line[-9:-4])
            # save image information
            im_orig, im_shape = get_blob(image_id)
            mask_all = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))

                            mask = np.zeros(shape=(im_shape[0], im_shape[1], 1), dtype=np.float32)
                            obj_box = obj_box[1:].astype(np.int32)
                            # print(obj_box)
                            # print(obj_box, blobs['O_boxes'])
                            mask[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                            mask_all[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                            # from skimage import transform
                            # mask = transform.resize(mask, [im_shape[0] // 16, im_shape[1] // 16, 1], order=0,
                            #                         preserve_range=True)
                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue
            # print(im_orig.shape, image_id, end=' ')
            # for k, v in blobs.items():
            #     if not k == 'H_num':
            #         blobs[k] = np.asarray(v)
            #         print(k, blobs[k].shape, end=' ')
            # print('\n')
            im_mask = np.multiply(im_orig, mask_all)
            im_orig = np.concatenate([im_orig, im_mask], axis=0)
            start = 0
            stride = 200
            while start < blobs['H_num']: # save GPU memory
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]

                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride
                # print('b_temp' , im_orig.shape, image_id, end=' ')
                # for k, v in blobs.items():
                #     if not k == 'H_num':
                #         blobs[k] = np.asarray(v)
                #         print(k, blobs[k].shape, end=' ')
                # print('\n')

                yield im_orig, b_temp, image_id
            # yield im_orig, blobs, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([2, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None])},
                                                            tf.TensorShape([]))
                                             )
    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000) # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id


def obtain_test_dataset1(object_thres, human_thres, dataset_name='test2015', test_type='default',
                         has_human_threhold=True, stride = 200, model_name=''):
    print('================================================================================')
    print(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg', glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'))
    from sys import version_info
    if dataset_name == 'test2015':
        print(test_type, version_info.major)
        if version_info.major == 3:
            # Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose
            Test_RCNN = obtain_obj_boxes(test_type)
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        # for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
        for image_id in Test_RCNN:
            i += 1
            # if i > 30: # TODO remove
            #     break
            # image_id = int(line[-9:-4])
            # save image information
            im_orig, im_shape = get_blob(image_id)
            mask_all = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            # print(len(Human_out), len(Human_out[6]))
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))

                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            """
            Notice: This strategy is based on TIN and is effective! This could improve the performance around 1.%
            Re-weighting and This strategy improve our baseline to 18.04%.
            """
            if blobs['H_num'] == 0 and has_human_threhold:
                print('\rDealing with zero-sample test Image ' + str(image_id), end='', flush=True)

                list_human_included = []
                list_object_included = []
                Human_out_list = []
                Object_list = []

                test_pair_all = Test_RCNN[image_id]
                length = len(test_pair_all)

                flag_continue_searching = 1

                while (len(list_human_included) < human_num_thres) or (
                        len(list_object_included) < object_num_thres):
                    h_max = [-1, -1.0]
                    o_max = [-1, -1.0]
                    flag_continue_searching = 0
                    for i in range(length):
                        if test_pair_all[i][1] == 'Human':
                            if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(
                                    list_human_included) < human_num_thres:
                                h_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1
                        else:
                            if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(
                                    list_object_included) < object_num_thres:
                                o_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1

                    if flag_continue_searching == 0:
                        break

                    list_human_included.append(h_max[0])
                    list_object_included.append(o_max[0])

                    Human_out_list.append(test_pair_all[h_max[0]])
                    Object_list.append(test_pair_all[o_max[0]])

                for Human_out in Human_out_list:
                    for Object in Object_list:

                        blobs['H_num'] += 1
                        blobs['H_boxes'].append(
                            np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                        obj_box = np.array(
                            [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                        blobs['O_boxes'].append(obj_box)
                        blobs['sp'].append(Get_next_sp(Human_out[2], Object[2], pattern_type))
                        blobs['O_cls'].append(Object[4])
                        blobs['H_score'].append(Human_out[5])
                        blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue
            start = 0
            # stride = 200
            while start < blobs['H_num']:
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]

                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride
                yield im_orig, b_temp, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([1, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None])},
                                                            tf.TensorShape([]))
                                             )

    dataset = dataset.prefetch(100)
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id

def obtain_obj_boxes(test_type):
    if test_type == 'vcl':
        # from VCL
        Test_RCNN = pickle.load(open(cfg.LOCAL_DATA + '/Test_HICO_res101_3x_FPN_hico.pkl', "rb"))
    elif test_type == 'drg':
        # from DRG
        Test_RCNN = pickle.load(open(cfg.LOCAL_DATA + '/test_HICO_finetuned_v3.pkl', 'rb'))
        pass
    elif test_type == 'gt':
        Test_RCNN = pickle.load(open(cfg.LOCAL_DATA + '/gt_annotations.pkl', 'rb'))
    elif test_type == 'coco50':
        Test_RCNN = pickle.load(open(cfg.LOCAL_DATA + '/Test_HICO_res50_coco_FPN_hico.pkl', 'rb'))
    elif test_type == 'coco101':
        print('coco101')
        Test_RCNN = pickle.load(open(cfg.LOCAL_DATA + '/Test_HICO_res101_coco101_FPN_hico.pkl', 'rb'))
    elif test_type == 'iCAN':
        Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"),
                                encoding='latin1')
    else:
        Test_RCNN = pickle.load(open(cfg.LOCAL_DATA + '/pkl/Test_HICO_res101_3x_FPN_hico.pkl', "rb"))
    return Test_RCNN


def obtain_test_dataset2(object_thres, human_thres, dataset_name='test2015', test_type='default'):
    from sys import version_info
    if dataset_name == 'test2015':
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"),
                                    encoding='latin1')
            # Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose
            if test_type != 'default':
                Test_RCNN = pickle.load(
                    open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_DET_finetune.pkl', "rb"),
                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
            i += 1
            # if i > 30: # TODO remove
            #     break
            image_id = int(line[-9:-4])
            # save image information
            im_orig, im_shape = get_blob(image_id)
            mask_all = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            # blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]))


                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                print('\rDealing with zero-sample test Image ' + str(image_id), end='', flush=True)

                list_human_included = []
                list_object_included = []
                Human_out_list = []
                Object_list = []

                test_pair_all = Test_RCNN[image_id]
                length = len(test_pair_all)

                flag_continue_searching = 1

                while (len(list_human_included) < human_num_thres) or (
                        len(list_object_included) < object_num_thres):
                    h_max = [-1, -1.0]
                    o_max = [-1, -1.0]
                    flag_continue_searching = 0
                    for i in range(length):
                        if test_pair_all[i][1] == 'Human':
                            if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(
                                    list_human_included) < human_num_thres:
                                h_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1
                        else:
                            if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(
                                    list_object_included) < object_num_thres:
                                o_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1

                    if flag_continue_searching == 0:
                        break

                    list_human_included.append(h_max[0])
                    list_object_included.append(o_max[0])

                    Human_out_list.append(test_pair_all[h_max[0]])
                    Object_list.append(test_pair_all[o_max[0]])

                for Human_out in Human_out_list:
                    for Object in Object_list:

                        blobs['H_num'] += 1
                        blobs['H_boxes'].append(
                            np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                        obj_box = np.array(
                            [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                        blobs['O_boxes'].append(obj_box)
                        blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            # blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]))


                        blobs['O_cls'].append(Object[4])
                        blobs['H_score'].append(Human_out[5])
                        blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue
            # print(im_orig.shape, image_id, end=' ')
            # for k, v in blobs.items():
            #     if not k == 'H_num':
            #         blobs[k] = np.asarray(v)
            #         print(k, blobs[k].shape, end=' ')
            # print('\n')
            im_mask = np.multiply(im_orig, mask_all)
            im_orig = np.concatenate([im_orig, im_mask], axis=0)
            start = 0
            stride = 200
            while start < blobs['H_num']:
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]

                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride

                yield im_orig, b_temp, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([2, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None])},
                                                            tf.TensorShape([]))
                                             )
    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000) # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id


def obtain_test_dataset_wo_obj(object_thres, human_thres, dataset_name='test2015', test_type='default', has_human_threhold=True, stride = 200):
    print('================================================================================')
    print(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg', glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'))
    from sys import version_info
    if dataset_name == 'test2015':
        if version_info.major == 3:
            Test_RCNN = obtain_obj_boxes(test_type)
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        for image_id in Test_RCNN:
            i += 1
            im_orig, im_shape = get_blob(image_id)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            blobs['O_all_score'] = []

            human_out_list = sorted([H for H in Test_RCNN[image_id] if H[1] == 'Human'], key=lambda x: x[5], reverse=True)
            obj_out_list = sorted([H for H in Test_RCNN[image_id]], key=lambda x: x[5], reverse=True)

            # for Human_out in Test_RCNN[image_id]:
            for Human_out in human_out_list:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in obj_out_list:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0 and has_human_threhold:
                print('\rDealing with zero-sample test Image ' + str(image_id), end='', flush=True)

                list_human_included = []
                list_object_included = []
                Human_out_list = []
                Object_list = []

                test_pair_all = obj_out_list
                length = len(test_pair_all)

                flag_continue_searching = 1

                while (len(list_human_included) < human_num_thres) or (
                        len(list_object_included) < object_num_thres):
                    h_max = [-1, -1.0]
                    o_max = [-1, -1.0]
                    flag_continue_searching = 0
                    for i in range(length):
                        if test_pair_all[i][1] == 'Human':
                            if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(
                                    list_human_included) < human_num_thres:
                                h_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1
                        else:
                            if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(
                                    list_object_included) < object_num_thres:
                                o_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1

                    if flag_continue_searching == 0:
                        break

                    list_human_included.append(h_max[0])
                    list_object_included.append(o_max[0])

                    Human_out_list.append(test_pair_all[h_max[0]])
                    Object_list.append(test_pair_all[o_max[0]])

                for Human_out in Human_out_list:
                    for Object in Object_list:

                        blobs['H_num'] += 1
                        blobs['H_boxes'].append(
                            np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                        obj_box = np.array(
                            [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                        blobs['O_boxes'].append(obj_box)
                        blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))

                        blobs['O_cls'].append(Object[4])
                        blobs['H_score'].append(Human_out[5])
                        blobs['O_score'].append(Object[5])
                        if len(Object) >= 7:
                            blobs['O_all_score'].append(Object[6])
                        else:
                            blobs['O_all_score'].append(np.ones([80], np.float32))

            if blobs['H_num'] == 0:
                continue
            start = 0
            # stride = 200
            while start < blobs['H_num']:
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]

                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride
                yield im_orig, b_temp, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32, 'O_all_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([1, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None]),
                                                             'O_all_score': tf.TensorShape([None, 80])},
                                                            tf.TensorShape([]))
                                             )
    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000) # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id