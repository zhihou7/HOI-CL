# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp
from ult.apply_prior import apply_prior

import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg'
    # print(im_file)
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection):

    im_orig, im_shape = get_blob(image_id)
    
    blobs = {}
    blobs['H_num']       = 1
   
    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human
            
            # Predict actrion using human appearance only
            blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
            # prediction_H  = net.test_image_H(sess, im_orig, blobs)

            # save image information
            dic = {}
            dic['image_id']   = image_id
            dic['person_box'] = Human_out[2]

            # Predict actrion using human and object appearance 
            Score_obj     = np.empty((0, 4 + 29), dtype=np.float32) 

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object

                    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                    blobs['sp']      = Get_next_sp(Human_out[2], Object[2]).reshape(1, 64, 64, 2)
                    prediction_HO  = net.test_image_HO(sess, im_orig, blobs)

                    if prior_flag == 1:
                        prediction_HO  = apply_prior(Object, prediction_HO)
                    if prior_flag == 2:
                        prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)
                    if prior_flag == 3:
                        prediction_HO  = apply_prior(Object, prediction_HO)
                        prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)

                    This_Score_obj = np.concatenate((Object[2].reshape(1,4), prediction_HO[0] * np.max(Object[5])), axis=1)
                    Score_obj      = np.concatenate((Score_obj, This_Score_obj), axis=0)

            # print(prediction_HO.shape, blobs['H_boxes'].shape, blobs['O_boxes'].shape)
            # exit()
            # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            if Score_obj.shape[0] == 0:
                continue

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj,0)[4:]

            # agent mAP
            for i in range(29):
                #'''
                # walk, smile, run, stand
                # if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                #     agent_name      = Action_dic_inv[i] + '_agent'
                #     dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                #     continue
                
                # cut
                if i == 2:
                    agent_name = 'cut_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
                    continue 
                if i == 4:
                    continue   

                # eat
                if i == 9:
                    agent_name = 'eat_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
                    continue  
                if i == 16:
                    continue

                # hit
                if i == 19:
                    agent_name = 'hit_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
                    continue  
                if i == 20:
                    continue  

                # These 2 classes need to save manually because there is '_' in action name
                if i == 6:
                    agent_name = 'talk_on_phone_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue 

                # all the rest
                agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                #'''

                '''
                if i == 6:
                    agent_name = 'talk_on_phone_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                    continue 

                agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                '''
                   
            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                # if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                #     dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * prediction_H[0][0][i])
                #     continue

                # Impossible to perform this action
                if np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i] == 0:
                   dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

                # Action with >0 score
                else:
                   dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4], np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

            detection.append(dic)


def test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_dir, object_thres, human_thres, prior_flag):


    np.random.seed(cfg.RNG_SEED)
    detection = []
    count = 0

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}


    for line in open(cfg.DATA_DIR + '/' + '/v-coco/data/splits/vcoco_test.ids', 'r'):

        _t['im_detect'].tic()
 
        image_id   = int(line.rstrip())
        
        im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 4946, _t['im_detect'].average_time))
        count += 1

    pickle.dump( detection, open( output_dir, "wb" ) )



def test_net_data_api_24(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_dir,
                         object_thres, human_thres, prior_flag, blobs, image_id, img_orig):
    detection = []

    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0
    _t['im_detect'].tic()
    _t['misc'].tic()
    while True:
        _t['im_detect'].tic()

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:

            # blobs = {}
            # blobs['H_num'] = 0
            # blobs['H_boxes'] = []
            # blobs['O_boxes'] = []
            # blobs['sp'] = []
            # blobs['H_score'] = []
            # blobs['O_score'] = []
            # blobs['O_mask'] = []
            # blobs['O_cls'] = []
            if 'cls_prob_H' in net.predictions:
                prediction_H, prediction_HO, _blobs, _image_id, _img_orig = sess.run([
                    net.predictions["cls_prob_H"], net.predictions["cls_prob_HO"], blobs, image_id, img_orig])
                # remove 3, 17 22, 23 27
                tmp = np.zeros([prediction_HO.shape[0], 29])
                tmp[:, 0:3] = prediction_HO[:, 0:3]
                tmp[:, 4:17] = prediction_HO[:, 3:16]
                tmp[:, 18:22] = prediction_HO[:, 16:19]
                tmp[:, 24:27] = prediction_HO[:, 20:23]
                tmp[:, 28:29] = prediction_HO[:, 23:24]
                prediction_HO = np.asarray([tmp])

                tmp = np.zeros([prediction_H.shape[0], 29])
                tmp[:, 0:3] = prediction_H[:, 0:3]
                tmp[:, 4:17] = prediction_H[:, 3:16]
                tmp[:, 18:22] = prediction_H[:, 16:19]
                tmp[:, 24:27] = prediction_H[:, 20:23]
                tmp[:, 28:29] = prediction_H[:, 23:24]

                prediction_H = np.asarray([tmp])
                # prediction_H = None
                # print("yes prob H")
                print(prediction_HO.shape, prediction_H.shape)
            else:
                prediction_HO, _blobs, _image_id, _img_orig = sess.run(
                    [net.predictions["cls_prob_HO"], blobs, image_id, img_orig])
                prediction_H = None
                tmp = np.zeros([prediction_HO.shape[0], 29])
                tmp[:, 0:3] = prediction_HO[:, 0:3]
                tmp[:, 4:17] = prediction_HO[:, 3:16]
                tmp[:, 18:22] = prediction_HO[:, 16:20]
                tmp[:, 24:27] = prediction_HO[:, 20:23]
                tmp[:, 28:29] = prediction_HO[:, 23:24]
                prediction_HO = np.asarray([tmp])

        except InvalidArgumentError as e:
            # cls_prob_HO = np.zeros(shape=[blobs['sp'].shape[0], self.num_classes])
            if net.model_name.__contains__('lamb'):
                print('InvalidArgumentError', sess.run([_image_id]))
                continue
            else:
                raise e
        except tf.errors.OutOfRangeError:
            print('END')
            break

        start = 0
        # print(_blobs['H_num'], prediction_HO[0].shape)
        for j in range(_blobs['H_num'] + 1):
            if j < _blobs['H_num'] and (_blobs['H_boxes'][j] == _blobs['H_boxes'][start]).all():
                continue

            h_score = _blobs['H_score'][start]
            h_boxes = _blobs['H_boxes'][start]

            o_boxes_list = _blobs['O_boxes']
            o_score_list = _blobs['O_score']
            o_mask_list = _blobs['O_mask']
            o_cls_list = _blobs['O_cls']

            # Predict actrion using human and object appearance
            Score_obj = np.empty((0, 4 + 29), dtype=np.float32)
            # save image information
            dic = {}
            dic['image_id'] = _image_id
            dic['person_box'] = h_boxes[1:]
            for i in range(start, j):
                h_score = _blobs['H_score'][i]

                object_score = [0, 0, 0, 0, 0, 0]
                object_score[4] = o_cls_list[i]
                # print(prediction_HO.shape, i, prediction_HO.dtype, _blobs['O_cls'])
                tmp_prediction_HO = prediction_HO[0:, i:i + 1]
                obj_cls = int(_blobs['O_cls'][i])
                if prior_flag == 1:
                    tmp_prediction_HO = apply_prior(object_score, tmp_prediction_HO)
                if prior_flag == 2:
                    tmp_prediction_HO = tmp_prediction_HO * prior_mask[:, obj_cls].reshape(1, 29)
                if prior_flag == 3:
                    tmp_prediction_HO = apply_prior(object_score, tmp_prediction_HO)
                    tmp_prediction_HO = tmp_prediction_HO * prior_mask[:, obj_cls].reshape(1, 29)

                # print(o_boxes_list[i:i+1, 1:].reshape(1, 4).shape, (tmp_prediction_HO* np.max(o_score_list[i])).shape, np.max(o_score_list[i]).shape)
                This_Score_obj = np.concatenate(
                    (o_boxes_list[i:i + 1, 1:].reshape(1, 4), tmp_prediction_HO[0] * np.max(o_score_list[i])),
                    axis=1)
                # print(This_Score_obj.shape, Score_obj.shape)
                Score_obj = np.concatenate((Score_obj, This_Score_obj), axis=0)

            start = j
            # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            if Score_obj.shape[0] == 0:
                print('no obj box', j, start)
                continue

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj, 0)[4:]

            # agent mAP
            for i in range(29):
                # '''
                # walk, smile, run, stand
                if prediction_H is not None:
                    if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                        agent_name = Action_dic_inv[i] + '_agent'
                        dic[agent_name] = np.max(h_score) * prediction_H[0][0][i]
                        continue

                # cut
                if i == 2:
                    agent_name = 'cut_agent'
                    dic[agent_name] = np.max(h_score) * max(Score_obj[max_idx[2]][4 + 2],
                                                            Score_obj[max_idx[4]][4 + 4])
                    continue
                if i == 4:
                    continue

                    # eat
                if i == 9:
                    agent_name = 'eat_agent'
                    dic[agent_name] = np.max(h_score) * max(Score_obj[max_idx[9]][4 + 9],
                                                            Score_obj[max_idx[16]][4 + 16])
                    continue
                if i == 16:
                    continue

                # hit
                if i == 19:
                    agent_name = 'hit_agent'
                    dic[agent_name] = np.max(h_score) * max(Score_obj[max_idx[19]][4 + 19],
                                                            Score_obj[max_idx[20]][4 + 20])
                    continue
                if i == 20:
                    continue

                    # These 2 classes need to save manually because there is '_' in action name
                if i == 6:
                    agent_name = 'talk_on_phone_agent'
                    dic[agent_name] = np.max(h_score) * Score_obj[max_idx[i]][4 + i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'
                    dic[agent_name] = np.max(h_score) * Score_obj[max_idx[i]][4 + i]
                    continue

                    # all the rest
                agent_name = Action_dic_inv[i].split("_")[0] + '_agent'
                dic[agent_name] = np.max(h_score) * Score_obj[max_idx[i]][4 + i]
                # '''

                '''
                if i == 6:
                    agent_name = 'talk_on_phone_agent'  
                    dic[agent_name] = np.max(h_score) * prediction_H[0][0][i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'  
                    dic[agent_name] = np.max(h_score) * prediction_H[0][0][i]
                    continue 

                agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                dic[agent_name] = np.max(h_score) * prediction_H[0][0][i]
                '''

            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                if prediction_H is not None:
                    if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                        dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                           np.max(h_score) * prediction_H[0][0][i])
                        continue

                # Impossible to perform this action
                if np.max(h_score) * Score_obj[max_idx[i]][4 + i] == 0:
                    dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                       np.max(h_score) * Score_obj[max_idx[i]][4 + i])

                # Action with >0 score
                else:
                    dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4],
                                                       np.max(h_score) * Score_obj[max_idx[i]][4 + i])

            detection.append(dic)

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 15765, _image_id,
                                                                       _t['im_detect'].average_time), end='',
              flush=True)
        # if count >= 100:
        #     break

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    _t['misc'].toc()
    print('dump detection', len(detection), _t['misc'].average_time)
    import copy
    # a = copy.deepcopy(detection)
    # print(detection)
    pickle.dump(detection, open(output_dir, "wb"))
    print('dump sucessfully')



def obtain_data(Test_RCNN, prior_mask, Action_dic_inv, output_dir, object_thres, human_thres, prior_flag):
    np.random.seed(cfg.RNG_SEED)

    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        for line in open(cfg.DATA_DIR + '/' + '/v-coco/data/splits/vcoco_test.ids', 'r'):
            # if i > 10:
            #     break
            i += 1

            image_id = int(line.rstrip())

            im_orig, im_shape = get_blob(image_id)
            # from skimage import transform
            # scale = 1
            # if scale != 1:
            #     im_orig = transform.resize(im_orig, [1, im_shape[0] * 2, im_shape[1] * 2, 1], preserve_range=True)

            blobs = {}
            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            blobs['O_mask'] = []
            blobs['O_cls'] = []
            blobs['pose_box'] = []

            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # Predict actrion using human appearance only
                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)
                    # prediction_H = net.test_image_H(sess, im_orig, blobs)

                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['O_boxes'].append(
                                    np.array([0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]))
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])
                            blobs['H_boxes'].append(np.array(
                                    [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            mask = np.zeros(shape=(im_shape[0] // 16, im_shape[1] // 16, 1), dtype=np.float32)
                            obj_box = blobs['O_boxes'][-1][1:].astype(np.int32)
                            # print(obj_box)
                            # print(obj_box, blobs['O_boxes'])
                            # mask[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                            # from skimage import transform
                            # mask = transform.resize(mask, [im_shape[0] // 16, im_shape[1] // 16, 1], order=0,
                            #                         preserve_range=True)
                            blobs['O_mask'].append(mask)
                            blobs['O_cls'].append(Object[4])

                    # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
                if blobs['H_num'] == 0:
                    continue
                yield im_orig, blobs, image_id
                blobs['H_num'] = 0
                blobs['H_boxes'] = []
                blobs['O_boxes'] = []
                blobs['sp'] = []
                blobs['H_score'] = []
                blobs['O_score'] = []
                blobs['O_mask'] = []
                blobs['O_cls'] = []

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32,
        {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32, 'O_mask': tf.float32,
         'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32, }, tf.int32,),
                                             output_shapes=(tf.TensorShape([1, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 3]),
                                                             'O_mask': tf.TensorShape([None, None, None, 1]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None]),
                                                             },
                                                            tf.TensorShape([]))
                                             )
    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000)  # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id


def test_net_data_api_21(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_dir,
                         object_thres, human_thres, prior_flag, blobs, image_id, img_orig):
    detection = []

    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0
    _t['im_detect'].tic()
    _t['misc'].tic()
    while True:
        _t['im_detect'].tic()

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:

            # blobs = {}
            # blobs['H_num'] = 0
            # blobs['H_boxes'] = []
            # blobs['O_boxes'] = []
            # blobs['sp'] = []
            # blobs['H_score'] = []
            # blobs['O_score'] = []
            # blobs['O_mask'] = []
            # blobs['O_cls'] = []
            if 'cls_prob_H' in net.predictions:
                prediction_H, prediction_HO, _blobs, _image_id, _img_orig = sess.run([
                    net.predictions["cls_prob_H"], net.predictions["cls_prob_HO"], blobs, image_id, img_orig])
                # remove 3, 17 22, 23 27
                tmp = np.zeros([prediction_HO.shape[0], 29])
                tmp[:, 0:3] = prediction_HO[:, 0:3]
                tmp[:, 4:17] = prediction_HO[:, 3:16]
                tmp[:, 18:22] = prediction_HO[:, 16:19]
                tmp[:, 24:27] = prediction_HO[:, 20:23]
                tmp[:, 28:29] = prediction_HO[:, 23:24]
                prediction_HO = np.asarray([tmp])

                tmp = np.zeros([prediction_H.shape[0], 29])
                tmp[:, 0:3] = prediction_H[:, 0:3]
                tmp[:, 4:17] = prediction_H[:, 3:16]
                tmp[:, 18:22] = prediction_H[:, 16:19]
                tmp[:, 24:27] = prediction_H[:, 20:23]
                tmp[:, 28:29] = prediction_H[:, 23:24]

                prediction_H = np.asarray([tmp])
                # prediction_H = None
                # print("yes prob H")
                print(prediction_HO.shape, prediction_H.shape)
            else:
                prediction_HO, _blobs, _image_id, _img_orig = sess.run(
                    [net.predictions["cls_prob_HO"], blobs, image_id, img_orig])
                prediction_H = None
                # tmp = np.zeros([prediction_HO.shape[0], 29])
                # tmp[:, 0:3] = prediction_HO[:, 0:3]
                # tmp[:, 4:17] = prediction_HO[:, 3:16]
                # tmp[:, 18:22] = prediction_HO[:, 16:20]
                # tmp[:, 24:27] = prediction_HO[:, 20:23]
                # tmp[:, 28:29] = prediction_HO[:, 23:24]
                prediction_HO = np.asarray([prediction_HO])


        except InvalidArgumentError as e:
            # cls_prob_HO = np.zeros(shape=[blobs['sp'].shape[0], self.num_classes])
            if net.model_name.__contains__('lamb'):
                print('InvalidArgumentError', sess.run([_image_id]))
                continue
            else:
                raise e
        except tf.errors.OutOfRangeError:
            print('END')
            break

        start = 0
        # print(_blobs['H_num'], prediction_HO[0].shape)
        for j in range(_blobs['H_num'] + 1):
            if j < _blobs['H_num'] and (_blobs['H_boxes'][j] == _blobs['H_boxes'][start]).all():
                continue

            h_score = _blobs['H_score'][start]
            h_boxes = _blobs['H_boxes'][start]

            o_boxes_list = _blobs['O_boxes']
            o_score_list = _blobs['O_score']
            o_cls_list = _blobs['O_cls']

            # Predict actrion using human and object appearance
            Score_obj = np.empty((0, 4 + 222), dtype=np.float32)
            # save image information
            dic = {}
            dic['image_id'] = _image_id
            dic['person_box'] = h_boxes[1:]
            for i in range(start, j):
                h_score = _blobs['H_score'][i]

                object_score = [0, 0, 0, 0, 0, 0]
                object_score[4] = o_cls_list[i]
                # print(prediction_HO.shape, i, prediction_HO.dtype, _blobs['O_cls'])
                tmp_prediction_HO = prediction_HO[0:, i:i + 1]
                obj_cls = int(_blobs['O_cls'][i])
                # print(o_boxes_list[i:i+1, 1:].reshape(1, 4).shape, (tmp_prediction_HO* np.max(o_score_list[i])).shape, np.max(o_score_list[i]).shape)
                This_Score_obj = np.concatenate(
                    (o_boxes_list[i:i + 1, 1:].reshape(1, 4), tmp_prediction_HO[0] * np.max(o_score_list[i])),
                    axis=1)
                # print(This_Score_obj.shape, Score_obj.shape)
                Score_obj = np.concatenate((Score_obj, This_Score_obj), axis=0)

            start = j
            # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            if Score_obj.shape[0] == 0:
                print('no obj box', j, start)
                continue

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj, 0)[4:]


            # role mAP
            for i in range(222):
                dic[i] = np.append(Score_obj[max_idx[i]][:4], np.max(h_score) * Score_obj[max_idx[i]][4 + i])
            # import ipdb
            # ipdb.set_trace()
            detection.append(dic)

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 15765, _image_id,
                                                                       _t['im_detect'].average_time), end='',
              flush=True)
        # if count >= 100:
        #     break

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    _t['misc'].toc()
    print('dump detection', len(detection), _t['misc'].average_time)
    import copy
    # a = copy.deepcopy(detection)
    # print(detection)
    pickle.dump(detection, open(output_dir, "wb"))
    print('dump sucessfully')


