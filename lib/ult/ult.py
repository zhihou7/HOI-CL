# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# --------------------------------------------------------

"""
Generating training instance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
import json
import pickle
import random
from random import randint
import tensorflow as tf
import cv2
from .config import cfg

# for merge COCO and HICO dataset
MAX_COCO_ID = 650000
MAX_HICO_ID = 40000


def bbox_trans(human_box_ori, object_box_ori, ratio, size=64):
    human_box = human_box_ori.copy()
    object_box = object_box_ori.copy()

    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)

    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally  
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (InteractionPattern[2] + 1) / 2
        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically 
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def Get_next_sp(human_box, object_box):
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64, 64, 2))
    Pattern[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 1
    Pattern[int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1, 1] = 1

    return Pattern


#
# def Get_next_sp_with_pose(human_box, object_box, human_pose, num_joints=17):
#     InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
#                           max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
#     height = InteractionPattern[3] - InteractionPattern[1] + 1
#     width = InteractionPattern[2] - InteractionPattern[0] + 1
#     if height > width:
#         H, O = bbox_trans(human_box, object_box, 'height')
#     else:
#         H, O = bbox_trans(human_box, object_box, 'width')
#
#     Pattern = np.zeros((64, 64, 2), dtype='float32')
#     Pattern[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 1
#     Pattern[int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1, 1] = 1
#
#     if human_pose != None and len(human_pose) == 51:
#         skeleton = get_skeleton(human_box, human_pose, H, num_joints)
#     else:
#         skeleton = np.zeros((64, 64, 1), dtype='float32')
#         skeleton[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 0.05
#
#     Pattern = np.concatenate((Pattern, skeleton), axis=2)
#
#     return Pattern


def get_skeleton(human_box, human_pose, human_pattern, num_joints=17, size=64):
    width = human_box[2] - human_box[0] + 1
    height = human_box[3] - human_box[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i: 3 * (i + 1)]
        x_ratio = (joint_x - human_box[0]) / float(width)
        y_ratio = (joint_y - human_box[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2

    return draw_relation(human_pattern, joints)


def draw_relation(human_pattern, joints, size=64):
    joint_relation = [[1, 3], [2, 4], [0, 1], [0, 2], [0, 17], [5, 17], [6, 17], [5, 7], [6, 8], [7, 9], [8, 10],
                      [11, 17], [12, 17], [11, 13], [12, 14], [13, 15], [14, 16]]
    color = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    skeleton = np.zeros((size, size, 1), dtype="float32")

    for i in range(len(joint_relation)):
        cv2.line(skeleton, tuple(joints[joint_relation[i][0]]), tuple(joints[joint_relation[i][1]]), (color[i]))

    # cv2.rectangle(skeleton, (int(human_pattern[0]), int(human_pattern[1])), (int(human_pattern[2]), int(human_pattern[3])), (255))
    # cv2.imshow("Joints", skeleton)
    # cv2.waitKey(0)
    # print(skeleton[:,:,0])

    return skeleton


def bb_IOU(boxA, boxB):
    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def Augmented_box(bbox, shape, image_id, augment=15):
    thres_ = 0.7

    box = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(1, 5)
    box = box.astype(np.float64)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return box
    count = 0
    time_count = 0
    while count < augment:

        time_count += 1
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10, 10) * 0.01

        height_shift = randint(-np.floor(height), np.floor(height)) * 0.1
        width_shift = randint(-np.floor(width), np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)

        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1, 5)
            box = np.concatenate((box, box_), axis=0)
            count += 1
        if time_count > 150:
            return box

    return box


def Generate_action(action_list, nums=29):
    action_ = np.zeros(nums)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1, nums)
    return action_


def Get_Next_Instance_HO_Neg(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):
    GT = trainval_GT[iter % Data_length]
    image_id = GT[0]
    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    import os
    if not os.path.exists(im_file):
        print("not existing:", im_file)
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H = Augmented_HO_Neg(
        GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)

    blobs = {}
    blobs['image'] = im_orig
    blobs['H_boxes_solo'] = Human_augmented_solo
    blobs['H_boxes'] = Human_augmented
    blobs['O_boxes'] = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H'] = action_H
    blobs['Mask_HO'] = mask_HO
    blobs['Mask_H'] = mask_H
    blobs['sp'] = Pattern
    blobs['H_num'] = len(action_H)

    return blobs


def Augmented_HO_Neg(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human = GT[2]
    Object = GT[3]

    action_HO_ = Generate_action(GT[1])
    action_H_ = Generate_action(GT[4])
    mask_HO_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29)
    mask_H_ = np.asarray(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 29)

    Human_augmented = Augmented_box(Human, shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()

    Human_augmented = Human_augmented[:min(len(Human_augmented), len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented), len(Object_augmented))]

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_HO = action_HO_
    action_H = action_H_
    mask_HO = mask_HO_
    mask_H = mask_H_
    Pattern = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H = np.concatenate((action_H, action_H_), axis=0)
        mask_H = np.concatenate((mask_H, mask_H_), axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO = np.concatenate((mask_HO, mask_HO_), axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1, 29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)

    Pattern = Pattern.reshape(num_pos_neg, 64, 64, 2)
    Human_augmented = Human_augmented.reshape(num_pos_neg, 5)
    Human_augmented_solo = Human_augmented_solo.reshape(num_pos, 5)
    Object_augmented = Object_augmented.reshape(num_pos_neg, 5)
    action_HO = action_HO.reshape(num_pos_neg, 29)
    action_H = action_H.reshape(num_pos, 29)
    mask_HO = mask_HO.reshape(num_pos_neg, 29)
    mask_H = mask_H.reshape(num_pos, 29)

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H


def Augmented_HO_spNeg(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human = GT[2]
    Object = GT[3]
    set_list = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (4, 1), (4, 19),
                (4, 28), (4, 46), (4, 47), (4, 48), (4, 49), (4, 51), (4, 52), (4, 54),
                (4, 55), (4, 56), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8), (5, 9),
                (5, 18), (5, 21), (6, 68), (7, 33), (8, 64), (9, 47), (9, 48), (9, 49),
                (9, 50), (9, 51), (9, 52), (9, 53), (9, 54), (9, 55), (9, 56), (10, 2),
                (10, 4), (10, 14), (10, 18), (10, 21), (10, 25), (10, 27), (10, 29),
                (10, 57), (10, 58), (10, 60), (10, 61), (10, 62), (10, 64), (11, 31),
                (11, 32), (11, 37), (11, 38), (12, 14), (12, 57), (12, 58), (12, 60),
                (12, 61), (13, 40), (13, 41), (13, 42), (13, 46), (14, 1), (14, 25),
                (14, 26), (14, 27), (14, 29), (14, 30), (14, 31), (14, 32), (14, 33),
                (14, 34), (14, 35), (14, 37), (14, 38), (14, 39), (14, 40), (14, 41),
                (14, 42), (14, 47), (14, 50), (14, 68), (14, 74), (14, 75), (14, 78),
                (15, 30), (15, 33), (16, 43), (16, 44), (16, 45), (18, 1), (18, 2),
                (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 11),
                (18, 14), (18, 15), (18, 16), (18, 17), (18, 18), (18, 19), (18, 20),
                (18, 21), (18, 24), (18, 25), (18, 26), (18, 27), (18, 28), (18, 29),
                (18, 30), (18, 31), (18, 32), (18, 33), (18, 34), (18, 35), (18, 36),
                (18, 37), (18, 38), (18, 39), (18, 40), (18, 41), (18, 42), (18, 43),
                (18, 44), (18, 45), (18, 46), (18, 47), (18, 48), (18, 49), (18, 51),
                (18, 53), (18, 54), (18, 55), (18, 56), (18, 57), (18, 61), (18, 62),
                (18, 63), (18, 64), (18, 65), (18, 66), (18, 67), (18, 68), (18, 73),
                (18, 74), (18, 75), (18, 77), (19, 35), (19, 39), (20, 33), (21, 31),
                (21, 32), (23, 1), (23, 11), (23, 19), (23, 20), (23, 24), (23, 28),
                (23, 34), (23, 49), (23, 53), (23, 56), (23, 61), (23, 63), (23, 64),
                (23, 67), (23, 68), (23, 73), (24, 74), (25, 1), (25, 2), (25, 4),
                (25, 8), (25, 9), (25, 14), (25, 15), (25, 16), (25, 17), (25, 18),
                (25, 19), (25, 21), (25, 25), (25, 26), (25, 27), (25, 28), (25, 29),
                (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36),
                (25, 37), (25, 38), (25, 39), (25, 40), (25, 41), (25, 42), (25, 43),
                (25, 44), (25, 45), (25, 46), (25, 47), (25, 48), (25, 49), (25, 50),
                (25, 51), (25, 52), (25, 53), (25, 54), (25, 55), (25, 56), (25, 57),
                (25, 64), (25, 65), (25, 66), (25, 67), (25, 68), (25, 73), (25, 74),
                (25, 77), (25, 78), (25, 79), (25, 80), (26, 32), (26, 37), (28, 30),
                (28, 33)]

    action_sp_ = Generate_action(GT[1])
    action_HO_ = Generate_action(GT[1])
    obj_cls = GT[-1]
    action_compose = [set_list.index(item) for item in [(ho, obj_cls[0]) for ho in GT[1]]]
    action_compose_ = Generate_action(action_compose, nums=len(set_list))
    action_H_ = Generate_action(GT[4])
    mask_sp_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29)
    mask_HO_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29)
    mask_H_ = np.asarray(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 29)

    Human_augmented = Augmented_box(Human, shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    if Human[0] == 0 and Human[1] == 0 and Human[2] == 0:
        while len(Human_augmented) < Pos_augment + 1:
            Human_augmented = np.concatenate(
                [Human_augmented, Human_augmented[-(Pos_augment + 1 - len(Human_augmented)):]], axis=0)

    Human_augmented = Human_augmented[:min(len(Human_augmented), len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented), len(Object_augmented))]

    num_pos = len(Human_augmented)
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H = action_H_
    action_compose = action_compose_
    mask_sp = mask_sp_
    mask_HO = mask_HO_
    mask_H = mask_H_
    Pattern = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H = np.concatenate((action_H, action_H_), axis=0)
        action_compose = np.concatenate((action_compose, action_compose_), axis=0)
        mask_HO = np.concatenate((mask_HO, mask_HO_), axis=0)
        mask_H = np.concatenate((mask_H, mask_H_), axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp = np.concatenate((mask_sp, mask_sp_), axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1, 29)), axis=0)
        action_compose = np.concatenate((action_compose, np.zeros(len(set_list)).reshape(1, len(set_list))), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)

    Pattern = Pattern.reshape(num_pos_neg, 64, 64, 2)
    Human_augmented_sp = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp = action_sp.reshape(num_pos_neg, 29)
    action_HO = action_HO.reshape(num_pos, 29)
    action_H = action_H.reshape(num_pos, 29)
    action_compose = action_compose.reshape(num_pos, len(set_list))
    mask_sp = mask_sp.reshape(num_pos_neg, 29)
    mask_HO = mask_HO.reshape(num_pos, 29)
    mask_H = mask_H.reshape(num_pos, 29)

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, action_compose


def Augmented_HO_spNeg2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human = GT[2]
    Object = GT[3]
    set_list = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (3, 1), (3, 19), (3, 28), (3, 46), (3, 47),
                (3, 48), (3, 49), (3, 51), (3, 52), (3, 54), (3, 55), (3, 56), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7),
                (4, 8), (4, 9), (4, 18), (4, 21), (5, 68), (6, 33), (7, 64), (8, 47), (8, 48), (8, 49), (8, 50),
                (8, 51), (8, 52), (8, 53), (8, 54), (8, 55), (8, 56), (9, 2), (9, 4), (9, 14), (9, 18), (9, 21),
                (9, 25), (9, 27), (9, 29), (9, 57), (9, 58), (9, 60), (9, 61), (9, 62), (9, 64), (10, 31), (10, 32),
                (10, 37), (10, 38), (11, 14), (11, 57), (11, 58), (11, 60), (11, 61), (12, 40), (12, 41), (12, 42),
                (12, 46), (13, 1), (13, 25), (13, 26), (13, 27), (13, 29), (13, 30), (13, 31), (13, 32), (13, 33),
                (13, 34), (13, 35), (13, 37), (13, 38), (13, 39), (13, 40), (13, 41), (13, 42), (13, 47), (13, 50),
                (13, 68), (13, 74), (13, 75), (13, 78), (14, 30), (14, 33), (15, 43), (15, 44), (15, 45), (16, 1),
                (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 11), (16, 14), (16, 15), (16, 16),
                (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 24), (16, 25), (16, 26), (16, 27), (16, 28),
                (16, 29), (16, 30), (16, 31), (16, 32), (16, 33), (16, 34), (16, 35), (16, 36), (16, 37), (16, 38),
                (16, 39), (16, 40), (16, 41), (16, 42), (16, 43), (16, 44), (16, 45), (16, 46), (16, 47), (16, 48),
                (16, 49), (16, 51), (16, 53), (16, 54), (16, 55), (16, 56), (16, 57), (16, 61), (16, 62), (16, 63),
                (16, 64), (16, 65), (16, 66), (16, 67), (16, 68), (16, 73), (16, 74), (16, 75), (16, 77), (17, 35),
                (17, 39), (18, 33), (19, 31), (19, 32), (20, 74), (21, 1), (21, 2), (21, 4), (21, 8), (21, 9), (21, 14),
                (21, 15), (21, 16), (21, 17), (21, 18), (21, 19), (21, 21), (21, 25), (21, 26), (21, 27), (21, 28),
                (21, 29), (21, 30), (21, 31), (21, 32), (21, 33), (21, 34), (21, 35), (21, 36), (21, 37), (21, 38),
                (21, 39), (21, 40), (21, 41), (21, 42), (21, 43), (21, 44), (21, 45), (21, 46), (21, 47), (21, 48),
                (21, 49), (21, 50), (21, 51), (21, 52), (21, 53), (21, 54), (21, 55), (21, 56), (21, 57), (21, 64),
                (21, 65), (21, 66), (21, 67), (21, 68), (21, 73), (21, 74), (21, 77), (21, 78), (21, 79), (21, 80),
                (22, 32), (22, 37), (23, 30), (23, 33)]

    action_sp_ = Generate_action(GT[1], nums=24)
    action_HO_ = Generate_action(GT[1], nums=24)
    obj_cls = GT[-1]
    action_compose = [set_list.index(item) for item in [(ho, obj_cls[0]) for ho in GT[1]]]
    action_compose_ = Generate_action(action_compose, nums=len(set_list))
    action_H_ = Generate_action(GT[4], nums=24)
    mask_sp_ = np.ones([1, 24], np.int32)
    mask_HO_ = np.ones([1, 24], np.int32)
    mask_H_ = np.ones([1, 24], np.int32)

    Human_augmented = Augmented_box(Human, shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented = Human_augmented[:min(len(Human_augmented), len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented), len(Object_augmented))]

    num_pos = len(Human_augmented)
    # pose_list = [GT[5]] * num_pos
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                # pose_list.append(Neg[7])
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                # pose_list.append(Neg[7])
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H = action_H_
    action_compose = action_compose_
    mask_sp = mask_sp_
    mask_HO = mask_HO_
    mask_H = mask_H_
    Pattern = np.empty((0, 64, 64, 2), dtype=np.float32)
    pose_box = []
    # print('pose infor:', GT[5], pose_list)
    # pose_box = obtain_pose_box(Human_augmented, pose_list, shape)
    for item in Human_augmented:
        pose_box.extend([item] * 17)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H = np.concatenate((action_H, action_H_), axis=0)
        action_compose = np.concatenate((action_compose, action_compose_), axis=0)
        mask_HO = np.concatenate((mask_HO, mask_HO_), axis=0)
        mask_H = np.concatenate((mask_H, mask_H_), axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp = np.concatenate((mask_sp, mask_sp_), axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(24).reshape(1, 24)), axis=0)
        action_compose = np.concatenate((action_compose, np.zeros(len(set_list)).reshape(1, len(set_list))), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)

        mask = np.zeros(shape=(1, shape[0] // 16, shape[1] // 16, 1), dtype=np.float32)
        # obj_box = Object_augmented[i][1:].astype(np.int32)
        # print(obj_box)
        # mask[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
        # from skimage import transform
        # mask = transform.resize(mask, [1, shape[0] // 16, shape[1] // 16, 1], order=0, preserve_range=True)

    Pattern = Pattern.reshape(num_pos_neg, 64, 64, 2)
    Human_augmented_sp = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp = action_sp.reshape(num_pos_neg, 24)
    action_HO = action_HO.reshape(num_pos, 24)
    action_H = action_H.reshape(num_pos, 24)
    action_compose = action_compose.reshape(num_pos_neg, len(set_list))
    mask_sp = mask_sp.reshape(num_pos_neg, 24)
    mask_HO = mask_HO.reshape(num_pos, 24)
    mask_H = mask_H.reshape(num_pos, 24)

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, action_compose


def Augmented_HO_spNeg3(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human = GT[2]
    Object = GT[3]
    set_list = [(0, 38), (1, 31), (1, 32), (2, 1), (2, 19), (2, 28), (2, 43), (2, 44), (2, 46), (2, 47), (2, 48),
                (2, 49),
                (2, 51), (2, 52), (2, 54), (2, 55), (2, 56), (2, 77), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8),
                (3, 9), (3, 18), (3, 21), (4, 68), (5, 33), (6, 64), (7, 43), (7, 44), (7, 45), (7, 47), (7, 48),
                (7, 49),
                (7, 50), (7, 51), (7, 52), (7, 53), (7, 54), (7, 55), (7, 56), (8, 2), (8, 4), (8, 14), (8, 18),
                (8, 21),
                (8, 25), (8, 27), (8, 29), (8, 57), (8, 58), (8, 60), (8, 61), (8, 62), (8, 64), (9, 31), (9, 32),
                (9, 37),
                (9, 38), (10, 14), (10, 57), (10, 58), (10, 60), (10, 61), (11, 40), (11, 41), (11, 42), (11, 46),
                (12, 1),
                (12, 25), (12, 26), (12, 27), (12, 29), (12, 30), (12, 31), (12, 32), (12, 33), (12, 34), (12, 35),
                (12, 37), (12, 38), (12, 39), (12, 40), (12, 41), (12, 42), (12, 47), (12, 50), (12, 68), (12, 74),
                (12, 75), (12, 78), (13, 30), (13, 33), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7),
                (14, 8), (14, 11), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21),
                (14, 24),
                (14, 25), (14, 26), (14, 27), (14, 28), (14, 29), (14, 30), (14, 31), (14, 32), (14, 33), (14, 34),
                (14, 35), (14, 36), (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 43), (14, 44),
                (14, 45), (14, 46), (14, 47), (14, 48), (14, 49), (14, 51), (14, 53), (14, 54), (14, 55), (14, 56),
                (14, 57), (14, 61), (14, 62), (14, 63), (14, 64), (14, 65), (14, 66), (14, 67), (14, 68), (14, 73),
                (14, 74), (14, 75), (14, 77), (15, 33), (15, 35), (15, 39), (16, 31), (16, 32), (17, 74), (18, 1),
                (18, 2),
                (18, 4), (18, 8), (18, 9), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18), (18, 19), (18, 21),
                (18, 25),
                (18, 26), (18, 27), (18, 28), (18, 29), (18, 30), (18, 31), (18, 32), (18, 33), (18, 34), (18, 35),
                (18, 36), (18, 37), (18, 38), (18, 39), (18, 40), (18, 41), (18, 42), (18, 43), (18, 44), (18, 45),
                (18, 46), (18, 47), (18, 48), (18, 49), (18, 50), (18, 51), (18, 52), (18, 53), (18, 54), (18, 55),
                (18, 56), (18, 57), (18, 64), (18, 65), (18, 66), (18, 67), (18, 68), (18, 73), (18, 74), (18, 77),
                (18, 78), (18, 79), (18, 80), (19, 32), (19, 37), (20, 30), (20, 33)]
    action_sp_ = Generate_action(GT[1], nums=21)
    action_HO_ = Generate_action(GT[1], nums=21)
    obj_cls = GT[-1]
    action_compose = [set_list.index(item) for item in [(ho, obj_cls[0]) for ho in GT[1]]]
    action_compose_ = Generate_action(action_compose, nums=len(set_list))
    action_H_ = Generate_action(GT[4], nums=21)
    mask_sp_ = np.ones([1, 21], np.int32)
    mask_HO_ = np.ones([1, 21], np.int32)
    mask_H_ = np.ones([1, 21], np.int32)

    Human_augmented = Augmented_box(Human, shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented = Human_augmented[:min(len(Human_augmented), len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented), len(Object_augmented))]

    num_pos = len(Human_augmented)
    # pose_list = [GT[5]] * num_pos
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                # pose_list.append(Neg[7])
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                # pose_list.append(Neg[7])
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H = action_H_
    action_compose = action_compose_
    mask_sp = mask_sp_
    mask_HO = mask_HO_
    mask_H = mask_H_
    Pattern = np.empty((0, 64, 64, 2), dtype=np.float32)
    pose_box = []
    # print('pose infor:', GT[5], pose_list)
    # pose_box = obtain_pose_box(Human_augmented, pose_list, shape)
    for item in Human_augmented:
        pose_box.extend([item] * 17)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H = np.concatenate((action_H, action_H_), axis=0)
        action_compose = np.concatenate((action_compose, action_compose_), axis=0)
        mask_HO = np.concatenate((mask_HO, mask_HO_), axis=0)
        mask_H = np.concatenate((mask_H, mask_H_), axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp = np.concatenate((mask_sp, mask_sp_), axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(21).reshape(1, 21)), axis=0)
        action_compose = np.concatenate((action_compose, np.zeros(len(set_list)).reshape(1, len(set_list))), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape([1, 64, 64, 2])
        # Pattern_ = np.concatenate([Pattern_, np.zeros([1, 64, 64, 1])], axis=-1)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)

        mask = np.zeros(shape=(1, shape[0] // 16, shape[1] // 16, 1), dtype=np.float32)

    Pattern = Pattern.reshape(num_pos_neg, 64, 64, 2)
    Human_augmented_sp = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp = action_sp.reshape(num_pos_neg, 21)
    action_HO = action_HO.reshape(num_pos, 21)
    action_H = action_H.reshape(num_pos, 21)
    action_compose = action_compose.reshape(num_pos_neg, len(set_list))
    mask_sp = mask_sp.reshape(num_pos_neg, 21)
    mask_HO = mask_HO.reshape(num_pos, 21)
    mask_H = mask_H.reshape(num_pos, 21)

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, action_compose


def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1, 600)
    return action_


def Get_Next_Instance_HO_Neg_HICO(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):
    GT = trainval_GT[iter % Data_length]
    image_id = GT[0]
    im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(
        8) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos = Augmented_HO_Neg_HICO(GT, Trainval_Neg, im_shape,
                                                                                           Pos_augment, Neg_select)

    blobs = {}
    blobs['image'] = im_orig
    blobs['H_boxes'] = Human_augmented
    blobs['O_boxes'] = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp'] = Pattern
    blobs['H_num'] = num_pos

    return blobs


def Augmented_neg_box(bbox, shape, image_id, augment=15, bbox_list=[]):
    thres_ = 0.25

    # box = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(1, 5)
    # box = box.astype(np.float64)
    box = np.empty([1, 5], np.float64)

    count = 0
    time_count = 0
    while count < augment:

        time_count += 1
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10, 10) * 0.01

        height_shift = randint(-np.floor(height), np.floor(height))
        height_shift = np.sign(height_shift) * 0.5 * height + height_shift
        width_shift = randint(-np.floor(width), np.floor(width)) * 0.1
        width_shift = np.sign(width_shift) * 0.5 * width + width_shift

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)

        valid_neg_box = True
        for bbox1 in bbox_list:
            if bb_IOU(bbox1, np.array([H_0, H_1, H_2, H_3])) > thres_:
                valid_neg_box = False
                break
        if valid_neg_box:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1, 5)
            box = np.concatenate((box, box_), axis=0)
            count += 1
        if time_count > 150:
            return box

    return box


def obtain_data2_large(Pos_augment=15, Neg_select=60, augment_type=0, model_name='',
                       pattern_type=False, zero_shot_type=0, isalign=False, bnum=2, neg_type_ratio=0):
    # bnum = 2
    if pattern_type == 1:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl', "rb"), encoding='latin1')
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose.pkl', "rb"), encoding='latin1')
    else:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb"), encoding='latin1')

    g_func = generator2

    def generator3(Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type):
        buffer = [[] for i in range(7)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        # np.random.seed(0)
        for im_orig, image_id, num_pos, Human_augmented, Object_augmented, \
            action_HO, Pattern in g_func(Trainval_GT, Trainval_N, Pos_augment, Neg_select,
                                                              augment_type,
                                                              pattern_type, zero_shot_type, isalign, 0, neg_type_ratio):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(Human_augmented)
            buffer[4].append(Object_augmented)
            buffer[5].append(action_HO)
            buffer[6].append(Pattern)
            buffer[3][-1][:, 0] = len(buffer[3]) - 1
            buffer[4][-1][:, 0] = len(buffer[3]) - 1
            if len(buffer[0]) >= bnum:

                # if len(buffer[3][0]) < len(buffer[3][1]):
                #     # make sure the second batch is less.
                #     for i in range(len(buffer)):
                #         tmp = buffer[i][0]
                #         buffer[i][0] = buffer[i][1]
                #         buffer[i][1] = tmp

                # print("inner:", buffer[0][0].shape, buffer[0][1].shape, buffer[1], buffer[2], buffer[3].shape, buffer[4].shape, buffer[5].shape, buffer[6].shape)
                # print("inner:", buffer[1], buffer[2][0], buffer[2][1], buffer[3][0].shape, buffer[3][1].shape, buffer[5][0].shape, buffer[5][1].shape)
                # yield buffer[0][0], buffer[0][1], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6]

                # print("inner hint:", buffer[1], 'num_pos:', buffer[2], 'len of h boxes:',len(buffer[3][0]), len(buffer[3][1]),
                #       len(buffer[4][0]), len(buffer[4][1]), len(buffer[5][0]), len(buffer[5][1]), len(buffer[6][0]), len(buffer[6][1]))

                pos_semi_list = []
                if model_name.__contains__('x5new'):
                    for b in range(bnum):
                        pos_semi_list.append(int(buffer[2][b] + (len(buffer[3][b]) - buffer[2][b]) // 8))
                else:
                    for b in range(bnum):
                        pos_semi_list.append(buffer[2][b])

                for ii in range(3, 7):
                    pos_h_boxes = np.concatenate([buffer[ii][pi][:pos2] for pi, pos2 in enumerate(pos_semi_list)],
                                                 axis=0)
                    neg_h_boxes = np.concatenate([buffer[ii][pi][pos2:] for pi, pos2 in enumerate(pos_semi_list)],
                                                 axis=0)
                    buffer[ii] = np.concatenate([pos_h_boxes, neg_h_boxes], axis=0)

                width = max([buffer[0][b].shape[1] for b in range(bnum)])
                height = max([buffer[0][b].shape[2] for b in range(bnum)])

                im_list = []
                for b in range(bnum):
                    im_list.append(np.pad(buffer[0][b], [(0, 0), (0, max(0, width - buffer[0][b].shape[1])),
                                                         (0, max(0, height - buffer[0][b].shape[2])), (0, 0)],
                                          mode='constant'))

                width = max([buffer[7][b].shape[1] for b in range(bnum)])
                height = max([buffer[7][b].shape[2] for b in range(bnum)])

                yield np.concatenate(im_list, axis=0), buffer[1], sum(pos_semi_list), \
                      buffer[3], buffer[4], buffer[5], buffer[6], pos_semi_list[0]

                buffer = [[] for i in range(8)]
                # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                # count_time += 1
                # print('generate batch:', time.time() - st, "average;",  avg_time)
                # st = time.time()

    if pattern_type == 1:
        pattern_channel = 3
    else:
        pattern_channel = 2
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type),
        output_types=(
            tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        output_shapes=(
            tf.TensorShape([bnum, None, None, 3]),
            tf.TensorShape([bnum, ]),
            tf.TensorShape([]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 600]),
            tf.TensorShape([None, 64, 64, pattern_channel]),
            tf.TensorShape([])
            # tf.TensorShape([2, None, None, None, 1])
        )
        )
    # dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32),
    #                                          output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([])))
    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx = iterator.get_next()
    return image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx


def Augmented_HO_Neg_HICO(GT, Trainval_Neg, shape, Pos_augment, Neg_select, pattern_type=False, isalign=False,
                          box_list=[],
                          real_neg_ratio=0):
    """

    :param GT:
    :param Trainval_Neg:
    :param shape:
    :param Pos_augment:
    :param Neg_select:
    :param pattern_type:
    :param isalign:
    :param box_list:
    :param real_neg_ratio: This is for no action HOI (all zeros)
    :return:
    """
    image_id = GT[0]
    Human = GT[2]
    Object = GT[3]

    action_HO_ = Generate_action_HICO(GT[1])
    action_HO = action_HO_

    Human_augmented = Augmented_box(Human, shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    max_augmented_nums = max(len(Human_augmented), len(Object_augmented))
    if isalign:
        while len(Human_augmented) < max_augmented_nums:
            Human_augmented = np.concatenate(
                [Human_augmented, Human_augmented[-(max_augmented_nums - len(Human_augmented)):]], axis=0)

    if isalign:
        while len(Object_augmented) < max_augmented_nums:
            Object_augmented = np.concatenate(
                [Object_augmented, Object_augmented[-(max_augmented_nums - len(Object_augmented)):]], axis=0)
    # print("shape:", Human_augmented.shape, Object_augmented.shape)
    Human_augmented = Human_augmented[:min(len(Human_augmented), len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented), len(Object_augmented))]

    action_HO = np.tile(action_HO, [len(Human_augmented), 1])

    if len(box_list) > 0 and real_neg_ratio > 0:
        aug_neg_objs = Augmented_neg_box(Object, shape, image_id, int(Pos_augment * real_neg_ratio), bbox_list=box_list)
        if len(aug_neg_objs) > 0:
            aug_neg_humans = np.tile([Human_augmented[0]], [len(aug_neg_objs), 1])
            aug_neg_actions = np.zeros([len(aug_neg_objs), 600], )
            # print(aug_neg_objs.shape, Object_augmented.shape, Human_augmented.shape, aug_neg_humans.shape)
            Human_augmented = np.concatenate([Human_augmented, aug_neg_humans])
            Object_augmented = np.concatenate([Object_augmented, aug_neg_objs])
            action_HO = np.concatenate([action_HO, aug_neg_actions])

    num_pos = len(Human_augmented)
    pose_list = []

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)
    pattern_channel = 2
    Pattern = np.empty((0, 64, 64, pattern_channel), dtype=np.float32)

    for i in range(num_pos_neg):
        # Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        # there are poses for the negative sample
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:])
        Pattern_ = Pattern_.reshape(1, 64, 64, pattern_channel)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)

    Pattern = Pattern.reshape(num_pos_neg, 64, 64, pattern_channel)
    Human_augmented = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented.reshape(num_pos_neg, 5)
    action_HO = action_HO.reshape(num_pos_neg, 600)

    # print("shape1:", Human_augmented.shape, Object_augmented.shape, num_pos, Neg_select)
    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos


def obtain_data2(Pos_augment=15, Neg_select=60, augment_type=0, model_name='', pattern_type=False,
                 zero_shot_type=0, isalign=False, neg_type_ratio=0):
    b_num = 2
    Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')
    Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb"), encoding='latin1')

    g_func = generator2

    def generator3(Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type):
        buffer = [[] for i in range(7)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        for im_orig, image_id, num_pos, Human_augmented, \
            Object_augmented, action_HO, Pattern in g_func(Trainval_GT, Trainval_N, Pos_augment,
                                                                                Neg_select,
                                                                                augment_type, pattern_type,
                                                                                zero_shot_type, isalign,
                                                                                0):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(Human_augmented)
            buffer[4].append(Object_augmented)
            buffer[5].append(action_HO)
            buffer[6].append(Pattern)
            # buffer[8].append(pose_list)
            # print(im_orig.shape, image_id, num_pos,
            if len(buffer[0]) >= b_num:

                # print("inner:", buffer[0][0].shape, buffer[0][1].shape, buffer[1], buffer[2], buffer[3].shape, buffer[4].shape, buffer[5].shape, buffer[6].shape)
                # print("inner:", buffer[1], buffer[2][0], buffer[2][1], buffer[3][0].shape, buffer[3][1].shape, buffer[5][0].shape, buffer[5][1].shape)
                # yield buffer[0][0], buffer[0][1], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6]
                if len(buffer[3][0]) < len(buffer[3][1]):
                    # make sure the second batch is less.
                    for i in range(len(buffer)):
                        tmp = buffer[i][0]
                        buffer[i][0] = buffer[i][1]
                        buffer[i][1] = tmp

                buffer[3][1][:, 0] = 1
                buffer[4][1][:, 0] = 1
                # print("inner hint:", buffer[1], 'num_pos:', buffer[2], 'len of h boxes:',len(buffer[3][0]), len(buffer[3][1]),
                #       len(buffer[4][0]), len(buffer[4][1]), len(buffer[5][0]), len(buffer[5][1]), len(buffer[6][0]), len(buffer[6][1]))

                if model_name.__contains__('x5new'):
                    pos1 = int(buffer[2][0] + (len(buffer[3][0]) - buffer[2][0]) // 8)
                    pos2 = int(buffer[2][1] + (len(buffer[3][1]) - buffer[2][1]) // 8)
                else:
                    pos1 = buffer[2][0]
                    pos2 = buffer[2][1]

                for ii in list(range(3, 7)):
                    pos_h_boxes = np.concatenate([buffer[ii][0][:pos1], buffer[ii][1][:pos2]], axis=0)
                    neg_h_boxes = np.concatenate([buffer[ii][0][pos1:], buffer[ii][1][pos2:]], axis=0)

                    buffer[ii] = np.concatenate([pos_h_boxes, neg_h_boxes], axis=0)
                    # buffer[ii] = np.concatenate([buffer[ii][0], buffer[ii][1]], axis=0)

                buffer = buffer[:-1] + buffer[-1:]

                im_shape1 = buffer[0][0].shape
                im_shape2 = buffer[0][1].shape
                width = max(im_shape1[1], im_shape2[1])
                height = max(im_shape1[2], im_shape2[2])
                im1 = np.pad(buffer[0][0],
                             [(0, 0), (0, max(0, width - im_shape1[1])), (0, max(0, height - im_shape1[2])), (0, 0)],
                             mode='constant')
                im2 = np.pad(buffer[0][1],
                             [(0, 0), (0, max(0, width - im_shape2[1])), (0, max(0, height - im_shape2[2])), (0, 0)],
                             mode='constant')


                split_idx = pos1
                yield np.concatenate([im1, im2], axis=0), buffer[1], pos1 + pos2, buffer[3], buffer[4], buffer[5], \
                      buffer[6], split_idx

                buffer = [[] for i in range(7   )]
                # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                # count_time += 1
                # print('generate batch:', time.time() - st, "average;",  avg_time)
                # st = time.time()

    if pattern_type == 1:
        pattern_channel = 3
    else:
        pattern_channel = 2
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type),
        output_types=(
            tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        output_shapes=(
            tf.TensorShape([2, None, None, 3]),
            tf.TensorShape([2, ]),
            tf.TensorShape([]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 600]),
            tf.TensorShape([None, 64, 64, pattern_channel]),
            tf.TensorShape([])
            # tf.TensorShape([2, None, None, None, 1])
        )
        )
    # dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32),
    #                                          output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([])))
    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx = iterator.get_next()
    return image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx

def get_new_Trainval_GT(Trainval_GT, is_zero_shot, unseen_idx):
    unseen_idx = set(unseen_idx)
    if is_zero_shot > 0:
        new_Trainval_GT = []
        for item in Trainval_GT:
            if len(set(list(item[1])).intersection(unseen_idx)) == 0:
                new_Trainval_GT.append(item)
        Trainval_GT = new_Trainval_GT
    return Trainval_GT


def extract_semi_data(semi_type, model_name):
    print(semi_type, '===========')
    semi_pkl_path = cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl'
    if semi_type == 'default':
        semi_pkl_path = cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl'
    elif semi_type == 'coco':
        semi_pkl_path = cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_semi.pkl'
    elif semi_type == 'coco2':
        semi_pkl_path = cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_semi_coco2.pkl'
    elif semi_type == 'coco1':  # train2017
        semi_pkl_path = cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_semi1.pkl'
    elif semi_type == 'rehico':
        semi_pkl_path = cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl'
    elif semi_type == 'vcoco':
        semi_pkl_path = cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_vcoco_semi.pkl'
    if semi_type == 'both':
        Trainval_semi = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_semi.pkl', "rb"), encoding='latin1')
        Trainval_semi1 = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')
        # Trainval_semi = Trainval_semi[:5000]
        for item in Trainval_semi:
            item[0] += MAX_HICO_ID

        Trainval_semi.extend(Trainval_semi1)
    elif semi_type == 'both1':
        Trainval_semi = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_vcoco_semi.pkl', "rb"),
                                    encoding='latin1')
        Trainval_semi1 = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')

        for item in Trainval_semi:
            item[0] += MAX_HICO_ID
        Trainval_semi.extend(Trainval_semi1)
        pass

    elif semi_type == 'bothzs':
        Trainval_semi = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_semi.pkl', "rb"), encoding='latin1')
        Trainval_semi1 = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')
        # ids1 = [item[0] for item in Trainval_semi]
        # ids2 = [item[0] for item in Trainval_semi1]
        # ids = set(ids1).intersection(set(ids2))
        # Trainval_semi = [item for item in Trainval_semi if item[0] not in ids]
        zero_shot_type = get_zero_shot_type(model_name)
        unseen_idx = get_unseen_index(zero_shot_type)
        print(unseen_idx)
        new_semi = []
        print(len(Trainval_semi))  # 604907

        for item in Trainval_semi:
            item[0] += MAX_HICO_ID
            # print(item)
            if len(item[1]) > 0 and len(list(set(item[1]).intersection(set(unseen_idx)))) > 0:
                new_semi.append(item)
        print(len(new_semi), 'bothzs semi')  # 524239 bothzs semi zs3   517008 bothzs semi zs4

        print(type(Trainval_semi))
        Trainval_semi = new_semi
        Trainval_semi.extend(Trainval_semi1)
    elif semi_type == 'cocozs':
        Trainval_semi = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_semi1.pkl', "rb"), encoding='latin1')
        # ids1 = [item[0] for item in Trainval_semi]
        # ids2 = [item[0] for item in Trainval_semi1]
        # ids = set(ids1).intersection(set(ids2))
        # Trainval_semi = [item for item in Trainval_semi if item[0] not in ids]
        zero_shot_type = get_zero_shot_type(model_name)
        unseen_idx = get_unseen_index(zero_shot_type)

        # Trainval_semi1 = [item for item in Trainval_semi1 if len(list(set(item[1]).intersection(set(unseen_idx)))) == 0] # remove unseen objects.

        print(unseen_idx)
        new_semi = []
        for item in Trainval_semi:
            item[0] += MAX_HICO_ID
            # print(item)
            if len(item[1]) > 0 and len(list(set(item[1]).intersection(set(unseen_idx)))) > 0:
                new_semi.append(item)

        print(type(Trainval_semi))
        Trainval_semi = new_semi
    elif semi_type == 'coco3':
        Trainval_semi = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_semi1.pkl', "rb"), encoding='latin1')
        Trainval_semi1 = pickle.load(
            open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_obj365_coco_semi_obj365_coco.pkl', "rb"), encoding='latin1')

        for item in Trainval_semi:
            item[0] += MAX_HICO_ID
        for item in Trainval_semi1:
            item[0] += MAX_COCO_ID

        Trainval_semi.extend(Trainval_semi1)
    else:
        with open(semi_pkl_path, "rb") as f:
            Trainval_semi = pickle.load(f, encoding='latin1')

        if semi_type == 'coco' or semi_type == 'coco2' or semi_type == 'coco1' or semi_type == 'vcoco':
            for item in Trainval_semi:
                item[0] += MAX_HICO_ID
        if semi_type == 'rehico' and model_name.__contains__('_zs11'):
            zero_shot_type = get_zero_shot_type(model_name)
            unseen_idx = get_unseen_index(zero_shot_type)
            Trainval_semi = get_new_Trainval_GT(Trainval_semi, zero_shot_type, unseen_idx)
            # Trainval_semi = [item for item in Trainval_semi if
            #                   len(list(set(item[1]).intersection(set(unseen_idx)))) == 0]  # remove unseen objects.

            pass
    return Trainval_semi


def obtain_data2_large(Pos_augment=15, Neg_select=60, augment_type=0, model_name='',
                       pattern_type=False, zero_shot_type=0, isalign=False, bnum=2, neg_type_ratio=0):
    # bnum = 2
    if pattern_type == 1:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl', "rb"), encoding='latin1')
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose.pkl', "rb"), encoding='latin1')
    else:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb"), encoding='latin1')

    g_func = generator2

    def generator3(Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type):
        buffer = [[] for i in range(8)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        # np.random.seed(0)
        for im_orig, image_id, num_pos, Human_augmented, Object_augmented, \
            action_HO, Pattern in g_func(Trainval_GT, Trainval_N, Pos_augment, Neg_select,
                                                              augment_type,
                                                              pattern_type, zero_shot_type, isalign, 0):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(Human_augmented)
            buffer[4].append(Object_augmented)
            buffer[5].append(action_HO)
            buffer[6].append(Pattern)
            buffer[3][-1][:, 0] = len(buffer[3]) - 1
            buffer[4][-1][:, 0] = len(buffer[3]) - 1
            if len(buffer[0]) >= bnum:

                # if len(buffer[3][0]) < len(buffer[3][1]):
                #     # make sure the second batch is less.
                #     for i in range(len(buffer)):
                #         tmp = buffer[i][0]
                #         buffer[i][0] = buffer[i][1]
                #         buffer[i][1] = tmp

                # print("inner:", buffer[0][0].shape, buffer[0][1].shape, buffer[1], buffer[2], buffer[3].shape, buffer[4].shape, buffer[5].shape, buffer[6].shape)
                # print("inner:", buffer[1], buffer[2][0], buffer[2][1], buffer[3][0].shape, buffer[3][1].shape, buffer[5][0].shape, buffer[5][1].shape)
                # yield buffer[0][0], buffer[0][1], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6]

                # print("inner hint:", buffer[1], 'num_pos:', buffer[2], 'len of h boxes:',len(buffer[3][0]), len(buffer[3][1]),
                #       len(buffer[4][0]), len(buffer[4][1]), len(buffer[5][0]), len(buffer[5][1]), len(buffer[6][0]), len(buffer[6][1]))

                pos_semi_list = []
                if model_name.__contains__('x5new'):
                    for b in range(bnum):
                        pos_semi_list.append(int(buffer[2][b] + (len(buffer[3][b]) - buffer[2][b]) // 8))
                else:
                    for b in range(bnum):
                        pos_semi_list.append(buffer[2][b])

                for ii in range(3, 7):
                    pos_h_boxes = np.concatenate([buffer[ii][pi][:pos2] for pi, pos2 in enumerate(pos_semi_list)],
                                                 axis=0)
                    neg_h_boxes = np.concatenate([buffer[ii][pi][pos2:] for pi, pos2 in enumerate(pos_semi_list)],
                                                 axis=0)
                    buffer[ii] = np.concatenate([pos_h_boxes, neg_h_boxes], axis=0)

                width = max([buffer[0][b].shape[1] for b in range(bnum)])
                height = max([buffer[0][b].shape[2] for b in range(bnum)])

                im_list = []
                for b in range(bnum):
                    im_list.append(np.pad(buffer[0][b], [(0, 0), (0, max(0, width - buffer[0][b].shape[1])),
                                                         (0, max(0, height - buffer[0][b].shape[2])), (0, 0)],
                                          mode='constant'))


                yield np.concatenate(im_list, axis=0), buffer[1], sum(pos_semi_list), \
                      buffer[3], buffer[4], buffer[5], buffer[6], pos_semi_list[0]

                buffer = [[] for i in range(8)]
                # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                # count_time += 1
                # print('generate batch:', time.time() - st, "average;",  avg_time)
                # st = time.time()

    if pattern_type == 1:
        pattern_channel = 3
    else:
        pattern_channel = 2
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type),
        output_types=(
            tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        output_shapes=(
            tf.TensorShape([bnum, None, None, 3]),
            tf.TensorShape([bnum, ]),
            tf.TensorShape([]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 600]),
            tf.TensorShape([None, 64, 64, pattern_channel]),
            tf.TensorShape([])
        )
        )
    # dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32),
    #                                          output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([])))
    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx = iterator.get_next()
    return image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx


def obtain_batch_data_semi1(Pos_augment=15, Neg_select=60, augment_type=0, model_name='', pattern_type=0,
                            zero_shot_type=0, isalign=False, epoch=0, semi_type='default', bnum=2, neg_type_ratio=0):
    assert len(model_name) > 1, model_name

    with open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb") as f:
        Trainval_GT = pickle.load(f, encoding='latin1')
    Trainval_semi = extract_semi_data(semi_type, model_name)
    with open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb") as f:
        Trainval_N = pickle.load(f, encoding='latin1')

    g_func = generator2

    def generator3(Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type):
        buffer = [[] for i in range(7)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        # np.random.seed(0)
        semi_g = generator2(Trainval_semi, {}, Pos_augment, Neg_select, augment_type, False, zero_shot_type, isalign,
                            epoch, )
        for im_orig, image_id, num_pos, Human_augmented, Object_augmented, \
            action_HO, Pattern in g_func(Trainval_GT, Trainval_N, Pos_augment, Neg_select,
                                                              augment_type,
                                                              pattern_type, zero_shot_type, False, epoch,
                                                              ):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(Human_augmented)
            buffer[4].append(Object_augmented)
            buffer[5].append(action_HO)
            buffer[6].append(Pattern)
            for b in range(bnum):
                im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern, = next(semi_g)
                buffer[0].append(im_orig)
                buffer[1].append(image_id)
                buffer[2].append(num_pos)
                buffer[3].append(Human_augmented)
                buffer[4].append(Object_augmented)
                buffer[5].append(action_HO)
                buffer[6].append(Pattern)
                buffer[3][b + 1][:, 0] = b + 1
                buffer[4][b + 1][:, 0] = b + 1
                assert num_pos == len(Human_augmented)
            # print(buffer[3])

            # print(len(buffer[0]))
            # print("inner hint:", buffer[1], 'num_pos:', buffer[2], 'len of h boxes:',len(buffer[3][0]), len(buffer[3][1]),
            #       len(buffer[4][0]), len(buffer[4][1]), len(buffer[5][0]), len(buffer[5][1]), len(buffer[6][0]), len(buffer[6][1]))
            pos_semi_list = []
            if model_name.__contains__('x5new'):
                pos1 = int(buffer[2][0] + (len(buffer[3][0]) - buffer[2][0]) // 8)
                assert len(buffer[3][1]) == buffer[2][1], (len(buffer[3][1]), buffer[2][1],)
                # print(pos1, (len(buffer[3][b+1]) - buffer[2][b+1]) // 8)
                for b in range(bnum):
                    pos_semi_list.append(int(buffer[2][b + 1] + (len(buffer[3][b + 1]) - buffer[2][b + 1]) // 8))
            else:
                pos1 = buffer[2][0]
                for b in range(bnum):
                    pos_semi_list.append(buffer[2][b + 1])
            # print('before', buffer[3])
            for ii in range(3, 7):
                pos_h_boxes = np.concatenate(
                    [buffer[ii][0][:pos1]] + [buffer[ii][pi + 1][:pos2] for pi, pos2 in enumerate(pos_semi_list)],
                    axis=0)
                neg_h_boxes = np.concatenate(
                    [buffer[ii][0][pos1:]] + [buffer[ii][pi + 1][pos2:] for pi, pos2 in enumerate(pos_semi_list)],
                    axis=0)

                buffer[ii] = np.concatenate([pos_h_boxes, neg_h_boxes], axis=0)
                # buffer[ii] = np.concatenate([buffer[ii][0], buffer[ii][1]], axis=0)
            # print('after', buffer[3])
            width = max([buffer[0][b].shape[1] for b in range(bnum + 1)])
            height = max([buffer[0][b].shape[2] for b in range(bnum + 1)])

            im_list = []
            for b in range(bnum + 1):
                im_list.append(np.pad(buffer[0][b], [(0, 0), (0, max(0, width - buffer[0][b].shape[1])),
                                                     (0, max(0, height - buffer[0][b].shape[2])), (0, 0)],
                                      mode='constant'))

            width = max([buffer[7][b].shape[1] for b in range(bnum + 1)])
            height = max([buffer[7][b].shape[2] for b in range(bnum + 1)])

            split_idx = pos1
            yield np.concatenate(im_list, axis=0), buffer[1], pos1 + sum(pos_semi_list), \
                  buffer[3], buffer[4], buffer[5], buffer[6], split_idx

            buffer = [[] for i in range(7)]
            # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
            # count_time += 1
            # print('generate batch:', time.time() - st, "average;",  avg_time)
            # st = time.time()

    pattern_channel = 2
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type),
        output_types=(
            tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        output_shapes=(
            tf.TensorShape([bnum + 1, None, None, 3]),
            tf.TensorShape([bnum + 1, ]),
            tf.TensorShape([]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 600]),
            tf.TensorShape([None, 64, 64, pattern_channel]),
            tf.TensorShape([])
            # tf.TensorShape([2, None, None, None, 1])
        )
        )
    # dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32),
    #                                          output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([])))
    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx = iterator.get_next()
    return image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx


def Augmented_HO_Neg_HICO2(GT, Trainval_Neg, shape, Pos_augment, Neg_select, pose_type=0, isalign=False):
    image_id = GT[0]
    Human = GT[2]
    Object = GT[3]

    action_HO_ = Generate_action_HICO(GT[1])
    action_HO = action_HO_

    Human_augmented = Augmented_box(Human, shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    if isalign:
        while len(Human_augmented) < Pos_augment + 1:
            Human_augmented = np.concatenate(
                [Human_augmented, Human_augmented[-(Pos_augment + 1 - len(Human_augmented)):]], axis=0)

    if isalign:
        while len(Object_augmented) < Pos_augment + 1:
            Object_augmented = np.concatenate(
                [Object_augmented, Object_augmented[-(Pos_augment + 1 - len(Human_augmented)):]], axis=0)
    # print("shape:", Human_augmented.shape, Object_augmented.shape)
    Human_augmented = Human_augmented[:min(len(Human_augmented), len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented), len(Object_augmented))]

    if isalign:
        assert len(Human_augmented) == Pos_augment + 1, (len(Human_augmented), Pos_augment)
    num_pos = len(Human_augmented)
    if pose_type > 0: pose_list = [GT[5]] * num_pos
    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                if pose_type > 0: pose_list.append(Neg[7])
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                if pose_type > 0: pose_list.append(Neg[7])
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)
    if pose_type > 0:
        pattern_channel = 3
    else:
        pattern_channel = 2
    Pattern = np.empty((0, 64, 64, pattern_channel), dtype=np.float32)

    for i in range(num_pos_neg):
        # Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        # there are poses for the negative sample
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:])
        Pattern_ = Pattern_.reshape(1, 64, 64, pattern_channel)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)

    Pattern = Pattern.reshape(num_pos_neg, 64, 64, pattern_channel)
    Human_augmented = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented.reshape(num_pos_neg, 5)
    action_HO = action_HO.reshape(num_pos_neg, 600)

    # print("shape1:", Human_augmented.shape, Object_augmented.shape, num_pos, Neg_select)
    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos


def coco_generator1(Pos_augment=15, Neg_select=30, augment_type=0, with_pose=False, is_zero_shot=0):
    Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO.pkl', "rb"), encoding='latin1')
    Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO.pkl', "rb"), encoding='latin1')
    Neg_select1, Pos_augment1, inters_per_img = get_aug_params(Neg_select, Pos_augment, augment_type)
    index_list = list(range(0, len(Trainval_GT)))
    print("generator1", inters_per_img, Pos_augment1, 'Neg_select:', Neg_select1, augment_type)
    import math
    img_id_index_map = {}
    for i, gt in enumerate(Trainval_GT):
        img_id = gt[0]
        if img_id in img_id_index_map:
            img_id_index_map[img_id].append(i)
        else:
            img_id_index_map[img_id] = [i]
    img_id_list = list(img_id_index_map.keys())
    for k, v in img_id_index_map.items():
        for i in range(math.ceil(len(v) * 1.0 / inters_per_img) - 1):
            img_id_list.append(k)
    import copy
    while True:
        running_map = copy.deepcopy(img_id_index_map)
        # print('Step: ', i)
        np.random.shuffle(index_list)
        for k in running_map.keys():
            np.random.shuffle(running_map[k])

        for img_id_tmp in img_id_list:
            gt_ids = running_map[img_id_tmp][:inters_per_img]
            running_map[img_id_tmp] = running_map[img_id_tmp][inters_per_img:]

            image_id = img_id_tmp
            im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(
                12) + '.jpg'
            import os
            if not os.path.exists(im_file):
                print('not exist', im_file)
            import cv2
            im = cv2.imread(im_file)
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im.shape

            blobs = {}
            blobs['H_boxes'] = np.empty([0, 5], dtype=np.float32)
            blobs['Hsp_boxes'] = np.empty([0, 5], dtype=np.float32)
            blobs['O_boxes'] = np.empty([0, 5], dtype=np.float32)
            blobs['gt_class_sp'] = np.empty([0, 29], dtype=np.float32)
            blobs['gt_class_HO'] = np.empty([0, 29], dtype=np.float32)
            blobs['gt_class_H'] = np.empty([0, 29], dtype=np.float32)
            blobs['gt_class_C'] = np.empty([0, 238], dtype=np.float32)
            blobs['Mask_sp'] = np.empty([0, 29], dtype=np.float32)
            blobs['Mask_HO'] = np.empty([0, 29], dtype=np.float32)
            blobs['Mask_H'] = np.empty([0, 29], dtype=np.float32)
            blobs['sp'] = np.empty([0, 64, 64, 2], dtype=np.float32)

            for i in gt_ids:
                GT = Trainval_GT[i]
                assert GT[0] == image_id

                # im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
                cur_neg_select = Neg_select1
                cur_pos_augment = Pos_augment1
                if augment_type > 1:
                    if i == gt_ids[-1]:
                        cur_neg_select = Neg_select1 * len(gt_ids)
                    else:
                        cur_neg_select = 0
                else:
                    cur_neg_select = Neg_select1
                Pattern, Human_augmented_sp, Human_augmented, Object_augmented, \
                action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, action_compose = Augmented_HO_spNeg(GT,
                                                                                                              Trainval_N,
                                                                                                              im_shape,
                                                                                                              Pos_augment=cur_pos_augment,
                                                                                                              Neg_select=cur_neg_select)

                # blobs['image'] = im_orig
                blobs['H_boxes'] = np.concatenate((blobs['H_boxes'], Human_augmented), axis=0)
                blobs['Hsp_boxes'] = np.concatenate((blobs['Hsp_boxes'], Human_augmented_sp), axis=0)
                blobs['O_boxes'] = np.concatenate((blobs['O_boxes'], Object_augmented), axis=0)
                blobs['gt_class_sp'] = np.concatenate((blobs['gt_class_sp'], action_sp), axis=0)
                blobs['gt_class_HO'] = np.concatenate((blobs['gt_class_HO'], action_HO), axis=0)
                blobs['gt_class_H'] = np.concatenate((blobs['gt_class_H'], action_H), axis=0)
                blobs['gt_class_C'] = np.concatenate((blobs['gt_class_C'], action_compose), axis=0)
                blobs['Mask_sp'] = np.concatenate((blobs['Mask_sp'], mask_sp), axis=0)
                blobs['Mask_HO'] = np.concatenate((blobs['Mask_HO'], mask_HO), axis=0)
                blobs['Mask_H'] = np.concatenate((blobs['Mask_H'], mask_H), axis=0)
                blobs['sp'] = np.concatenate((blobs['sp'], Pattern), axis=0)
            yield (im_orig, image_id, len(blobs['gt_class_H']), blobs)


def coco_generator(Pos_augment=15, Neg_select=30, augment_type=0, with_pose=False, is_zero_shot=0):
    Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_with_pose_obj.pkl', "rb"), encoding='latin1')
    Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO_with_pose_obj.pkl', "rb"), encoding='latin1')
    i = 0
    index_list = list(range(0, len(Trainval_GT)))
    set_list = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (4, 1), (4, 19), (4, 28), (4, 46), (4, 47),
                (4, 48), (4, 49), (4, 51), (4, 52), (4, 54), (4, 55), (4, 56), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7),
                (5, 8), (5, 9), (5, 18), (5, 21), (6, 68), (7, 33), (8, 64), (9, 47), (9, 48), (9, 49), (9, 50),
                (9, 51), (9, 52), (9, 53), (9, 54), (9, 55), (9, 56), (10, 2), (10, 4), (10, 14), (10, 18), (10, 21),
                (10, 25), (10, 27), (10, 29), (10, 57), (10, 58), (10, 60), (10, 61), (10, 62), (10, 64), (11, 31),
                (11, 32), (11, 37), (11, 38), (12, 14), (12, 57), (12, 58), (12, 60), (12, 61), (13, 40), (13, 41),
                (13, 42), (13, 46), (14, 1), (14, 25), (14, 26), (14, 27), (14, 29), (14, 30), (14, 31), (14, 32),
                (14, 33), (14, 34), (14, 35), (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 47),
                (14, 50), (14, 68), (14, 74), (14, 75), (14, 78), (15, 30), (15, 33), (16, 43), (16, 44), (16, 45),
                (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 11), (18, 14), (18, 15),
                (18, 16), (18, 17), (18, 18), (18, 19), (18, 20), (18, 21), (18, 24), (18, 25), (18, 26), (18, 27),
                (18, 28), (18, 29), (18, 30), (18, 31), (18, 32), (18, 33), (18, 34), (18, 35), (18, 36), (18, 37),
                (18, 38), (18, 39), (18, 40), (18, 41), (18, 42), (18, 43), (18, 44), (18, 45), (18, 46), (18, 47),
                (18, 48), (18, 49), (18, 51), (18, 53), (18, 54), (18, 55), (18, 56), (18, 57), (18, 61), (18, 62),
                (18, 63), (18, 64), (18, 65), (18, 66), (18, 67), (18, 68), (18, 73), (18, 74), (18, 75), (18, 77),
                (19, 35), (19, 39), (20, 33), (21, 31), (21, 32), (23, 1), (23, 11), (23, 19), (23, 20), (23, 24),
                (23, 28), (23, 34), (23, 49), (23, 53), (23, 56), (23, 61), (23, 63), (23, 64), (23, 67), (23, 68),
                (23, 73), (24, 74), (25, 1), (25, 2), (25, 4), (25, 8), (25, 9), (25, 14), (25, 15), (25, 16), (25, 17),
                (25, 18), (25, 19), (25, 21), (25, 25), (25, 26), (25, 27), (25, 28), (25, 29), (25, 30), (25, 31),
                (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37), (25, 38), (25, 39), (25, 40), (25, 41),
                (25, 42), (25, 43), (25, 44), (25, 45), (25, 46), (25, 47), (25, 48), (25, 49), (25, 50), (25, 51),
                (25, 52), (25, 53), (25, 54), (25, 55), (25, 56), (25, 57), (25, 64), (25, 65), (25, 66), (25, 67),
                (25, 68), (25, 73), (25, 74), (25, 77), (25, 78), (25, 79), (25, 80), (26, 32), (26, 37), (28, 30),
                (28, 33)]

    while True:
        # print('Step: ', i)
        np.random.shuffle(index_list)
        for i in index_list:
            GT = Trainval_GT[i]
            image_id = GT[0]
            im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(
                12) + '.jpg'
            im = cv2.imread(im_file)
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im_orig.shape
            im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

            Pattern, Human_augmented_sp, Human_augmented, Object_augmented, \
            action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, gt_compose = Augmented_HO_spNeg(GT, Trainval_N,
                                                                                                      im_shape,
                                                                                                      Pos_augment,
                                                                                                      Neg_select)

            blobs = {}
            # blobs['image'] = im_orig
            blobs['H_boxes'] = Human_augmented
            blobs['Hsp_boxes'] = Human_augmented_sp
            blobs['O_boxes'] = Object_augmented
            blobs['gt_class_sp'] = action_sp
            blobs['gt_class_HO'] = action_HO
            blobs['gt_class_H'] = action_H
            blobs['gt_class_C'] = gt_compose
            blobs['Mask_sp'] = mask_sp
            blobs['Mask_HO'] = mask_HO
            blobs['Mask_H'] = mask_H
            blobs['sp'] = Pattern

            yield (im_orig, image_id, len(action_H), blobs)


def obtain_coco_data(Pos_augment=15, Neg_select=30, augment_type=0):
    if augment_type == 0:
        g = coco_generator
    else:
        g = coco_generator1
    # generator()
    dataset = tf.data.Dataset.from_generator(partial(g, Pos_augment, Neg_select, augment_type),
                                             output_types=(tf.float32, tf.int32, tf.int32, {
                                                 'H_boxes': tf.float32,
                                                 'Hsp_boxes': tf.float32,
                                                 'O_boxes': tf.float32,
                                                 'gt_class_sp': tf.float32,
                                                 'gt_class_HO': tf.float32,
                                                 'gt_class_H': tf.float32,
                                                 'gt_class_C': tf.float32,
                                                 'Mask_sp': tf.float32,
                                                 'Mask_HO': tf.float32,
                                                 'Mask_H': tf.float32,
                                                 'sp': tf.float32,
                                             }), output_shapes=(
        tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
        {
            'H_boxes': tf.TensorShape([None, 5]),
            'Hsp_boxes': tf.TensorShape([None, 5]),
            'O_boxes': tf.TensorShape([None, 5]),
            'gt_class_sp': tf.TensorShape([None, 29]),
            'gt_class_HO': tf.TensorShape([None, 29]),
            'gt_class_H': tf.TensorShape([None, 29]),
            'gt_class_C': tf.TensorShape([None, 238]),
            'Mask_sp': tf.TensorShape([None, 29]),
            'Mask_HO': tf.TensorShape([None, 29]),
            'Mask_H': tf.TensorShape([None, 29]),
            'sp': tf.TensorShape([None, 64, 64, 3]),
        }))

    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, blobs = iterator.get_next()
    return image, image_id, num_pos, blobs
    # image, num_pos = iterator.get_next()
    # return image, num_pos


def obtain_coco_data1(Pos_augment=15, Neg_select=30, augment_type=0, with_pose=False, is_zero_shot=0):
    if augment_type == 0:
        g_func = coco_generator
    else:
        g_func = coco_generator1

    def generator3(Pos_augment, Neg_select, augment_type, with_pose, is_zero_shot):
        buffer = [[] for i in range(4)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        for im_orig, image_id, num_pos, blobs in g_func(Pos_augment, Neg_select, augment_type, with_pose, is_zero_shot):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(blobs)

            if len(buffer[0]) > 1:

                if buffer[2][0] < buffer[2][1]:
                    # make sure the first batch is less.
                    for i in range(len(buffer)):
                        tmp = buffer[i][0]
                        buffer[i][0] = buffer[i][1]
                        buffer[i][1] = tmp

                yield buffer[0][0], buffer[1][0], buffer[2][0], buffer[3][0], buffer[0][1], buffer[1][1], buffer[2][1], \
                      buffer[3][1],

                buffer = [[] for i in range(4)]
                # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                # count_time += 1
                # print('generate batch:', time.time() - st, "average;",  avg_time)
                # st = time.time()

    # generator()
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Pos_augment, Neg_select, augment_type, with_pose, is_zero_shot),
        output_types=(tf.float32, tf.int32, tf.int32, {
            'H_boxes': tf.float32,
            'Hsp_boxes': tf.float32,
            'O_boxes': tf.float32,
            'gt_class_sp': tf.float32,
            'gt_class_HO': tf.float32,
            'gt_class_H': tf.float32,
            'gt_class_C': tf.float32,
            'Mask_sp': tf.float32,
            'Mask_HO': tf.float32,
            'Mask_H': tf.float32,
            'sp': tf.float32,
        }, tf.float32, tf.int32, tf.int32, {
                          'H_boxes': tf.float32,
                          'Hsp_boxes': tf.float32,
                          'O_boxes': tf.float32,
                          'gt_class_sp': tf.float32,
                          'gt_class_HO': tf.float32,
                          'gt_class_H': tf.float32,
                          'gt_class_C': tf.float32,
                          'Mask_sp': tf.float32,
                          'Mask_HO': tf.float32,
                          'Mask_H': tf.float32,
                          'sp': tf.float32,
                      }), output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, 29]),
                                             'gt_class_HO': tf.TensorShape([None, 29]),
                                             'gt_class_H': tf.TensorShape([None, 29]),
                                             'gt_class_C': tf.TensorShape([None, 238]),
                                             'Mask_sp': tf.TensorShape([None, 29]),
                                             'Mask_HO': tf.TensorShape([None, 29]),
                                             'Mask_H': tf.TensorShape([None, 29]),
                                             'sp': tf.TensorShape([None, 64, 64, 3]),
                                         }, tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, 29]),
                                             'gt_class_HO': tf.TensorShape([None, 29]),
                                             'gt_class_H': tf.TensorShape([None, 29]),
                                             'gt_class_C': tf.TensorShape([None, 238]),
                                             'Mask_sp': tf.TensorShape([None, 29]),
                                             'Mask_HO': tf.TensorShape([None, 29]),
                                             'Mask_H': tf.TensorShape([None, 29]),
                                             'sp': tf.TensorShape([None, 64, 64, 3]),
                                         }))

    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, blobs, image1, image_id1, num_pos1, blobs1 = iterator.get_next()
    return [image, image1], [image_id, image_id1], [num_pos, num_pos1], [blobs, blobs1]


def obtain_coco_data_hoicoco_24(Pos_augment = 15, Neg_select=30, augment_type = 0, pattern_type=False, is_zero_shot=0, type=0):
    if type == 0:
        verb_num = 24
        g_func = coco_generator2
    elif type == 1:
        verb_num = 21
        g_func = coco_generator3

    def generator3(Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot):
        buffer = [[] for i in range(4)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        for im_orig, image_id, num_pos, blobs in g_func(Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(blobs)


            # print(im_orig.shape, image_id, num_pos,
            if len(buffer[0]) > 1:

                if buffer[2][0] < buffer[2][1]:
                    # make sure the first batch is less.
                    for i in range(len(buffer)):
                        tmp = buffer[i][0]
                        buffer[i][0] = buffer[i][1]
                        buffer[i][1] = tmp

                yield buffer[0][0], buffer[1][0], buffer[2][0], buffer[3][0],buffer[0][1], buffer[1][1], buffer[2][1],buffer[3][1],

                buffer = [[] for i in range(4)]
                # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                # count_time += 1
                # print('generate batch:', time.time() - st, "average;",  avg_time)
                # st = time.time()
    dataset = tf.data.Dataset.from_generator(partial(generator3, Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot),
                                             output_types=(tf.float32, tf.int32, tf.int32, {
        'H_boxes': tf.float32,
        'Hsp_boxes': tf.float32,
        'pose_box':tf.float32,
        'O_boxes': tf.float32,
        'gt_class_sp': tf.float32,
        'gt_class_HO': tf.float32,
        'gt_class_H': tf.float32,
        'gt_class_C': tf.float32,
        'Mask_sp': tf.float32,
        'Mask_HO': tf.float32,
        'Mask_H': tf.float32,
        'sp': tf.float32,
    },tf.float32, tf.int32, tf.int32, {
        'H_boxes': tf.float32,
        'Hsp_boxes': tf.float32,
        'pose_box': tf.float32,
        'O_boxes': tf.float32,
        'gt_class_sp': tf.float32,
        'gt_class_HO': tf.float32,
        'gt_class_H': tf.float32,
        'gt_class_C': tf.float32,
        'Mask_sp': tf.float32,
        'Mask_HO': tf.float32,
        'Mask_H': tf.float32,
        'sp': tf.float32,
    }), output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                       {
                           'H_boxes': tf.TensorShape([None, 5]),
                           'Hsp_boxes': tf.TensorShape([None, 5]),
                           'pose_box': tf.TensorShape([None, 5]),
                           'O_boxes': tf.TensorShape([None, 5]),
                           'gt_class_sp': tf.TensorShape([None, verb_num]),
                           'gt_class_HO': tf.TensorShape([None, verb_num]),
                           'gt_class_H': tf.TensorShape([None, verb_num]),
                           'gt_class_C': tf.TensorShape([None, 222]),
                           'Mask_sp': tf.TensorShape([None, verb_num]),
                           'Mask_HO': tf.TensorShape([None, verb_num]),
                           'Mask_H': tf.TensorShape([None, verb_num]),
                           'sp': tf.TensorShape([None, 64, 64, 2]),
                       },tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                       {
                           'H_boxes': tf.TensorShape([None, 5]),
                           'Hsp_boxes': tf.TensorShape([None, 5]),
                           'pose_box': tf.TensorShape([None, 5]),
                           'O_boxes': tf.TensorShape([None, 5]),
                           'gt_class_sp': tf.TensorShape([None, verb_num]),
                           'gt_class_HO': tf.TensorShape([None, verb_num]),
                           'gt_class_H': tf.TensorShape([None, verb_num]),
                           'gt_class_C': tf.TensorShape([None, 222]),
                           'Mask_sp': tf.TensorShape([None, verb_num]),
                           'Mask_HO': tf.TensorShape([None, verb_num]),
                           'Mask_H': tf.TensorShape([None, verb_num]),
                           'sp': tf.TensorShape([None, 64, 64, 2]),
                       }))

    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, blobs, image1, image_id1, num_pos1, blobs1 = iterator.get_next()
    return [image, image1], [image_id, image_id1], [num_pos, num_pos1], [blobs, blobs1]



def get_new_Trainval_N(Trainval_N, is_zero_shot, unseen_idx):
    if is_zero_shot > 0:
        new_Trainval_N = {}
        for k in Trainval_N.keys():
            new_Trainval_N[k] = []
            for item in Trainval_N[k]: # the original code include a bug (k is wrongly set to 4)
                if item[1] not in unseen_idx:
                    new_Trainval_N[k].append(item)
        Trainval_N = new_Trainval_N
    return Trainval_N


def get_zero_shot_type(model_name):
    zero_shot_type = 0
    if model_name.__contains__('_zs_'):
        # for open long-tailed hoi detection
        zero_shot_type = 7
    elif model_name.__contains__('zsnrare'):
        zero_shot_type = 4
    elif model_name.__contains__('_zsrare_'):
        zero_shot_type = 3
    elif model_name.__contains__('_zs11_'):
        # for unseen object
        zero_shot_type = 11
    elif model_name.__contains__('_zs3_'):
        # for VCL model
        zero_shot_type = 3
    elif model_name.__contains__('_zs4_'):
        zero_shot_type = 4
    return zero_shot_type


def get_epoch_iters(model_name):
    epoch_iters = 43273
    if model_name.__contains__('zsnrare'):
        epoch_iters = 20000
    elif model_name.__contains__('zs_'):
        epoch_iters = 20000
    elif model_name.__contains__('_zs4_'):
        epoch_iters = 20000
    elif model_name.__contains__('zsrare'):
        epoch_iters = 40000
    else:
        epoch_iters = 43273
    return epoch_iters


def get_augment_type(model_name):
    augment_type = 0
    if model_name.__contains__('_aug5'):
        augment_type = 4
    elif model_name.__contains__('_aug6'):
        augment_type = 5
    else:
        # raise Exception('params wrong', args.model)
        pass
    return augment_type


def get_unseen_index(zero_shot_type):
    unseen_idx = None
    if zero_shot_type == 3:
        # rare first
        unseen_idx = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                      70, 416,
                      389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260,
                      596, 345, 189,
                      205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                      158, 195,
                      238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8,
                      188, 216, 597,
                      77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354,
                      104, 55, 50,
                      198, 168, 391, 192, 595, 136, 581]
    elif zero_shot_type == 4:
        # non rare first
        unseen_idx = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
                      212, 472, 61,
                      457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230,
                      385, 73,
                      159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
                      29, 594, 346,
                      456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266,
                      304, 6, 572,
                      529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
                      246, 173, 506,
                      383, 93, 516, 64]
    elif zero_shot_type == 11:
        unseen_idx = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                      294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                      338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                      429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                      463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                      537, 558, 559, 560, 561, 595, 596, 597, 598, 599]
        #  miss [ 5, 6, 28, 56, 88] verbs 006  break    007  brush_with 029  flip  057  move  089  slide
    elif zero_shot_type == 7:
        # 24 rare merge of zs3 & zs4
        unseen_idx = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416, 389,
         90, 38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61,
         457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73, 159,
         190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346, 456, 589,
         45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572, 529, 312,
         9]
        # 22529, 14830, 22493, 17411, 21912,
    return unseen_idx


def generator2(Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type, pattern_type, zero_shot_type, isalign,
               epoch=0):
    """
    :param Trainval_GT:
    :param Trainval_N:
    :param Pos_augment:
    :param Neg_select:
    :param augment_type:
    :param pattern_type:
    :return:
    """
    # import skimage
    # assert skimage.__version__ == '0.14.2', "The version of skimage might affect the speed largely. I use 0.14.2"
    Neg_select1, Pos_augment1, inters_per_img = get_aug_params(Neg_select, Pos_augment, augment_type)
    unseen_idx = get_unseen_index(zero_shot_type)
    Trainval_N = get_new_Trainval_N(Trainval_N, zero_shot_type, unseen_idx)
    print("generator2", inters_per_img, Pos_augment1, 'Neg_select:', Neg_select1, augment_type, 'zero shot:',
          zero_shot_type)
    import math
    img_id_index_map = {}
    for i, gt in enumerate(Trainval_GT):
        img_id = gt[0]
        if img_id in img_id_index_map:
            img_id_index_map[img_id].append(i)
        else:
            img_id_index_map[img_id] = [i]
    img_id_list = list(img_id_index_map.keys())
    for k, v in img_id_index_map.items():
        for i in range(math.ceil(len(v) * 1.0 / inters_per_img) - 1):
            img_id_list.append(k)
    import copy
    import time
    st = time.time()
    count_time = 0
    avg_time = 0
    while True:
        running_map = copy.deepcopy(img_id_index_map)
        # print('Step: ', i)
        np.random.shuffle(img_id_list)
        for k in running_map.keys():
            np.random.shuffle(running_map[k])

        for img_id_tmp in img_id_list:
            gt_ids = running_map[img_id_tmp][:inters_per_img]
            running_map[img_id_tmp] = running_map[img_id_tmp][inters_per_img:]
            Pattern_list = []
            Human_augmented_list = []
            Object_augmented_list = []
            action_HO_list = []
            num_pos_list = 0
            mask_all_list = []

            image_id = img_id_tmp
            if image_id in [528, 791, 1453, 2783, 3489, 3946, 3946, 11747, 11978, 12677, 16946, 17833, 19218, 19218,
                            22347, 27293, 27584, 28514, 33683, 35399]:
                # This is a list contain multiple objects within the same object box. It seems like wrong annotations.
                # We remove those images. This do not affect the performance in our experiment.
                continue
            im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (
                str(image_id)).zfill(
                8) + '.jpg'
            # id, gt, h, o
            # print(gt_ids, gt_ids[0], Trainval_GT[gt_ids[0]])
            import cv2
            import os
            if not os.path.exists(im_file):
                print('not exist', im_file)
                continue
            im = cv2.imread(im_file)
            if im is None:
                print('node', im_file)
                continue
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im.shape
            import os
            # print('generate batch read image:', time.time() - st, "average;", avg_time)
            for i in gt_ids:
                GT = Trainval_GT[i]
                # rare data
                if zero_shot_type > 0:
                    has_rare = False
                    for label in GT[1]:
                        if label in unseen_idx:
                            has_rare = True
                    if has_rare:
                        continue
                assert GT[0] == image_id

                # im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

                cur_pos_augment = Pos_augment1
                if augment_type > 1:
                    if i == gt_ids[-1]:  # This must be -1
                        cur_neg_select = Neg_select1 * len(gt_ids)
                    else:
                        cur_neg_select = 0
                else:
                    cur_neg_select = Neg_select1
                # st1 = time.time()

                Pattern, Human_augmented, Object_augmented, action_HO, num_pos = Augmented_HO_Neg_HICO(
                    GT,
                    Trainval_N,
                    im_shape,
                    Pos_augment=cur_pos_augment,
                    Neg_select=cur_neg_select,
                    pattern_type=pattern_type,
                    isalign=isalign)

                # maintain same number of augmentation,

                # print('generate batch read image:', i, time.time() - st1, cur_neg_select, len(Trainval_N[image_id]) if image_id in Trainval_N else 0)
                Pattern_list.append(Pattern)
                Human_augmented_list.append(Human_augmented)
                Object_augmented_list.append(Object_augmented)
                action_HO_list.append(action_HO)
                num_pos_list += num_pos
                # print('item:', Pattern.shape, num_pos)
            if len(Pattern_list) <= 0:
                continue
            Pattern = np.concatenate(Pattern_list, axis=0)
            Human_augmented = np.concatenate(Human_augmented_list, axis=0)
            Object_augmented = np.concatenate(Object_augmented_list, axis=0)
            action_HO = np.concatenate(action_HO_list, axis=0)
            num_pos = num_pos_list
            im_orig = np.expand_dims(im_orig, axis=0)
            yield (im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern)
        if augment_type < 0:
            break


def get_aug_params(Neg_select, Pos_augment, augment_type):
    Pos_augment1 = Pos_augment
    Neg_select1 = Neg_select
    inters_per_img = 2
    if augment_type == 0:
        inters_per_img = 1
        Pos_augment1 = 15
        Neg_select1 = 60
    elif augment_type == 4:
        inters_per_img = 5
        Pos_augment1 = 6
        Neg_select1 = 24
    elif augment_type == 5:
        inters_per_img = 7
        Pos_augment1 = 10
        Neg_select1 = 40
    return Neg_select1, Pos_augment1, inters_per_img


def get_vcoco_aug_params(Neg_select, Pos_augment, augment_type):
    Pos_augment1 = Pos_augment
    Neg_select1 = Neg_select
    inters_per_img = 2
    if augment_type == 0:
        inters_per_img = 1
        Pos_augment1 = 15
        Neg_select1 = 30
    elif augment_type == 1:
        inters_per_img = 2
        Pos_augment1 = 15
        Neg_select1 = 30
    elif augment_type == 2:
        inters_per_img = 3
        Pos_augment1 = 15
        Neg_select1 = 30
    elif augment_type == -1:
        inters_per_img = 1
        Pos_augment1 = 0
        Neg_select1 = 0
    return Neg_select1, Pos_augment1, inters_per_img


def obtain_data(Pos_augment=15, Neg_select=60, augment_type=0, pattern_type=0, zero_shot_type=0, isalign=False,
                epoch=0, coco=False, neg_type=0):
    with open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb") as f:
        Trainval_N = pickle.load(f, encoding='latin1')
    if not coco:
        with open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb") as f:
            Trainval_GT = pickle.load(f, encoding='latin1')
    elif coco == 2:
        # 115904
        with open(cfg.DATA_DIR + '/' + 'new_list_pickle_2.pkl', "rb") as f:
            Trainval_GT = pickle.load(f, encoding='latin1')
    elif coco == 3:
        # 115904
        with open(cfg.DATA_DIR + '/' + 'new_list_pickle_3.pkl', "rb") as f:
            Trainval_GT = pickle.load(f, encoding='latin1')
        with open(cfg.DATA_DIR + '/' + 'new_neg_dict.pkl', "rb") as f:
            Trainval_N1 = pickle.load(f, encoding='latin1')
        for k in Trainval_N:
            if k in Trainval_N1:
                Trainval_N[k].extend(Trainval_N1[k])
    else:
        print('Trainval_GT_HICO_COCO')
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_COCO.pkl', "rb"), encoding='latin1')

    dataset = tf.data.Dataset.from_generator(partial(generator2, Trainval_GT, Trainval_N, Pos_augment, Neg_select,
                                                     augment_type, pattern_type, zero_shot_type, isalign, epoch,
                                                     ), output_types=(
        tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32),
                                             output_shapes=(
                                                 tf.TensorShape([1, None, None, 3]), tf.TensorShape([]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([None, 5]), tf.TensorShape([None, 5]),
                                                 tf.TensorShape([None, 600]),
                                                 tf.TensorShape([None, 64, 64, 2])))
    # (im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern)
    # dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32),
    #                                          output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([])))
    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = iterator.get_next()
    return image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp


def obtain_test_data(Pos_augment=15, Neg_select=60, augment_type=0, with_pose=False, large_neg_for_ho=False,
                     isalign=False):
    Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_GT_HICO.pkl', "rb"), encoding='latin1')
    Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_GT_HICO.pkl', "rb"), encoding='latin1')

    g = generator2
    dataset = tf.data.Dataset.from_generator(
        partial(g, Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type, with_pose, 0, isalign),
        output_types=(
            tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
            tf.TensorShape([None, 5]), tf.TensorShape([None, 5]),
            tf.TensorShape([None, 600]),
            tf.TensorShape([None, 64, 64, 2]),
        ))
    # (im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern)
    # dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32),
    #                                          output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([])))
    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = iterator.get_next()
    return image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp


def obtain_coco_data_hoicoco(Pos_augment=15, Neg_select=30, augment_type=0, pattern_type=False, is_zero_shot=0, type=0):
    if type == 1:
        verb_num = 21
        g_func = coco_generator3

    def generator3(Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot):
        buffer = [[] for i in range(4)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        for im_orig, image_id, num_pos, blobs in g_func(Pos_augment, Neg_select, augment_type, pattern_type,
                                                        is_zero_shot):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(blobs)

            # print(im_orig.shape, image_id, num_pos,
            if len(buffer[0]) > 1:

                if buffer[2][0] < buffer[2][1]:
                    # make sure the first batch is less.
                    for i in range(len(buffer)):
                        tmp = buffer[i][0]
                        buffer[i][0] = buffer[i][1]
                        buffer[i][1] = tmp

                yield buffer[0][0], buffer[1][0], buffer[2][0], buffer[3][0], buffer[0][1], buffer[1][1], buffer[2][1], \
                      buffer[3][1],

                buffer = [[] for i in range(4)]
                # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                # count_time += 1
                # print('generate batch:', time.time() - st, "average;",  avg_time)
                # st = time.time()

    # generator()
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot),
        output_types=(tf.float32, tf.int32, tf.int32, {
            'H_boxes': tf.float32,
            'Hsp_boxes': tf.float32,
            'O_boxes': tf.float32,
            'gt_class_sp': tf.float32,
            'gt_class_HO': tf.float32,
            'gt_class_H': tf.float32,
            'gt_class_C': tf.float32,
            'Mask_sp': tf.float32,
            'Mask_HO': tf.float32,
            'Mask_H': tf.float32,
            'sp': tf.float32,
        }, tf.float32, tf.int32, tf.int32, {
                          'H_boxes': tf.float32,
                          'Hsp_boxes': tf.float32,
                          'O_boxes': tf.float32,
                          'gt_class_sp': tf.float32,
                          'gt_class_HO': tf.float32,
                          'gt_class_H': tf.float32,
                          'gt_class_C': tf.float32,
                          'Mask_sp': tf.float32,
                          'Mask_HO': tf.float32,
                          'Mask_H': tf.float32,
                          'sp': tf.float32,
                      }), output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, verb_num]),
                                             'gt_class_HO': tf.TensorShape([None, verb_num]),
                                             'gt_class_H': tf.TensorShape([None, verb_num]),
                                             'gt_class_C': tf.TensorShape([None, 222]),
                                             'Mask_sp': tf.TensorShape([None, verb_num]),
                                             'Mask_HO': tf.TensorShape([None, verb_num]),
                                             'Mask_H': tf.TensorShape([None, verb_num]),
                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                         }, tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, verb_num]),
                                             'gt_class_HO': tf.TensorShape([None, verb_num]),
                                             'gt_class_H': tf.TensorShape([None, verb_num]),
                                             'gt_class_C': tf.TensorShape([None, 222]),
                                             'Mask_sp': tf.TensorShape([None, verb_num]),
                                             'Mask_HO': tf.TensorShape([None, verb_num]),
                                             'Mask_H': tf.TensorShape([None, verb_num]),
                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                         }))

    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, blobs, image1, image_id1, num_pos1, blobs1 = iterator.get_next()
    return [image, image1], [image_id, image_id1], [num_pos, num_pos1], [blobs, blobs1]


def coco_generator2(Pos_augment = 15, Neg_select=30, augment_type = 0, pattern_type=False, is_zero_shot=0):
    Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_obj_24.pkl', "rb"), encoding='latin1')
    Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO_obj_24.pkl', "rb"), encoding='latin1')
    i = 0
    index_list = list(range(0, len(Trainval_GT)))

    while True:
        # print('Step: ', i)
        np.random.shuffle(index_list)
        for i in index_list:

            GT = Trainval_GT[i]
            image_id = GT[0]
            im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(
                12) + '.jpg'
            im = cv2.imread(im_file)
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im_orig.shape
            im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

            Pattern, Human_augmented_sp, Human_augmented, Object_augmented, \
            action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, gt_compose = Augmented_HO_spNeg2(GT, Trainval_N, im_shape, Pos_augment, Neg_select)

            blobs = {}
            # blobs['image'] = im_orig
            blobs['H_boxes'] = Human_augmented
            blobs['Hsp_boxes'] = Human_augmented_sp
            blobs['O_boxes'] = Object_augmented
            blobs['gt_class_sp'] = action_sp
            blobs['gt_class_HO'] = action_HO
            blobs['gt_class_H'] = action_H
            blobs['gt_class_C'] = gt_compose
            blobs['Mask_sp'] = mask_sp
            blobs['Mask_HO'] = mask_HO
            blobs['Mask_H'] = mask_H
            blobs['sp'] = Pattern

            # blobs['H_num'] = len(action_H)
            # print(image_id, len(action_H))
            yield (im_orig, image_id, len(action_H), blobs)
            # print(i, image_id, len(Trainval_GT))
            # i += 1
            # i = i % len(Trainval_GT)




def coco_generator3(Pos_augment = 15, Neg_select=30, augment_type = 0, pattern_type=False, is_zero_shot=0):
    Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_obj_21.pkl', "rb"), encoding='latin1')
    Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO_obj_21.pkl', "rb"), encoding='latin1')
    i = 0
    index_list = list(range(0, len(Trainval_GT)))
    print(len(index_list))

    while True:
        # print('Step: ', i)
        np.random.shuffle(index_list)
        for i in index_list:

            GT = Trainval_GT[i]
            image_id = GT[0]
            im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(
                12) + '.jpg'
            im = cv2.imread(im_file)
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im_orig.shape
            im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

            Pattern, Human_augmented_sp, Human_augmented, Object_augmented, \
            action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, gt_compose = Augmented_HO_spNeg3(GT, Trainval_N, im_shape, Pos_augment, Neg_select)

            blobs = {}
            # blobs['image'] = im_orig
            blobs['H_boxes'] = Human_augmented
            blobs['Hsp_boxes'] = Human_augmented_sp
            blobs['O_boxes'] = Object_augmented
            blobs['gt_class_sp'] = action_sp
            blobs['gt_class_HO'] = action_HO
            blobs['gt_class_H'] = action_H
            blobs['gt_class_C'] = gt_compose
            blobs['Mask_sp'] = mask_sp
            blobs['Mask_HO'] = mask_HO
            blobs['Mask_H'] = mask_H
            blobs['sp'] = Pattern

            yield (im_orig, image_id, len(action_H), blobs)
        if augment_type < 0:
            break

def coco_generator_atl(Pos_augment = 15, Neg_select=0, augment_type = 0, pattern_type=False, is_zero_shot=0, type =0, vcoco_type = 21):
    """
    Here, the name semi means atl. For objects, we do not have verb labels. Thus, we can only provide object id.
    """
    print(type)
    if type == 0:
        # coco 2014 570834 length
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_obj_semi.pkl', "rb"), encoding='latin1')
    elif type == 2:
        # hico 68389 length
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_hico_obj_semi_21.pkl', "rb"),
                                  encoding='latin1')
    elif type == 3:
        # both
        Trainval_GT_hico = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_hico_obj_semi_21.pkl', "rb"),
                                  encoding='latin1')

        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_obj_semi_21.pkl', "rb"),
                                  encoding='latin1')
        for item in Trainval_GT:
            item[0] += MAX_HICO_ID
        Trainval_GT.extend(Trainval_GT_hico)

    elif type == 4:
        # --- 42631
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_vcoco_obj_semi_21.pkl', "rb"),
                                  encoding='latin1')
    elif type == 5:
        # vcoco
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_vcoco1_obj_semi_21.pkl', "rb"),
                                  encoding='latin1')
    else:
        # coco 2014 train 570834
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_obj_semi_21.pkl', "rb"), encoding='latin1')

    i = 0
    index_list = list(range(0, len(Trainval_GT)))

    if vcoco_type == 24:
        g_func = Augmented_HO_spNeg2
    else:
        g_func = Augmented_HO_spNeg3
    while True:
        # print('Step: ', i)
        np.random.shuffle(index_list)
        for i in index_list:

            GT = Trainval_GT[i]
            image_id = GT[0]
            if type == 2:
                im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (
                    str(image_id)).zfill(
                    8) + '.jpg'
            elif type == 3:
                if image_id < MAX_HICO_ID:
                    # obj365
                    tmp_id = image_id
                    im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (
                        str(image_id)).zfill(
                        8) + '.jpg'
                    pass
                else:
                    tmp_id = image_id - MAX_HICO_ID
                    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(tmp_id)).zfill(
                        12) + '.jpg'
                    import os
                    if not os.path.exists(im_file):
                        im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (
                            str(tmp_id)).zfill(12) + '.jpg'
                        if not os.path.exists(im_file):
                            print(im_file)
                import os
                if not os.path.exists(im_file):
                    print(im_file)

            elif type == 6:
                im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(
                    12) + '.jpg'
                import os
                if not os.path.exists(im_file):
                    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (
                        str(image_id)).zfill(12) + '.jpg'
                if not os.path.exists(im_file):
                    print(im_file)
            elif type == 7:
                if image_id >= MAX_COCO_ID:
                    # obj365
                    tmp_id = image_id - MAX_COCO_ID
                    im_file = cfg.LOCAL_DATA + '/dataset/Objects365/Images/train/train/obj365_train_' + (str(tmp_id)).zfill(
                        12) + '.jpg'
                    pass
                else:
                    tmp_id = image_id
                    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(tmp_id)).zfill(
                        12) + '.jpg'
                    import os
                    if not os.path.exists(im_file):
                        im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (
                            str(tmp_id)).zfill(12) + '.jpg'
                        if not os.path.exists(im_file):
                            print(im_file)
                import os
                if not os.path.exists(im_file):
                    print(im_file)
            else:
                im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(
                    12) + '.jpg'
            im = cv2.imread(im_file)
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im_orig.shape
            im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

            Pattern, Human_augmented_sp, Human_augmented, Object_augmented, \
            action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, gt_compose = g_func(GT, {}, im_shape, Pos_augment, Neg_select)

            blobs = {}
            # blobs['image'] = im_orig
            blobs['H_boxes'] = Human_augmented
            blobs['Hsp_boxes'] = Human_augmented_sp
            blobs['O_boxes'] = Object_augmented
            blobs['gt_class_sp'] = action_sp
            blobs['gt_class_HO'] = action_HO
            blobs['gt_class_H'] = action_H
            blobs['gt_class_C'] = gt_compose
            blobs['Mask_sp'] = mask_sp
            blobs['Mask_HO'] = mask_HO
            blobs['Mask_H'] = mask_H
            blobs['sp'] = Pattern

            # blobs['H_num'] = len(action_H)
            # print(image_id, len(action_H))
            yield (im_orig, image_id, len(action_H), blobs)
            # print(i, image_id, len(Trainval_GT))
            # i += 1
            # i = i % len(Trainval_GT)


def obtain_coco_data2(Pos_augment = 15, Neg_select=30, augment_type = 0, type =0 ):
    # Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO.pkl', "rb"), encoding='latin1')
    # Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO.pkl', "rb"), encoding='latin1')
    if type == 0:
        compose_classes = 222
        verb_num = 24
        g_func = coco_generator2
    elif type == 1:
        compose_classes = 222
        verb_num = 21
        g_func = coco_generator3
    elif type == 2:
        compose_classes = 238
        verb_num = 29
        g_func = coco_generator1

    # generator()
    dataset = tf.data.Dataset.from_generator(partial(g_func, Pos_augment, Neg_select, augment_type), output_types=(tf.float32, tf.int32, tf.int32, {
        'H_boxes': tf.float32,
        'Hsp_boxes': tf.float32,
        'O_boxes': tf.float32,
        'gt_class_sp': tf.float32,
        'gt_class_HO': tf.float32,
        'gt_class_H': tf.float32,
        'gt_class_C': tf.float32,
        'Mask_sp': tf.float32,
        'Mask_HO': tf.float32,
        'Mask_H': tf.float32,
        'sp': tf.float32,
    }), output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                       {
                           'H_boxes': tf.TensorShape([None, 5]),
                           'Hsp_boxes': tf.TensorShape([None, 5]),
                           'O_boxes': tf.TensorShape([None, 5]),
                           'gt_class_sp': tf.TensorShape([None, verb_num]),
                           'gt_class_HO': tf.TensorShape([None, verb_num]),
                           'gt_class_H': tf.TensorShape([None, verb_num]),
                           'gt_class_C': tf.TensorShape([None, compose_classes]),
                           'Mask_sp': tf.TensorShape([None, verb_num]),
                           'Mask_HO': tf.TensorShape([None, verb_num]),
                           'Mask_H': tf.TensorShape([None, verb_num]),
                           'sp': tf.TensorShape([None, 64, 64, 2]),
                       }))

    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, blobs = iterator.get_next()
    return image, image_id, num_pos, blobs
    # image, num_pos = iterator.get_next()
    # return image, num_pos


def obtain_coco_data_atl(Pos_augment=15, Neg_select=30, augment_type=0, pattern_type=False, is_zero_shot=0, type=0, vcoco_type=21):
    if vcoco_type == 21:
        verb_num = 21
        g_func = coco_generator3
    elif vcoco_type == 24:
        verb_num = 24
        g_func = coco_generator2
    else:
        # default
        verb_num = 21
        g_func = coco_generator3

    def generator3(Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot):
        buffer = [[] for i in range(4)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        semi_func = coco_generator_atl(Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot, type, vcoco_type = vcoco_type)
        # semi is atl. a weak-supervised manner.
        for im_orig, image_id, num_pos, blobs in g_func(Pos_augment, Neg_select, augment_type, pattern_type,
                                                        is_zero_shot):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(blobs)

            im_orig, image_id, num_pos, blobs = next(semi_func)
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(blobs)

            # print(im_orig.shape, image_id, num_pos,
            yield buffer[0][0], buffer[1][0], buffer[2][0], buffer[3][0], buffer[0][1], buffer[1][1], buffer[2][1], \
                  buffer[3][1],
            buffer = [[] for i in range(4)]
            # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
            # count_time += 1
            # print('generate batch:', time.time() - st, "average;",  avg_time)
            # st = time.time()

    # generator()
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot),
        output_types=(tf.float32, tf.int32, tf.int32, {
            'H_boxes': tf.float32,
            'Hsp_boxes': tf.float32,
            'O_boxes': tf.float32,
            'gt_class_sp': tf.float32,
            'gt_class_HO': tf.float32,
            'gt_class_H': tf.float32,
            'gt_class_C': tf.float32,
            'Mask_sp': tf.float32,
            'Mask_HO': tf.float32,
            'Mask_H': tf.float32,
            'sp': tf.float32,
        }, tf.float32, tf.int32, tf.int32, {
                          'H_boxes': tf.float32,
                          'Hsp_boxes': tf.float32,
                          'O_boxes': tf.float32,
                          'gt_class_sp': tf.float32,
                          'gt_class_HO': tf.float32,
                          'gt_class_H': tf.float32,
                          'gt_class_C': tf.float32,
                          'Mask_sp': tf.float32,
                          'Mask_HO': tf.float32,
                          'Mask_H': tf.float32,
                          'sp': tf.float32,
                      }), output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, verb_num]),
                                             'gt_class_HO': tf.TensorShape([None, verb_num]),
                                             'gt_class_H': tf.TensorShape([None, verb_num]),
                                             'gt_class_C': tf.TensorShape([None, 222]),
                                             'Mask_sp': tf.TensorShape([None, verb_num]),
                                             'Mask_HO': tf.TensorShape([None, verb_num]),
                                             'Mask_H': tf.TensorShape([None, verb_num]),
                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                         }, tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, verb_num]),
                                             'gt_class_HO': tf.TensorShape([None, verb_num]),
                                             'gt_class_H': tf.TensorShape([None, verb_num]),
                                             'gt_class_C': tf.TensorShape([None, 222]),
                                             'Mask_sp': tf.TensorShape([None, verb_num]),
                                             'Mask_HO': tf.TensorShape([None, verb_num]),
                                             'Mask_H': tf.TensorShape([None, verb_num]),
                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                         }))

    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, blobs, image1, image_id1, num_pos1, blobs1 = iterator.get_next()
    return [image, image1], [image_id, image_id1], [num_pos, num_pos1], [blobs, blobs1]

def obtain_coco_data_hoicoco_24_atl(Pos_augment=15, Neg_select=30, augment_type=0, pattern_type=False, is_zero_shot=0, type=0):
    # default
    verb_num = 24
    g_func = coco_generator2

    def generator3(Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot):
        buffer = [[] for i in range(4)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        semi_func = coco_generator_atl(Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot, type)
        # semi is atl. a weak-supervised manner.
        for im_orig, image_id, num_pos, blobs in g_func(Pos_augment, Neg_select, augment_type, pattern_type,
                                                        is_zero_shot):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(blobs)

            im_orig, image_id, num_pos, blobs = next(semi_func)
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(blobs)

            # print(im_orig.shape, image_id, num_pos,
            yield buffer[0][0], buffer[1][0], buffer[2][0], buffer[3][0], buffer[0][1], buffer[1][1], buffer[2][1], \
                  buffer[3][1],
            buffer = [[] for i in range(4)]
            # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
            # count_time += 1
            # print('generate batch:', time.time() - st, "average;",  avg_time)
            # st = time.time()

    # generator()
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Pos_augment, Neg_select, augment_type, pattern_type, is_zero_shot),
        output_types=(tf.float32, tf.int32, tf.int32, {
            'H_boxes': tf.float32,
            'Hsp_boxes': tf.float32,
            'O_boxes': tf.float32,
            'gt_class_sp': tf.float32,
            'gt_class_HO': tf.float32,
            'gt_class_H': tf.float32,
            'gt_class_C': tf.float32,
            'Mask_sp': tf.float32,
            'Mask_HO': tf.float32,
            'Mask_H': tf.float32,
            'sp': tf.float32,
        }, tf.float32, tf.int32, tf.int32, {
                          'H_boxes': tf.float32,
                          'Hsp_boxes': tf.float32,
                          'O_boxes': tf.float32,
                          'gt_class_sp': tf.float32,
                          'gt_class_HO': tf.float32,
                          'gt_class_H': tf.float32,
                          'gt_class_C': tf.float32,
                          'Mask_sp': tf.float32,
                          'Mask_HO': tf.float32,
                          'Mask_H': tf.float32,
                          'sp': tf.float32,
                      }), output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, verb_num]),
                                             'gt_class_HO': tf.TensorShape([None, verb_num]),
                                             'gt_class_H': tf.TensorShape([None, verb_num]),
                                             'gt_class_C': tf.TensorShape([None, 222]),
                                             'Mask_sp': tf.TensorShape([None, verb_num]),
                                             'Mask_HO': tf.TensorShape([None, verb_num]),
                                             'Mask_H': tf.TensorShape([None, verb_num]),
                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                         }, tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                         {
                                             'H_boxes': tf.TensorShape([None, 5]),
                                             'Hsp_boxes': tf.TensorShape([None, 5]),
                                             'O_boxes': tf.TensorShape([None, 5]),
                                             'gt_class_sp': tf.TensorShape([None, verb_num]),
                                             'gt_class_HO': tf.TensorShape([None, verb_num]),
                                             'gt_class_H': tf.TensorShape([None, verb_num]),
                                             'gt_class_C': tf.TensorShape([None, 222]),
                                             'Mask_sp': tf.TensorShape([None, verb_num]),
                                             'Mask_HO': tf.TensorShape([None, verb_num]),
                                             'Mask_H': tf.TensorShape([None, verb_num]),
                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                         }))

    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, blobs, image1, image_id1, num_pos1, blobs1 = iterator.get_next()
    return [image, image1], [image_id, image_id1], [num_pos, num_pos1], [blobs, blobs1]


def get_epoch_iters(model_name):
    epoch_iters = 43273
    if model_name.__contains__('zsnrare'):
        epoch_iters = 20000
    elif model_name.__contains__('zs_'):
        epoch_iters = 20000
    elif model_name.__contains__('zsrare'):
        epoch_iters = 40000
    else:
        epoch_iters = 43273
    return epoch_iters


def obtain_data_vcl_hico(Pos_augment=15, Neg_select=60, augment_type=0, with_pose=False, zero_shot_type=0, isalign=False,
                         epoch=0):
    # we do not use pose, thus we remove it.
    with open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb") as f:
        Trainval_GT = pickle.load(f, encoding='latin1')
    with open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb") as f:
        Trainval_N = pickle.load(f, encoding='latin1')

    g_func = generator2

    def generator3(Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type):
        buffer = [[] for i in range(7)]
        import time
        st = time.time()
        count_time = 0
        avg_time = 0
        for im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern in g_func(Trainval_GT,
                                                                                                        Trainval_N,
                                                                                                        Pos_augment,
                                                                                                        Neg_select,
                                                                                                        augment_type,
                                                                                                        with_pose,
                                                                                                        zero_shot_type,
                                                                                                        isalign, epoch):
            buffer[0].append(im_orig)
            buffer[1].append(image_id)
            buffer[2].append(num_pos)
            buffer[3].append(Human_augmented)
            buffer[4].append(Object_augmented)
            buffer[5].append(action_HO)
            buffer[6].append(Pattern)
            if len(buffer[0]) > 1:

                # print("inner:", buffer[0][0].shape, buffer[0][1].shape, buffer[1], buffer[2], buffer[3].shape, buffer[4].shape, buffer[5].shape, buffer[6].shape)
                # print("inner:", buffer[1], buffer[2][0], buffer[2][1], buffer[3][0].shape, buffer[3][1].shape, buffer[5][0].shape, buffer[5][1].shape)
                # yield buffer[0][0], buffer[0][1], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6]
                if len(buffer[3][0]) < len(buffer[3][1]):
                    # make sure the second batch is less.
                    for i in range(len(buffer)):
                        tmp = buffer[i][0]
                        buffer[i][0] = buffer[i][1]
                        buffer[i][1] = tmp
                split_idx = len(buffer[5][0])
                buffer = buffer[:3] + [np.concatenate(item, axis=0) for item in buffer[3:]] + buffer[-1:]

                yield buffer[0][0], buffer[0][1], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[
                    6], split_idx

                buffer = [[] for i in range(7)]
                # avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                # count_time += 1
                # print('generate batch:', time.time() - st, "average;",  avg_time)
                # st = time.time()

    if with_pose:
        pattern_channel = 3
    else:
        pattern_channel = 2
    dataset = tf.data.Dataset.from_generator(
        partial(generator3, Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type),
        output_types=(
            tf.float32, tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        output_shapes=(
            tf.TensorShape([1, None, None, 3]),
            tf.TensorShape([1, None, None, 3]),
            tf.TensorShape([2, ]),
            tf.TensorShape([2, ]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 5]),
            tf.TensorShape([None, 600]),
            tf.TensorShape([None, 64, 64, pattern_channel]),
            tf.TensorShape([])
        )
        )
    dataset = dataset.prefetch(100)
    iterator = dataset.make_one_shot_iterator()
    image, image2, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, split_idx = iterator.get_next()
    return [image, image2], image_id, num_pos, [Human_augmented[:split_idx], Human_augmented[split_idx:]], \
           [Object_augmented[:split_idx], Object_augmented[split_idx:]], \
           [action_HO[:split_idx], action_HO[split_idx:]], \
           [sp[:split_idx], sp[split_idx:]]


def Augmented_HO_Neg_HICO_inner(GT, negs, shape, Pos_augment, Neg_select, with_pose):
    image_id = GT[0]
    Human = GT[2]
    Object = GT[3]
    pose_list = []
    if Pos_augment < 0:
        action_HO = np.empty([0, 600])
        Human_augmented = np.empty([0, 5])
        Object_augmented = np.empty([0, 5])
        num_pos = 0
    else:
        action_HO_ = Generate_action_HICO(GT[1])
        action_HO = action_HO_

        Human_augmented = Augmented_box(Human, shape, image_id, Pos_augment)
        Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

        Human_augmented = Human_augmented[:min(len(Human_augmented), len(Object_augmented))]
        Object_augmented = Object_augmented[:min(len(Human_augmented), len(Object_augmented))]

        num_pos = len(Human_augmented)
        for i in range(num_pos - 1):
            action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    if with_pose: pose_list = [GT[5]] * num_pos

    num_pos_neg = len(Human_augmented)

    if with_pose:
        pattern_channel = 3
    else:
        pattern_channel = 2

    Pattern = get_pattern(Human_augmented, Object_augmented, num_pos_neg, pose_list, shape, with_pose)

    if negs is not None and Neg_select > 0:

        if len(negs) < Neg_select:
            Neg_select = len(negs)
            List = range(Neg_select)
        else:
            List = random.sample(range(len(negs)), Neg_select)

        _Human_augmented, _Object_augmented, _action_HO, _Pattern = get_neg_items(List, negs, shape, with_pose)
        Human_augmented = np.concatenate([Human_augmented, _Human_augmented], axis=0)
        Object_augmented = np.concatenate([Object_augmented, _Object_augmented], axis=0)
        action_HO = np.concatenate([action_HO, _action_HO], axis=0)
        Pattern = np.concatenate([Pattern, _Pattern], axis=0)

    num_pos_neg = len(Human_augmented)
    Pattern = Pattern.reshape(num_pos_neg, 64, 64, pattern_channel)
    Human_augmented = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented.reshape(num_pos_neg, 5)
    action_HO = action_HO.reshape(num_pos_neg, 600)

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos


def get_pattern(Human_augmented, Object_augmented, num_pos_neg, pose_list, shape, with_pose):
    pattern_channel = 2
    Pattern = np.empty((0, 64, 64, pattern_channel), dtype=np.float32)
    for i in range(num_pos_neg):
        # Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        # there are poses for the negative sample
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:])
        Pattern_ = Pattern_.reshape(1, 64, 64, pattern_channel)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)
    return Pattern


def get_neg_items(neg_select_list, negs, shape, with_pose):
    action_HO = np.empty([0, 600])
    Human_augmented = np.empty([0, 5])
    Object_augmented = np.empty([0, 5])
    pose_list = []
    for i in range(len(neg_select_list)):
        Neg = negs[neg_select_list[i]]
        if with_pose: pose_list.append(Neg[7])
        Human_augmented = np.concatenate(
            (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
        Object_augmented = np.concatenate(
            (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
        action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern = get_pattern(Human_augmented, Object_augmented, num_pos_neg, pose_list, shape, with_pose)

    return Human_augmented, Object_augmented, action_HO, Pattern
