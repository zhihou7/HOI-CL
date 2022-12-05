
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


from networks.HOI import HOI
from networks.tools import get_convert_matrix


import numpy as np


import os

from ult.ult import get_zero_shot_type, get_unseen_index


class HOIICLNet(HOI):
    def __init__(self, model_name='iCAN_ResNet50_HICO', task_id = 0, incremental_class_pairs = [[(), (), (), ]]):
        super(HOIICLNet, self).__init__(model_name)
        self.task_id = task_id
        self.per_incremental_num_classes = len(incremental_class_pairs[0]) # We should plus 1 for each incremental period
        self.incremental_class_pairs = incremental_class_pairs
        # TODO classes matrix
        if model_name.__contains__('zs11'):
            self.base_classes = 500
        elif model_name.__contains__('zs'):
            self.base_classes = 480
        else:
            self.base_classes = 600
        # incremental classes, 
        self.class_place_dim = 1
        
        zs_convert_matrix = np.zeros([600, self.base_classes], np.float32)
        zs_j = 0
        if model_name.__contains__('zs'):
            zero_shot_type = get_zero_shot_type(model_name)
        elif model_name.__contains__('iCAN_R_union_batch_large2_ml5_def1_ICL_vloss2_l2_rew2_aug5_3_x5new_res101_')\
                and model_name.__contains__('inc'):
            zero_shot_type = get_zero_shot_type(model_name)
        else:
            # TODO  remove
            print("please remove this default setting !!!!!!!!!!!!")
            zero_shot_type = 0
        unseen_idx = get_unseen_index(zero_shot_type)
        if unseen_idx is None:
            unseen_idx = []
        for i in range(600):
            if i in unseen_idx:
                continue
            zs_convert_matrix[i][zs_j] = 1
            zs_j += 1
        print(self.verb_num_classes)
        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix(self.verb_num_classes, self.obj_num_classes)
        new_v_hoi = np.matmul(verb_to_HO_matrix, zs_convert_matrix)
        new_o_hoi = np.matmul(obj_to_HO_matrix, zs_convert_matrix)

        self.sum_num_classes = self.base_classes + sum([len(item) for item in incremental_class_pairs])
        new_verb_to_HOI = np.zeros([self.verb_num_classes, self.sum_num_classes], np.float32)
        new_obj_to_HOI = np.zeros([self.obj_num_classes, self.sum_num_classes], np.float32)

        inc_cls_i = 0
        new_zs_convert_matrix = np.zeros([600, self.sum_num_classes], np.float32)
        old_hoi_id_list = []
        for i in range(self.sum_num_classes):
            if i < self.base_classes:
                v_id = np.argmax(new_v_hoi[:, i])
                o_id = np.argmax(new_o_hoi[:, i])
            else:
                v_id = incremental_class_pairs[inc_cls_i][i - self.base_classes - inc_cls_i*self.per_incremental_num_classes][0]
                o_id = incremental_class_pairs[inc_cls_i][i - self.base_classes- inc_cls_i*self.per_incremental_num_classes][1]
            new_verb_to_HOI[v_id][i] = 1
            new_obj_to_HOI[o_id][i] = 1
            hoi_tmp = verb_to_HO_matrix[v_id] + obj_to_HO_matrix[o_id] == 2.
            if np.sum(hoi_tmp) == 0: # the new HOI do not exist in annotated HOIs.
                continue
            else:
                assert np.sum(hoi_tmp) == 1, (hoi_tmp)
                old_hoi_idx = np.argmax(hoi_tmp)
                assert old_hoi_idx not in old_hoi_id_list, (old_hoi_idx, old_hoi_id_list)
                old_hoi_id_list.append(old_hoi_idx)
                new_zs_convert_matrix[old_hoi_idx][i] = 1.
        self.zs_convert_matrix_base = self.zs_convert_matrix

        """
        convert original HOI label to current HOI label.
        """
        self.zs_convert_matrix = tf.constant(new_zs_convert_matrix, tf.float32)


        self.verb_to_HO_matrix_np = new_verb_to_HOI
        self.obj_to_HO_matrix_np = new_obj_to_HOI
        self.incre_verb_to_HOI = tf.constant(new_verb_to_HOI, tf.float32)
        self.incre_obj_to_HOI = tf.constant(new_obj_to_HOI, tf.float32)

    def set_ph(self, image, image_id, num_pos, Human_augmented, Object_augmented, action_HO=None, sp=None, obj_mask=None):
        if image is not None: self.image = image
        if image_id is not None: self.image_id = image_id
        if sp is not None: self.spatial = sp
        if Human_augmented is not None: self.H_boxes = Human_augmented
        if Object_augmented is not None: self.O_boxes = Object_augmented
        if action_HO is not None:
            # self.gt_class_HO = action_HO
            # obtain object class
            self.gt_obj = tf.matmul(action_HO, self.obj_to_HO_matrix_orig, transpose_b=True)
            self.gt_class_HO = tf.matmul(action_HO, self.zs_convert_matrix_base)

        self.H_num = num_pos
        self.reset_classes()

    def region_classification_ho(self, fc7_verbs, is_training, initializer, name, nameprefix=''):
        # if not self.model_name.startswith('iCAN_R_') and not self.model_name.__contains__('_orig_'):
        #     return None
        with tf.variable_scope(name) as scope:
            if self.model_name.__contains__('VERB'):
                cls_score_verbs = slim.fully_connected(fc7_verbs, self.verb_num_classes,
                                                       weights_initializer=initializer,
                                                       trainable=is_training,
                                                       reuse=tf.AUTO_REUSE,
                                                       activation_fn=None, scope='cls_score_verb_hoi')
                pass
            else:
                cls_score_verbs = slim.fully_connected(fc7_verbs, self.sum_num_classes,
                                                       weights_initializer=initializer,
                                                       trainable=is_training,
                                                       reuse=tf.AUTO_REUSE,
                                                       activation_fn=None, scope='cls_score_verbs')
            cls_prob_verbs = tf.nn.sigmoid(cls_score_verbs, name='cls_prob_verbs')
            with tf.device("/cpu:0"):
                cls_score_verbs = tf.Print(cls_score_verbs,
                                           [tf.reduce_max(cls_prob_verbs, axis=-1), cls_prob_verbs],
                                           first_n=00000, summarize=100, message='score hoi:')

            self.predictions[nameprefix + "cls_score_verbs"] = cls_score_verbs
            if self.model_name.__contains__('VERB'):
                self.predictions[nameprefix + "cls_prob_verbs_VERB"] = cls_prob_verbs
                self.predictions[nameprefix + "cls_prob_verbs"] = tf.matmul(cls_prob_verbs, self.verb_to_HO_matrix)
            else:
                self.predictions[nameprefix + "cls_prob_verbs"] = cls_prob_verbs

            if self.model_name.__contains__("VCOCO"):
                if self.model_name.__contains__('_CL_'):
                    assert self.num_classes == 222
                    print(cls_score_verbs, '=============================================')
                if self.model_name.__contains__("R_V"):
                    self.predictions[nameprefix + "cls_prob_HO"] = cls_prob_verbs if nameprefix == '' else 0
                else:
                    self.predictions[nameprefix + "cls_prob_HO"] = self.predictions[
                                                                       "cls_prob_sp"] * cls_prob_verbs if nameprefix == '' else 0
        return cls_prob_verbs

    def add_loss(self):
        import math
        with tf.variable_scope('LOSS') as scope:

            num_stop = self.get_num_stop()
            if self.model_name.__contains__('_VCOCO'):
                label_H = self.gt_class_H
                label_HO = self.gt_class_HO
                label_sp = self.gt_class_sp
                if self.model_name.__contains__('_CL'):
                    label_H = self.gt_compose
                    label_HO = self.gt_compose
                    label_sp = self.gt_compose
            else:
                label_H = self.gt_class_HO[:num_stop]
                # label_HO = self.gt_class_HO_for_verbs
                label_HO = self.gt_class_HO[:num_stop]
                label_sp = self.gt_class_HO
                label_hoi = tf.concat([self.gt_class_HO,
                                              tf.zeros([tf.shape(self.gt_class_HO)[0], self.sum_num_classes - self.base_classes])], axis=-1)

            if "cls_score_sp" in self.predictions:
                cls_score_sp = self.predictions["cls_score_sp"]

                with tf.device("/cpu:0"): cls_score_sp = tf.Print(cls_score_sp, [cls_score_sp, label_sp], 'cls_score_sp', first_n=1, summarize=100)
                sp_cross_entropy = self.filter_loss(cls_score_sp, label_sp)
                with tf.device("/cpu:0"): sp_cross_entropy = tf.Print(sp_cross_entropy, [sp_cross_entropy, cls_score_sp, label_sp], 'cls_score_sp', first_n=1,
                                        summarize=100)
                self.losses['sp_cross_entropy'] = sp_cross_entropy

            tmp_label_HO = label_hoi[:num_stop]
            cls_score_verbs = self.predictions["cls_score_verbs"][:tf.shape(tmp_label_HO)[0], :]
            print('debug gt_class_HO_for_D_verbs:', tmp_label_HO, cls_score_verbs)
            if self.model_name.__contains__('VERB'):
                # we direct predict verb rather than HOI.
                # convert HOI label to verb label
                tmp_label_HO = tf.matmul(tmp_label_HO, self.verb_to_HO_matrix, transpose_b=True)

            # tmp_label_HO = tf.Print(tmp_label_HO, [tf.shape(tmp_label_HO), tf.shape(cls_score_verbs)],'sdfsdfsdf')
            print('=======', tmp_label_HO, cls_score_verbs)
            if self.model_name.__contains__('batch') and self.model_name.__contains__('semi'):
                semi_filter = tf.reduce_sum(self.H_boxes[:tf.shape(cls_score_verbs)[0], 1:], axis=-1)

                semi_filter = tf.cast(semi_filter, tf.bool)

                tmp_label_HO = tf.boolean_mask(tmp_label_HO, semi_filter, axis=0)
                cls_score_verbs = tf.boolean_mask(cls_score_verbs, semi_filter, axis=0)

            tmp_hoi_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tmp_label_HO, logits=cls_score_verbs)

            if self.model_name.__contains__('VCOCO_1'):
                tmp_hoi_loss = tf.multiply(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tmp_label_HO, logits=cls_score_verbs),
                    self.Mask_HO)

            verbs_cross_entropy = tf.reduce_mean(tmp_hoi_loss)
            self.losses['verbs_cross_entropy'] = verbs_cross_entropy

            lamb = self.get_lamb_1()
            if "cls_score_sp" not in self.predictions:
                sp_cross_entropy = 0
                self.losses['sp_cross_entropy'] = 0
            loss = sp_cross_entropy + verbs_cross_entropy * lamb

            # else:
            #     loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy
            # verb loss
            temp = self.add_verb_loss(num_stop)
            loss += temp

            # interactiveness
            interactiveness_loss = 0
            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)
        print(self.losses)
        print(self.predictions)
        return loss

