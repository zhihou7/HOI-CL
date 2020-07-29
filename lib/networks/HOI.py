# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops

from ult.config import cfg

import numpy as np
import ipdb
import math

import os
# print(os.environ['DATASET'])
if 'DATASET' not in os.environ or os.environ['DATASET'] == 'HICO':
    from networks.ResNet50_HICO import ResNet50, resnet_arg_scope
    parent_model = ResNet50
elif os.environ['DATASET'] == 'HICO_res101':
    from networks.ResNet101_HICO import ResNet101, resnet_arg_scope
    parent_model = ResNet101
else:
    from networks.ResNet50_VCOCO import ResNet50, resnet_arg_scope
    parent_model = ResNet50

class HOI(parent_model):
    def __init__(self, model_name='VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101'):
        super(HOI, self).__init__(model_name)
        import pickle
        self.update_ops = []

    def res5_ho(self, pool5_HO, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            if self.model_name.startswith('VCL'):
                if self.model_name.__contains__('unique_weights'):
                    print("unique_weights")
                    st = -3
                    reuse = tf.AUTO_REUSE
                    if name != 'res5':
                        reuse = True
                else:
                    st = -2
                    reuse = tf.AUTO_REUSE
                fc7_HO, _ = resnet_v1.resnet_v1(pool5_HO,
                                                self.blocks[st:st+1],
                                                global_pool=False,
                                                include_root_block=False,
                                                reuse=reuse,
                                                scope=self.scope)
            else:
                fc7_HO = None
        return fc7_HO

    def head_to_tail_ho(self, fc7_O, fc7_verbs, fc7_O_raw, fc7_verbs_raw, is_training, name):
        if name == 'fc_HO':
            nameprefix = ''  # TODO should improve
        else:
            nameprefix = name
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            if self.model_name.startswith('VCL'):
                print('others concat')
                concat_verbs = tf.concat([fc7_verbs, fc7_O], 1)  # TODO fix
                print(concat_verbs)
                concat_verbs = slim.fully_connected(concat_verbs, self.num_fc, reuse=tf.AUTO_REUSE, scope=nameprefix+'Concat_verbs')
                concat_verbs = slim.dropout(concat_verbs, keep_prob=0.5, is_training=is_training,
                                            scope=nameprefix+'dropout6_verbs')
                fc9_verbs = slim.fully_connected(concat_verbs, self.num_fc, reuse=tf.AUTO_REUSE, scope=nameprefix+'fc7_verbs')
                fc9_verbs = slim.dropout(fc9_verbs, keep_prob=0.5, is_training=is_training, scope=nameprefix+'dropout7_verbs')
            else:
                fc9_verbs = None
        return fc9_verbs

    def head_to_tail_sp(self, fc7_H, fc7_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            Concat_SHsp   = tf.concat([fc7_H, sp], 1)
            Concat_SHsp   = slim.fully_connected(Concat_SHsp, self.num_fc, reuse=tf.AUTO_REUSE, scope='Concat_SHsp')
            Concat_SHsp   = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp      = slim.fully_connected(Concat_SHsp, self.num_fc, reuse=tf.AUTO_REUSE, scope='fc7_SHsp')
            fc7_SHsp      = slim.dropout(fc7_SHsp,  keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

        return fc7_SHsp

    def region_classification_sp(self, fc7_SHsp, is_training, initializer, name):
        with tf.variable_scope(name) as scope:

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes,
                                                weights_initializer=initializer,
                                                trainable=is_training,
                                                reuse=tf.AUTO_REUSE,
                                                activation_fn=None, scope='cls_score_sp')
            cls_prob_sp = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp')
            tf.reshape(cls_prob_sp, [-1, self.num_classes])

            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"] = cls_prob_sp

        return cls_prob_sp


    def region_classification_ho(self, fc7_verbs, is_training, initializer, name, nameprefix = ''):
        # if not self.model_name.startswith('VCL_') and not self.model_name.__contains__('_orig_'):
        #     return None
        with tf.variable_scope(name) as scope:

            cls_score_verbs = slim.fully_connected(fc7_verbs, self.num_classes,
                                                   weights_initializer=initializer,
                                                   trainable=is_training,
                                                   reuse=tf.AUTO_REUSE,
                                                   activation_fn=None, scope='cls_score_verbs')
            cls_prob_verbs = tf.nn.sigmoid(cls_score_verbs, name='cls_prob_verbs')
            self.predictions[nameprefix+"cls_score_verbs"] = cls_score_verbs
            self.predictions[nameprefix+"cls_prob_verbs"] = cls_prob_verbs

            if self.model_name.__contains__("VCOCO"):
                # if self.model_name.__contains__('_CL_'):
                #     assert self.num_classes == 222
                #     print(cls_score_verbs, '=============================================')
                if self.model_name.__contains__("VCL_V"):
                    self.predictions[nameprefix + "cls_prob_HO"] = cls_prob_verbs if nameprefix == '' else 0
                else:
                    self.predictions[nameprefix+"cls_prob_HO"] = self.predictions["cls_prob_sp"] * cls_prob_verbs if nameprefix =='' else 0
        return cls_prob_verbs

    def get_compose_boxes(self, h_boxes, o_boxes):
        with tf.control_dependencies([tf.assert_equal(h_boxes[:, 0], o_boxes[:, 0],
                                                                data=[h_boxes[:, 0], o_boxes[:, 0]])]):
            cboxes1 = tf.minimum(tf.slice(h_boxes, [0, 0], [-1, 3]),
                                 tf.slice(o_boxes, [0, 0], [-1, 3]))
            cboxes2 = tf.maximum(tf.slice(h_boxes, [0, 3], [-1, 2]),
                                 tf.slice(o_boxes, [0, 3], [-1, 2]))
            cboxes = tf.concat(values=[cboxes1, cboxes2], axis=1)
            return cboxes

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        num_stop = tf.cast(self.get_num_stop(), tf.int32)
        # ResNet Backbone
        head = self.image_to_head(is_training)
        sp = self.sp_to_head()
        cboxes = self.get_compose_boxes(self.H_boxes[:num_stop] if self.model_name.__contains__('VCOCO') else self.H_boxes, self.O_boxes)
        pool5_O = self.crop_pool_layer(head, self.O_boxes, 'Crop_O')
        pool5_H = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')
        cboxes = cboxes[:num_stop]

        pool5_HO = self.extract_pool5_HO(head, cboxes, is_training, pool5_O, None, name='ho_')
        # further resnet feature
        fc7_H_raw, fc7_O_raw = self.res5(pool5_H, pool5_O, None, is_training, 'res5')
        fc7_H = tf.reduce_mean(fc7_H_raw, axis=[1, 2])
        fc7_O = tf.reduce_mean(fc7_O_raw, axis=[1, 2])
        fc7_H_pos = fc7_H[:num_stop]
        fc7_O_pos = fc7_O[:num_stop]
        fc7_HO_raw = self.res5_ho(pool5_HO, is_training, 'res5')

        fc7_HO = None if fc7_HO_raw is None else tf.reduce_mean(fc7_HO_raw, axis=[1, 2])

        if not is_training:
            # add visualization for test
            self.add_visual_for_test(fc7_HO_raw, fc7_H_raw, fc7_O_raw, head, is_training, pool5_O)

        fc7_verbs_raw = fc7_HO_raw
        fc7_verbs = fc7_HO

        self.score_summaries.update({'orth_HO': fc7_HO,
                                     'orth_H': fc7_H, 'orth_O': fc7_O})
        if not self.model_name.startswith('VCL_') or self.model_name.__contains__('_orig_'):
            print('ICAN original code')
            # Phi
            head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')

            # g
            head_g = slim.conv2d(head, 512, [1, 1], scope='head_g')

            Att_H = self.attention_pool_layer_H(head_phi, fc7_H, is_training, 'Att_H')
            Att_H = self.attention_norm_H(Att_H, 'Norm_Att_H')
            att_head_H = tf.multiply(head_g, Att_H)

            Att_O = self.attention_pool_layer_O(head_phi, fc7_O_pos, is_training, 'Att_O')
            Att_O = self.attention_norm_O(Att_O, 'Norm_Att_O')
            att_head_O = tf.multiply(head_g, Att_O)

            pool5_SH = self.bottleneck(att_head_H, is_training, 'bottleneck', False)
            pool5_SO = self.bottleneck(att_head_O, is_training, 'bottleneck', True)
            fc7_SH, fc7_SO, fc7_SHsp = self.head_to_tail(fc7_H, fc7_O_pos, pool5_SH, pool5_SO, sp, is_training, 'fc_HO')
            cls_prob_H, cls_prob_O, cls_prob_sp = self.region_classification(fc7_SH, fc7_SO, fc7_SHsp, is_training,
                                                                             initializer, 'classification')
        elif not self.model_name.startswith('VCL_V_'):
            print('sp', sp)
            fc7_SHsp = self.head_to_tail_sp(fc7_H, fc7_O, sp, is_training, 'fc_HO')
            cls_prob_sp = self.region_classification_sp(fc7_SHsp, is_training, initializer, 'classification')
            print("sp:", fc7_SHsp)
        else:
            fc7_SHsp = self.head_to_tail_sp(fc7_H, fc7_O, sp, is_training, 'fc_HO')
            cls_prob_sp = self.region_classification_sp(fc7_SHsp, is_training, initializer, 'classification')

        print('verbs')
        if self.model_name.__contains__('VCL_'):
            if not is_training:
                self.test_visualize['fc7_O_feats'] = fc7_O
                self.test_visualize['fc7_verbs_feats'] = fc7_verbs
                self.test_visualize['fc7_H_feats'] = fc7_H_pos

            # This is a simple try to add pose, and This can improve the performance slightly
            if self.model_name.__contains__('_posesp'):
                pose = self.add_pose_pattern('posesp')
                fc7_verbs = tf.concat([fc7_verbs, pose], axis=-1)
            elif self.model_name.__contains__('_pose1'):
                pose = self.add_pose1('pose')
                fc7_verbs = tf.concat([fc7_verbs, pose], axis=-1)
            elif self.model_name.__contains__('_pose'):
                pose = self.add_pose('pose')
                fc7_verbs = tf.concat([fc7_verbs, pose], axis=-1)
            if self.model_name.__contains__('_sp1'):
                pattern = self.add_pattern()
                fc7_verbs = tf.concat([fc7_verbs, pattern], axis=-1)
            elif self.model_name.__contains__('_sp'):
                fc7_verbs = tf.concat([fc7_verbs, sp[:num_stop]], axis=-1)

            self.intermediate['fc7_O'] = fc7_O[:num_stop]
            self.intermediate['fc7_verbs'] = fc7_verbs[:num_stop]

            fc7_vo = self.head_to_tail_ho(fc7_O[:num_stop], fc7_verbs[:num_stop], fc7_O_raw, fc7_verbs_raw, is_training, 'fc_HO')
            cls_prob_verbs = self.region_classification_ho(fc7_vo, is_training, initializer, 'classification')
        else:
            cls_prob_verbs = None

        self.score_summaries.update(self.predictions)

    def add_visual_for_test(self, fc7_HO_raw, fc7_H_raw, fc7_O_raw, head, is_training, pool5_O):
        self.test_visualize['fc7_H_raw'] = tf.expand_dims(tf.reduce_mean(fc7_H_raw, axis=-1), axis=-1)
        self.test_visualize['fc7_O_raw'] = tf.expand_dims(tf.reduce_mean(fc7_O_raw, axis=-1), axis=-1)
        if fc7_HO_raw is not None:
            self.test_visualize['fc7_HO_raw'] = tf.expand_dims(tf.reduce_mean(fc7_HO_raw, axis=-1), axis=-1)
        self.test_visualize['fc7_H_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(fc7_H_raw, 0), tf.float32))
        self.test_visualize['fc7_O_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(fc7_O_raw, 0), tf.float32))
        if fc7_HO_raw is not None:
            self.test_visualize['fc7_HO_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(fc7_HO_raw, 0), tf.float32))
        res5_ho_h = self.res5_ho(self.extract_pool5_HO(head, self.H_boxes, is_training, pool5_O, None), is_training,
                                 'h')
        if self.model_name.__contains__('VCL_humans'):
            res5_ho_o = self.crop_pool_layer(head, self.O_boxes, 'Crop_HO_h')
        else:
            res5_ho_o = self.res5_ho(self.extract_pool5_HO(head, self.O_boxes, is_training, pool5_O, None), is_training,
                                     'o')
        print("res5_ho_o", res5_ho_o, res5_ho_h)
        if res5_ho_h is not None and res5_ho_o is not None:
            self.test_visualize['res5_ho_H'] = tf.expand_dims(tf.reduce_mean(res5_ho_h, axis=-1), axis=-1)
            self.test_visualize['res5_ho_O'] = tf.expand_dims(tf.reduce_mean(res5_ho_o, axis=-1), axis=-1)
            self.test_visualize['res5_ho_H_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(res5_ho_h, 0), tf.float32))
            self.test_visualize['res5_ho_O_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(res5_ho_o, 0), tf.float32))

    def add_pose(self, name):
        with tf.variable_scope(name) as scope:
            conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:][:self.get_num_stop()], 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_pose_map')
            pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
            conv2_pose_map = slim.conv2d(pool1_pose_map, 16, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_pose_map')
            pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)
        return pool2_flat_pose_map

    def add_pose1(self, name):
        with tf.variable_scope(name) as scope:
            conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:][:self.get_num_stop()], 64, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_pose_map')
            pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
            conv2_pose_map = slim.conv2d(pool1_pose_map, 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_pose_map')
            pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)
        return pool2_flat_pose_map

    def add_pose_pattern(self, name = "pose_sp"):
        with tf.variable_scope(name) as scope:
            conv1_pose_map = slim.conv2d(self.spatial[:self.get_num_stop()], 64, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_sp_pose_map')
            pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_sp_pose_map')
            conv2_pose_map = slim.conv2d(pool1_pose_map, 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_sp_pose_map')
            pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_sp_pose_map')
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)
        return pool2_flat_pose_map

    def add_pattern(self, name = 'pattern'):
        with tf.variable_scope(name) as scope:
            with tf.variable_scope(self.scope, self.scope):
                conv1_sp = slim.conv2d(self.spatial[:, :, :, 0:2][:self.get_num_stop()], 64, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_sp')
                pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
                conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_sp')
                pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
                pool2_flat_sp = slim.flatten(pool2_sp)
        return pool2_flat_sp

    def get_num_stop(self):
        """
        following iCAN, spatial pattern include all negative samples. verb-object branch is for positive samples
        self.H_num is the partition for positive sample and negative samples.
        :return:
        """
        num_stop = tf.shape(self.H_boxes)[0]  # for selecting the positive items
        if self.model_name.__contains__('_new') \
                or not self.model_name.startswith('VCL_'):
            print('new Add H_num constrains')
            num_stop = self.H_num
        elif self.model_name.__contains__('_x5new'):  # contain some negative items
            # I use this strategy cause I found by accident that including
            # some negative samples in the positive samples can improve the performance a bit (abount 0.2%).
            # TODO I think it might have a better solution.
            #  No-Frills Human-Object Interaction Detection provides some support
            H_num_tmp = tf.cast(self.H_num, tf.int32)
            num_stop = tf.cast(num_stop, tf.int32)
            num_stop = H_num_tmp + tf.cast((num_stop - H_num_tmp) // 8, tf.int32)
        else:
            num_stop = self.H_num
        return num_stop

    def get_compose_num_stop(self):
        num_stop = self.get_num_stop()
        return num_stop

    def extract_pool5_HO(self, head, cboxes, is_training, pool5_O, head_mask = None, name=''):
        if self.model_name.__contains__('_union'):
            pool5_HO = self.crop_pool_layer(head, cboxes, name + 'Crop_HO')
            self.test_visualize["pool5_HO"] = tf.expand_dims(tf.reduce_mean(pool5_HO, axis=-1), axis=-1)
        elif self.model_name.__contains__('_humans'):
            print("humans")
            pool5_HO = self.crop_pool_layer(head, self.H_boxes[:self.get_num_stop()],name +  'Crop_HO_h')
            self.test_visualize["pool5_HO"] = tf.expand_dims(tf.reduce_mean(pool5_HO, axis=-1), axis=-1)
        else:
            # pool5_HO = self.crop_pool_layer(head, cboxes, 'Crop_HO')
            pool5_HO = None
            print("{} doesn\'t support pool5_HO".format(self.model_name))
        return pool5_HO

    def add_loss(self):
        import math
        with tf.variable_scope('LOSS') as scope:
            num_stop = self.get_num_stop()
            if self.model_name.__contains__('_VCOCO'):
                label_H = self.gt_class_H
                label_HO = self.gt_class_HO
                label_sp = self.gt_class_sp
            else:
                label_H = self.gt_class_HO[:num_stop]
                # label_HO = self.gt_class_HO_for_verbs
                label_HO = self.gt_class_HO[:num_stop]
                label_sp = self.gt_class_HO
            if "cls_score_H" in self.predictions:
                cls_score_H = self.predictions["cls_score_H"]
                if self.model_name.__contains__('_rew'):
                    cls_score_H = tf.multiply(cls_score_H, self.HO_weight)
                H_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label_H,
                                                            logits=cls_score_H[:num_stop,  :]))
                self.losses['H_cross_entropy'] = H_cross_entropy
            if "cls_score_O" in self.predictions:
                cls_score_O = self.predictions["cls_score_O"]
                if self.model_name.__contains__('_rew'):
                    cls_score_O = tf.multiply(cls_score_O, self.HO_weight)
                O_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO,
                                                            logits=cls_score_O[:num_stop,  :]))
                self.losses['O_cross_entropy'] = O_cross_entropy
            if "cls_score_sp" in self.predictions:
                cls_score_sp = self.predictions["cls_score_sp"]
                sp_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label_sp, logits=cls_score_sp))

                self.losses['sp_cross_entropy'] = sp_cross_entropy

            if self.model_name.startswith('VCL_V_'):
                cls_score_verbs = self.predictions["cls_score_verbs"]

                verbs_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO[:num_stop, :], logits=cls_score_verbs[:num_stop, :]))
                self.losses['verbs_cross_entropy'] = verbs_cross_entropy

                loss = verbs_cross_entropy
            elif self.model_name.startswith('VCL_'):

                tmp_label_HO = self.gt_class_HO[:num_stop]
                cls_score_verbs = self.predictions["cls_score_verbs"][:tf.shape(self.gt_class_HO[:num_stop])[0], :]
                if self.model_name.__contains__('_rew'):
                    cls_score_verbs = tf.multiply(cls_score_verbs, self.HO_weight)

                print('=======', tmp_label_HO, cls_score_verbs)
                tmp_verb_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tmp_label_HO, logits=cls_score_verbs)

                verbs_cross_entropy = tf.reduce_mean(tmp_verb_loss)
                self.losses['verbs_cross_entropy'] = verbs_cross_entropy

                lamb = 1
                if self.model_name.__contains__('_l05_'):
                    lamb = 0.5
                elif self.model_name.__contains__('_l2_'):
                    lamb = 2
                elif self.model_name.__contains__('_l0_'):
                    lamb = 0
                elif self.model_name.__contains__('_l1_'):
                    lamb = 1
                elif self.model_name.__contains__('_l15_'):
                    lamb = 1.5
                elif self.model_name.__contains__('_l25_'):
                    lamb = 2.5
                elif self.model_name.__contains__('_l3_'):
                    lamb = 3
                elif self.model_name.__contains__('_l4_'):
                    lamb = 4
                if "cls_score_sp" not in self.predictions:
                    sp_cross_entropy = 0
                    self.losses['sp_cross_entropy'] = 0
                loss = sp_cross_entropy + verbs_cross_entropy * lamb
                if self.model_name.__contains__('_orig_'):
                    loss = loss + O_cross_entropy + H_cross_entropy
                    print('Add all loss')

            else:
                loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy

            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)
        print(self.losses)
        print(self.predictions)
        return loss

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = self.get_feed_dict(blobs)

        loss, _ = sess.run([self.losses['total_loss'],
                            train_op],
                           feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = self.get_feed_dict(blobs)

        loss, summary, _ = sess.run([self.losses['total_loss'],
                                     self.summary_op,
                                     train_op],
                                    feed_dict=feed_dict)
        return loss, summary


    def obtain_all_preds(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'], self.H_num: blobs['H_num']}
        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:
            cls_prob_HO, pH, pO, pSp, pVerbs = sess.run([self.predictions["cls_prob_HO"], self.predictions["cls_prob_H"],
                                    self.predictions["cls_prob_O"], self.predictions["cls_prob_sp"],
                                    self.predictions["cls_prob_verbs"]], feed_dict=feed_dict)

        except InvalidArgumentError as e:
            cls_prob_HO, pH, pO, pSp, pVerbs = sess.run(
                [self.predictions["cls_prob_HO_original"], self.predictions["cls_prob_H"],
                 self.predictions["cls_prob_O"], self.predictions["cls_prob_sp"],
                 self.predictions["cls_prob_H"]], feed_dict=feed_dict)
            print("InvalidArgumentError")

        return cls_prob_HO, pH, pO, pSp, pVerbs
