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

from networks.Fabricator import Fabricator
from ult.config import cfg

import numpy as np
import math

import os
# print(os.environ['DATASET'])
if 'DATASET' not in os.environ or os.environ['DATASET'] == 'HICO':
    from networks.ResNet50_HICO import ResNet50, resnet_arg_scope
    parent_model = ResNet50
elif os.environ['DATASET'] == 'HICO_res101':
    from networks.ResNet101_HICO import ResNet101, resnet_arg_scope
    parent_model = ResNet101
elif os.environ['DATASET'] == 'HICO_res101_icl':
    from networks.ResNet101_HICO_zs import ResNet101, resnet_arg_scope

    parent_model = ResNet101
elif os.environ['DATASET'] == 'HICO_icl':
    from networks.ResNet50_HICO_zs import ResNet50, resnet_arg_scope

    parent_model = ResNet50
elif os.environ['DATASET'] == 'VCOCO1':
    from networks.ResNet50_VCOCO_HOI import ResNet50, resnet_arg_scope
    parent_model = ResNet50
else:
    from networks.ResNet50_VCOCO import ResNet50, resnet_arg_scope
    parent_model = ResNet50

class HOI(parent_model):
    def __init__(self, model_name='VCL_union_multi_ml5_def1_l2_rew2_aug5_3_x5new_res101'):
        super(HOI, self).__init__(model_name)
        self.pos1_idx = None
        import pickle
        self.update_ops = []
        self.feature_gen = Fabricator(self)
        self.gt_class_HO_for_G_verbs = None
        self.gt_class_HO_for_D_verbs = None
        self.losses['fake_D_total_loss'] = 0
        self.losses['fake_G_total_loss'] = 0
        self.losses['fake_total_loss'] = 0
        if self.model_name.__contains__('affordance'):
            self.affordance_count = tf.get_variable("affordance_count", shape=[self.verb_num_classes, self.obj_num_classes],
                                                    initializer=tf.zeros_initializer)
            if self.model_name.__contains__('OFFLINE'): # For ablation study, you can ignore this
                if self.model_name.__contains__('VCOCO'):
                    init_v = np.load('/project/ZHIHOU//afford/iCAN_R_union_multi_ml5_l05_t5_VERB_def2_aug5_3_new_VCOCO_test_CL_21_affordance_9_HOI_iter_160000.ckpt.npy')
                else:
                    if self.model_name.__contains__('inito'):
                        init_v = np.load('/project/ZHIHOU//afford/iCAN_R_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_inito_OFFLINE1_AF713_r_1_9ATL_2.npy')
                    elif self.model_name.__contains__('OFFLINE1'):
                        init_v = np.load('/project/ZHIHOU/afford/iCAN_R_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_9ATL_2.npy')
                    else:
                        init_v = np.load("/project/ZHIHOU//afford/iCAN_R_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_AF713_OFFLINE_9ATL_2.npy")
                init_v = init_v.reshape([self.verb_num_classes, self.obj_num_classes])
                self.affordance_stat = tf.get_variable("affordance_stat", shape=[self.verb_num_classes, self.obj_num_classes], dtype=tf.float32,
                                                       trainable=True,
                                                       initializer=tf.constant_initializer(init_v),
                                                       regularizer=regularizers.l2_regularizer(0.001))
            elif self.model_name.__contains__('inito'):  # For debug, you can ignore this
                init_v = np.load('/project/ZHIHOU/afford/iCAN_R_union_batch_large2_ml5_def1_vloss2_VERB_l2_aug5_3_x5new_res101_affordance_9ATL_2.npy')
                init_v = init_v.reshape([self.verb_num_classes, self.obj_num_classes])
                self.affordance_stat = tf.get_variable("affordance_stat", shape=[self.verb_num_classes, self.obj_num_classes], dtype=tf.float32,
                                                       trainable=True,
                                                       initializer=tf.constant_initializer(init_v),
                                                       regularizer=regularizers.l2_regularizer(0.001))
            elif self.model_name.__contains__('init'):
                self.affordance_stat = tf.get_variable("affordance_stat", shape=[self.verb_num_classes, self.obj_num_classes], dtype=tf.float32,
                                                       trainable=True,
                                                       initializer=tf.constant_initializer(np.matmul(self.verb_to_HO_matrix_np, self.obj_to_HO_matrix_np.transpose())),
                                                       regularizer=regularizers.l2_regularizer(0.001))
            else:
                self.affordance_stat = tf.get_variable("affordance_stat", shape=[self.verb_num_classes, self.obj_num_classes], dtype=tf.float32, trainable=True,
                                                       initializer=tf.zeros_initializer, regularizer=regularizers.l2_regularizer(0.001))

    def set_gt_class_HO_for_G_verbs(self, gt_class_HO_for_G_verbs):
        self.gt_class_HO_for_G_verbs = gt_class_HO_for_G_verbs

    def set_gt_class_HO_for_D_verbs(self, gt_class_HO_for_D_verbs):
        self.gt_class_HO_for_D_verbs = gt_class_HO_for_D_verbs

    def set_add_ph(self, pos1_idx=None):
        self.pos1_idx = pos1_idx


    def res5_ho(self, pool5_HO, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
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
        return fc7_HO

    def head_to_tail_ho(self, fc7_O, fc7_verbs, fc7_O_raw, fc7_verbs_raw, is_training, name, gt_verb_class=None, gt_obj_class=None):
        if name == 'fc_HO':
            nameprefix = ''  # TODO should improve
        else:
            nameprefix = name
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            print('others concat')
            concat_hoi = tf.concat([fc7_verbs, fc7_O], 1)  # TODO fix
            print(concat_hoi)
            concat_hoi = slim.fully_connected(concat_hoi, self.num_fc, reuse=tf.AUTO_REUSE, scope=nameprefix+'Concat_verbs')
            concat_hoi = slim.dropout(concat_hoi, keep_prob=0.5, is_training=is_training,
                                        scope=nameprefix+'dropout6_verbs')
            fc9_hoi = slim.fully_connected(concat_hoi, self.num_fc, reuse=tf.AUTO_REUSE, scope=nameprefix+'fc7_verbs')
            fc9_hoi = slim.dropout(fc9_hoi, keep_prob=0.5, is_training=is_training, scope=nameprefix+'dropout7_verbs')

        return fc9_hoi

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
            if self.model_name.__contains__('VERB'):
                cls_score_hoi = slim.fully_connected(fc7_verbs, self.verb_num_classes,
                                                       weights_initializer=initializer,
                                                       trainable=is_training,
                                                       reuse=tf.AUTO_REUSE,
                                                       activation_fn=None, scope='cls_score_verb_hoi')
                pass
            else:
                cls_score_hoi = slim.fully_connected(fc7_verbs, self.num_classes,
                                                       weights_initializer=initializer,
                                                       trainable=is_training,
                                                       reuse=tf.AUTO_REUSE,
                                                       activation_fn=None, scope='cls_score_verbs')
            cls_prob_hoi = tf.nn.sigmoid(cls_score_hoi, name='cls_prob_verbs')
            self.predictions[nameprefix+"cls_score_hoi"] = cls_score_hoi
            if self.model_name.__contains__('VERB'):
                self.predictions[nameprefix + "cls_prob_verbs_VERB"] = cls_prob_hoi
                self.predictions[nameprefix + "cls_prob_hoi"] = tf.matmul(cls_prob_hoi, self.verb_to_HO_matrix)
            else:
                self.predictions[nameprefix+"cls_prob_hoi"] = cls_prob_hoi

            if self.model_name.__contains__("VCOCO"):
                # if self.model_name.__contains__('_CL_'):
                #     assert self.num_classes == 222
                #     print(cls_score_hoi, '=============================================')
                if self.model_name.__contains__("VCL_V"):
                    self.predictions[nameprefix + "cls_prob_HO"] = cls_prob_hoi if nameprefix == '' else 0
                elif self.model_name.__contains__('VERB'):
                    self.predictions[nameprefix + "cls_prob_HO"] = self.predictions[
                                                                       "cls_prob_sp"] * tf.matmul(cls_prob_hoi, self.verb_to_HO_matrix) if nameprefix == '' else 0

                else:
                    self.predictions[nameprefix+"cls_prob_HO"] = self.predictions["cls_prob_sp"] * cls_prob_hoi if nameprefix =='' else 0
        return cls_prob_hoi

    def sigmoid_hoi(self, fc7_hoi, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            cls_score_hoi = slim.fully_connected(fc7_hoi, 1,
                                                 weights_initializer=initializer,
                                                 trainable=is_training,
                                                 reuse=tf.AUTO_REUSE,
                                                 activation_fn=None, scope='inte_cls_score_hoi')
            cls_prob_hoi = tf.nn.sigmoid(cls_score_hoi, name='inte_cls_prob_hoi')

            self.predictions["inte_cls_score_hoi"] = cls_score_hoi
            self.predictions["inte_cls_prob_hoi"] = cls_prob_hoi

        return cls_prob_hoi

    def get_compose_boxes(self, h_boxes, o_boxes):
        with tf.control_dependencies([tf.assert_equal(h_boxes[:, 0], o_boxes[:, 0],
                                                                data=[h_boxes[:, 0], o_boxes[:, 0]])]):
            cboxes1 = tf.minimum(tf.slice(h_boxes, [0, 0], [-1, 3]),
                                 tf.slice(o_boxes, [0, 0], [-1, 3]))
            cboxes2 = tf.maximum(tf.slice(h_boxes, [0, 3], [-1, 2]),
                                 tf.slice(o_boxes, [0, 3], [-1, 2]))
            cboxes = tf.concat(values=[cboxes1, cboxes2], axis=1)
            return cboxes

    def verbs_loss(self, fc7_verbs, is_training, initializer, label='', ):
        with tf.variable_scope('verbs_loss', reuse=tf.AUTO_REUSE):
            cls_verbs = fc7_verbs
            # cls_verbs = slim.fully_connected(cls_verbs, self.num_fc, scope='fc8_cls_verbs')
            # cls_verbs = slim.dropout(cls_verbs, keep_prob=0.5, is_training=is_training, scope='dropout8_cls_verbs')
            # fc9_verbs = slim.fully_connected(cls_verbs, self.num_fc, scope='fc9_cls_verbs')
            # verbs_cls = slim.dropout(fc9_verbs, keep_prob=0.5, is_training=is_training, scope='dropout9_cls_verbs')
            verbs_cls_score = slim.fully_connected(cls_verbs, self.verb_num_classes,
                                                   weights_initializer=initializer,
                                                   trainable=is_training,
                                                   reuse=tf.AUTO_REUSE,
                                                   activation_fn=None, scope='verbs_cls_score')
            verb_cls_prob = tf.nn.sigmoid(verbs_cls_score, name='verb_cls_prob')
            tf.reshape(verb_cls_prob, [-1, self.verb_num_classes])

            self.predictions["verb_cls_score" + label] = verbs_cls_score
            self.predictions["verb_cls_prob" + label] = verb_cls_prob


    # We do not use this.
    def objects_loss(self, input_feature, is_training, initializer, name='objects_loss', label='', is_stop_grads=False):
        with tf.variable_scope(name):
            print('objects_loss:', self.model_name)
            if is_stop_grads:
                input_feature = tf.stop_gradient(input_feature)
            # cls_verbs = slim.fully_connected(cls_verbs, self.num_fc, scope='fc8_cls_verbs')
            # cls_verbs = slim.dropout(cls_verbs, keep_prob=0.5, is_training=is_training, scope='dropout8_cls_verbs')
            # fc9_verbs = slim.fully_connected(cls_verbs, self.num_fc, scope='fc9_cls_verbs')
            # verbs_cls = slim.dropout(fc9_verbs, keep_prob=0.5, is_training=is_training, scope='dropout9_cls_verbs')

            obj_cls_score = slim.fully_connected(input_feature, self.obj_num_classes,
                                                 weights_initializer=initializer,
                                                 trainable=is_training,
                                                 reuse=tf.AUTO_REUSE,
                                                 activation_fn=None, scope='obj_cls_score')
            obj_cls_prob = tf.nn.sigmoid(obj_cls_score, name='obj_cls_prob')
            tf.reshape(obj_cls_prob, [-1, self.obj_num_classes])

            self.predictions["obj_cls_score" + label] = obj_cls_score
            self.predictions["obj_cls_prob" + label] = obj_cls_prob


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
        if self.model_name.__contains__('_orig_'):
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
        elif not self.model_name.startswith('_V_'):
            print('sp', sp)
            fc7_SHsp = self.head_to_tail_sp(fc7_H, fc7_O, sp, is_training, 'fc_HO')
            cls_prob_sp = self.region_classification_sp(fc7_SHsp, is_training, initializer, 'classification')
            print("sp:", fc7_SHsp)
        else:
            fc7_SHsp = self.head_to_tail_sp(fc7_H, fc7_O, sp, is_training, 'fc_HO')
            cls_prob_sp = self.region_classification_sp(fc7_SHsp, is_training, initializer, 'classification')
        self.additional_loss(fc7_O, fc7_H_pos, fc7_verbs, fc7_verbs_raw, initializer, is_training)

        print('verbs')
        if not is_training:
            self.test_visualize['fc7_O_feats'] = fc7_O
            self.test_visualize['fc7_verbs_feats'] = fc7_verbs
            self.test_visualize['fc7_H_feats'] = fc7_H_pos

        self.intermediate['fc7_O'] = fc7_O[:num_stop]
        self.intermediate['fc7_verbs'] = fc7_verbs[:num_stop]

        if is_training and self.model_name.__contains__('gan'):
            # if model_name contains gan, we will use fabricator.
            # here, gan do not mean that we use generative adversarial network.
            # We just was planning to use to GAN. But, it is useless.
            # Possibly, it is too difficult to tune the network with gan.
            gt_class = self.gt_class_HO[:num_stop]
            tmp_fc7_O = fc7_O[:num_stop]
            tmp_fc7_verbs = fc7_verbs[:num_stop]
            tmp_O_raw = fc7_O_raw[:num_stop]
            if self.model_name.__contains__('batch') and self.model_name.__contains__('atl'):
                tmp_O_raw = fc7_O[:num_stop]
                tmp_gt_class = gt_class
                # remove object list
                semi_filter = tf.reduce_sum(self.H_boxes[:tf.shape(tmp_fc7_O)[0], 1:], axis=-1)
                semi_filter = tf.cast(semi_filter, tf.bool)

                gt_class = tf.boolean_mask(gt_class, semi_filter, axis=0)
                tmp_fc7_O = tf.boolean_mask(tmp_fc7_O, semi_filter, axis=0)
                tmp_fc7_verbs = tf.boolean_mask(tmp_fc7_verbs, semi_filter, axis=0)

            fc7_O, fc7_verbs = self.feature_gen.fabricate_model(tmp_fc7_O, tmp_O_raw,
                                                                tmp_fc7_verbs, fc7_verbs_raw[:num_stop], initializer, is_training,
                                                                gt_class)

            # if self.model_name.__contains__('laobj'):
            #     # this aims to evaluate the effect of regularizing fabricated object features, we do not use.
            #     all_fc7_O = fc7_O
            #     tmp_class = self.get_hoi_labels()
            #     self.gt_obj_class = tf.cast(
            #         tf.matmul(tmp_class, self.obj_to_HO_matrix, transpose_b=True) > 0,
            #         tf.float32)
            #     self.objects_loss(all_fc7_O, is_training, initializer, 'objects_loss', label='_o')
            #     pass
        else:
            if 'FEATS' in os.environ and self.model_name.__contains__(
                    'gan'):
                # This is only for visualization
                gt_class = self.gt_class_HO if not self.model_name.__contains__(
                    'VCOCO') else self.gt_compose[:num_stop]
                old_fc7_O = fc7_O
                fc7_O, fc7_verbs = self.feature_gen.fabricate_model(fc7_O, None,
                                                              fc7_verbs, fc7_verbs, initializer,
                                                              True,
                                                              gt_class)
                with tf.device("/cpu:0"): fc7_O = tf.Print(fc7_O, [tf.shape(fc7_O), num_stop, tf.shape(self.H_boxes), tf.shape(old_fc7_O), ],
                                 'after gan:', first_n=100, summarize=10000)

                if self.model_name.__contains__('varv'):
                    self.test_visualize['fc7_fake_O_feats'] = fc7_verbs[tf.shape(old_fc7_O)[0]:]
                else:
                    self.test_visualize['fc7_fake_O_feats'] = fc7_O[tf.shape(old_fc7_O)[0]:]
            pass
            fc7_O = fc7_O[:num_stop]
            fc7_verbs = fc7_verbs[:num_stop]

        fc7_vo = self.head_to_tail_ho(fc7_O, fc7_verbs, fc7_O_raw, fc7_verbs_raw, is_training, 'fc_HO')
        cls_prob_verbs = self.region_classification_ho(fc7_vo, is_training, initializer, 'classification')

        if self.model_name.__contains__('_l0_') or self.model_name.__contains__('_scale_'):
            """
            This is for factorized model.
            """
            verb_prob = self.predictions['verb_cls_prob']
            obj_prob = self.predictions["obj_cls_prob_o"]
            print(verb_prob, obj_prob)
            tmp_fc7_O_vectors = tf.cast(
                tf.matmul(obj_prob, self.obj_to_HO_matrix) > 0,
                tf.float32)
            tmp_fc7_verbs_vectors = tf.cast(
                tf.matmul(verb_prob, self.verb_to_HO_matrix) > 0,
                tf.float32)
            if 'cls_prob_verbs' not in self.predictions:
                self.predictions['cls_prob_verbs'] = 0
            if self.model_name.__contains__('_l0_'):
                self.predictions['cls_prob_verbs'] = 0
            self.predictions['cls_prob_verbs'] += (tmp_fc7_O_vectors + tmp_fc7_verbs_vectors)

        self.score_summaries.update(self.predictions)

    def get_hoi_labels(self):
        if self.gt_class_HO_for_D_verbs is not None:
            # we might have changed label in Fabricator
            return self.gt_class_HO_for_D_verbs
        else:
            if self.model_name.__contains__('VCOCO') and self.model_name.__contains__('CL'):
                return self.gt_compose
            return self.gt_class_HO

    def stat_running_affordance(self, cls_prob_hoi, gt_verb_obj):
        # hoi_preds = tf.sigmoid(cls_prob_hoi)  # Nx600
        if self.model_name.__contains__('VERB'):
            verb_preds = cls_prob_hoi
        else:
            verb_preds = tf.matmul(cls_prob_hoi, self.verb_to_HO_matrix, transpose_b=True) / \
                         tf.reduce_sum(self.verb_to_HO_matrix, axis=-1)

        verb_preds = tf.expand_dims(verb_preds, axis=-1)
        verb_preds = tf.tile(verb_preds, [1, 1, 80])
        with tf.device("/cpu:0"): verb_preds = tf.Print(verb_preds,
                                                        [tf.shape(gt_verb_obj),
                                                         tf.shape(verb_preds),
                                                         tf.reduce_sum(self.affordance_count),
                                                         tf.reduce_min(self.affordance_stat),
                                                         tf.reduce_max(self.affordance_stat),
                                                         gt_verb_obj,
                                                         # tf.reduce_min(verb_preds),
                                                         # tf.reduce_max(verb_preds),
                                                         ], first_n=1000, summarize=100,
                                                        message="tmp_affordance v1")
        stat_affordance = tf.multiply(verb_preds, gt_verb_obj)

        new_count = self.affordance_count + tf.reduce_sum(gt_verb_obj, axis=0)
        tmp_afford = tf.reduce_sum(stat_affordance, axis=0) + self.affordance_stat * self.affordance_count
        with tf.device("/cpu:0"): tmp_afford = tf.Print(tmp_afford,
                                                        [tf.shape(gt_verb_obj), tf.reduce_sum(tf.cast(tmp_afford == 0,tf.float32)),
                                                         tf.reduce_sum(tf.cast(new_count == 0, tf.float32)),
                                                         tf.shape(stat_affordance),
                                                         tf.reduce_min(stat_affordance),
                                                         'stat_affordance max:', tf.reduce_max(stat_affordance),
                                                         tf.reduce_max(tf.div_no_nan(tmp_afford,  new_count)),
                                                         tf.reduce_max(verb_preds), tf.reduce_max(gt_verb_obj),
                                                         tf.reduce_max(cls_prob_hoi),
                                                         '==',
                                                         tf.reduce_min(tmp_afford),
                                                         tf.reduce_max(tmp_afford),
                                                         tf.reduce_max(tf.div_no_nan(tmp_afford,  new_count)),
                                                         tf.reduce_max(new_count), tf.reduce_max(self.affordance_count),
                                                         stat_affordance,
                                                         # tf.reduce_min(verb_preds),
                                                         # tf.reduce_max(verb_preds),
                                                         ], first_n=1000, summarize=100,
                                                        message="tmp_affordance v2")
        new_affordance = tf.div_no_nan(tmp_afford,  new_count)
        # self.affordance_stat = tf.assign(self.affordance_stat, new_affordance)
        # self.affordance_count = tf.assign(self.affordance_count, new_count)
        # tf.cast(tf.shape(cls_prob_hoi)[0], tf.bool),
        self.affordance_stat = tf.assign(self.affordance_stat, tf.cond(
            tf.cast(tf.shape(cls_prob_hoi)[0], tf.bool),
            true_fn=lambda: new_affordance,
            false_fn=lambda: self.affordance_stat,
            name=None
        ))
        self.affordance_count = tf.assign(self.affordance_count, tf.cond(
            tf.cast(tf.shape(cls_prob_hoi)[0], tf.bool),
            true_fn=lambda: new_count,
            false_fn=lambda: self.affordance_count,
            name=None
        ))
        return self.affordance_stat, gt_verb_obj


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
        if self.model_name.__contains__('humans'):
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

    def add_pattern(self, name = 'pattern'):
        with tf.variable_scope(name) as scope:
            with tf.variable_scope(self.scope, self.scope):
                conv1_sp = slim.conv2d(self.spatial[:, :, :, 0:2][:self.get_num_stop()], 64, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_sp')
                pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
                conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_sp')
                pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
                pool2_flat_sp = slim.flatten(pool2_sp)
        return pool2_flat_sp

    def additional_loss(self, fc7_O, fc7_H, fc7_verbs, fc7_verbs_raw, initializer, is_training):
        if self.model_name.__contains__('_vloss'):
            self.verbs_loss(fc7_verbs, is_training, initializer)
        if self.model_name.__contains__('_objloss'):
            self.objects_loss(fc7_O, is_training, initializer, 'objects_loss', label='_o')

    def get_num_stop(self):
        """
        following iCAN, spatial pattern include all negative samples. verb-object branch is for positive samples
        self.H_num is the partition for positive sample and negative samples.
        :return:
        """
        if self.model_name.__contains__('batch'):
            # This is for batch style. i.e. there are multiple images in each batch.
            return self.H_num
        num_stop = tf.shape(self.H_boxes)[0]  # for selecting the positive items
        if self.model_name.__contains__('_new'):
            print('new Add H_num constrains')
            num_stop = self.H_num
        elif self.model_name.__contains__('_x5new'):  # contain some negative items
            # I use this strategy cause I found by accident that including
            # some negative samples in the positive samples can improve the performance a bit (abount 0.2%).
            # TODO I think it might have a better solution.
            #  No-Frills Human-Object Interaction Detection provides some support
            #  I think VCL do not depend on this. If someone finds This has important impact on result,
            #  feel happy to contact me.
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
                if self.model_name.__contains__('_CL'):
                    label_H = self.gt_compose
                    label_HO = self.gt_compose
                    label_sp = self.gt_compose
            else:
                label_H = self.gt_class_HO[:num_stop]
                # label_HO = self.gt_class_HO_for_verbs
                label_HO = self.gt_class_HO[:num_stop]
                label_sp = self.gt_class_HO
            if "cls_score_H" in self.predictions:
                cls_score_H = self.predictions["cls_score_H"]
                """
                The re-weighting strategy has an important effect on the performance. 
                This will also improve largely our baseline in both common and zero-shot setting.
                We copy from TIN.
                """
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
                if self.model_name.__contains__('_rew'):
                    cls_score_sp = tf.multiply(cls_score_sp, self.HO_weight)
                elif self.model_name.__contains__('_xrew'):
                    reweights = np.log(1 / (self.num_inst_all / np.sum(self.num_inst_all)))
                    cls_score_sp = tf.multiply(cls_score_sp, reweights)

                print(label_sp, cls_score_sp)
                sp_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label_sp, logits=cls_score_sp))

                self.losses['sp_cross_entropy'] = sp_cross_entropy

            if self.model_name.startswith('_V_'):
                cls_score_hoi = self.predictions["cls_score_hoi"]
                if self.model_name.__contains__('_rew'):
                    cls_score_hoi = tf.multiply(cls_score_hoi, self.HO_weight)
                tmp_label_HO = label_HO
                if self.model_name.__contains__('VERB'):
                    # we direct predict verb rather than HOI.
                    # convert HOI label to verb label
                    tmp_label_HO = tf.matmul(tmp_label_HO, self.verb_to_HO_matrix, transpose_b=True)
                hoi_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tmp_label_HO[:num_stop, :], logits=cls_score_hoi[:num_stop, :]))
                self.losses['hoi_cross_entropy'] = hoi_cross_entropy

                loss = hoi_cross_entropy
            elif self.model_name.__contains__('_fac_'):
                # factorized
                gt_verb_label = self.gt_verb_class[:num_stop, :]
                gt_obj_label = self.gt_obj_class[:num_stop, :]
                # label_verb = tf.matmul()
                cls_score_verbs = self.predictions["cls_score_verbs_f"][:num_stop, :]
                cls_score_objs = self.predictions["cls_score_objs"][:num_stop, :]
                hoi_cross_entropy = self.add_factorized_hoi_loss(cls_score_objs, cls_score_verbs, gt_obj_label,
                                                                 gt_verb_label)

                # result = tf.equal(tf.cast(tmp_verb_prob * gt_verb_label > 0.5, tf.float32),
                #                   tf.cast(gt_verb_label, tf.float32))
                # print('res', result)
                # tmp_hoi_loss = tf.Print(tmp_hoi_loss, [tf.shape(result)], 'HOI acc:')

                self.losses['verbs_cross_entropy'] = hoi_cross_entropy
                # self.losses["pos_hoi_cross_entropy"] = tf.reduce_mean(
                #     tf.reduce_sum(tmp_verb_loss * gt_verb_label, axis=-1) / tf.reduce_sum(gt_verb_label, axis=-1))
                # self.losses["pos_sp_cross_entropy"] = tf.reduce_mean(
                #     tf.reduce_sum(tmp_sp_cross_entropy * label_sp, axis=-1) / tf.reduce_sum(label_sp, axis=-1))

                lamb = self.get_lamb_1()
                if "cls_score_sp" not in self.predictions:
                    sp_cross_entropy = 0
                    self.losses['sp_cross_entropy'] = 0
                loss = sp_cross_entropy + hoi_cross_entropy * lamb
            elif self.model_name.startswith('VCL_') or self.model_name.startswith('FCL_') \
                    or self.model_name.startswith('ATL_'):

                tmp_label_HO = self.get_hoi_labels()[:num_stop]
                if self.model_name.__contains__('VERB'):
                    # we direct predict verb rather than HOI.
                    # convert HOI label to verb label
                    tmp_label_HO = tf.matmul(tmp_label_HO, self.verb_to_HO_matrix, transpose_b=True)


                cls_score_hoi = self.predictions["cls_score_hoi"][:num_stop, :]
                if self.model_name.__contains__('_rew'):
                    cls_score_hoi = tf.multiply(cls_score_hoi, self.HO_weight)
                elif self.model_name.__contains__('_xrew'):
                    reweights = np.log(1 / (self.num_inst / np.sum(self.num_inst)))
                    # print(reweights, self.HO_weight, self.num_inst_all, self.num_inst)
                    # import ipdb;ipdb.set_trace()
                    cls_score_hoi = tf.multiply(cls_score_hoi, reweights)
                if self.model_name.__contains__('batch') and self.model_name.__contains__('semi'):
                    semi_filter = tf.reduce_sum(self.H_boxes[:tf.shape(cls_score_hoi)[0], 1:], axis=-1)

                    semi_filter = tf.cast(semi_filter, tf.bool)

                    tmp_label_HO = tf.boolean_mask(tmp_label_HO, semi_filter, axis=0)
                    cls_score_hoi = tf.boolean_mask(cls_score_hoi, semi_filter, axis=0)

                tmp_hoi_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                         labels=tmp_label_HO, logits=cls_score_hoi)

                hoi_cross_entropy = tf.reduce_mean(tmp_hoi_loss)
                self.losses['hoi_cross_entropy'] = hoi_cross_entropy

                lamb = self.get_lamb_1()
                if "cls_score_sp" not in self.predictions:
                    sp_cross_entropy = 0
                    self.losses['sp_cross_entropy'] = 0
                loss = sp_cross_entropy + hoi_cross_entropy * lamb
                if self.model_name.__contains__('_orig_'):
                    loss = loss + O_cross_entropy + H_cross_entropy
                    print('Add all loss')
                if 'fake_G_cls_score_hoi' in self.predictions:
                    fake_cls_score_verbs = self.predictions["fake_G_cls_score_hoi"]
                    if self.model_name.__contains__('_rew_'):
                        fake_cls_score_verbs = tf.multiply(fake_cls_score_verbs, self.HO_weight)

                    elif self.model_name.__contains__('_rew2'):
                        fake_cls_score_verbs = tf.multiply(fake_cls_score_verbs, self.HO_weight / 10)
                    elif self.model_name.__contains__('_rew1'):
                        fake_cls_score_verbs = tf.multiply(fake_cls_score_verbs, self.HO_weight)
                    elif self.model_name.__contains__('rewn'):
                        pass
                    print(self.gt_class_HO_for_G_verbs, fake_cls_score_verbs, '======================================')
                    self.losses['fake_G_verbs_cross_entropy'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.gt_class_HO_for_G_verbs, logits=fake_cls_score_verbs))
                    if 'fake_G_total_loss' not in self.losses:
                        self.losses['fake_G_total_loss'] = 0
                    gll = 1.
                    self.losses['fake_G_total_loss'] += (self.losses['fake_G_verbs_cross_entropy'] * gll)
            else:
                loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy
            # verb loss
            temp = self.add_verb_loss(num_stop)
            loss += temp

            if self.model_name.__contains__('_objloss'):
                obj_cls_cross_entropy = self.add_objloss(num_stop)

                print('add objloss')
                loss += obj_cls_cross_entropy

            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)
        print(self.losses)
        print(self.predictions)
        return loss

    def add_factorized_hoi_loss(self, cls_score_objs, cls_score_verbs, gt_obj_label, gt_verb_label):
        # cls_score_verbs = tf.multiply(cls_score_verbs, self.HO_weight)
        # cls_score_objs = tf.multiply(cls_score_objs, self.HO_weight)
        # tmp_label_HO = tf.Print(tmp_label_HO, [tf.shape(tmp_label_HO), tf.shape(cls_score_verbs)],'sdfsdfsdf')
        # print('=======', tmp_label_HO, cls_score_verbs)
        if self.model_name.__contains__('batch') and self.model_name.__contains__('semi'):
            semi_filter = tf.reduce_sum(self.H_boxes[:tf.shape(cls_score_verbs)[0], 1:], axis=-1)

            semi_filter = tf.cast(semi_filter, tf.bool)

            gt_verb_label = tf.boolean_mask(gt_verb_label, semi_filter, axis=0)
            gt_obj_label = tf.boolean_mask(gt_obj_label, semi_filter, axis=0)
            cls_score_verbs = tf.boolean_mask(cls_score_verbs, semi_filter, axis=0)
            cls_score_objs = tf.boolean_mask(cls_score_objs, semi_filter, axis=0)
        tmp_verb_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=gt_verb_label, logits=cls_score_verbs)

        tmp_obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=gt_obj_label, logits=cls_score_objs)

        hoi_cross_entropy = tf.reduce_mean(tmp_verb_loss) + tf.reduce_mean(tmp_obj_loss)
        return hoi_cross_entropy

    def get_lamb_1(self):
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
        return lamb

    def filter_loss(self, cls_score, label):
        if self.model_name.__contains__('batch') and self.model_name.__contains__('semi'):
            semi_filter = tf.reduce_sum(self.H_boxes[:tf.shape(cls_score)[0], 1:], axis=-1)
            # label_sp = tf.Print(label_sp, [tf.shape(semi_filter), semi_filter, self.H_boxes, tf.shape(label_sp)], 'batch debug0:', first_n=000, summarize=1000)

            semi_filter = tf.cast(semi_filter, tf.bool)

            # label_sp = tf.Print(label_sp, [tf.shape(semi_filter), semi_filter, tf.shape(label_sp)], 'batch debug:', first_n=000, summarize=1000)
            label = tf.boolean_mask(label, semi_filter, axis=0)
            logits = tf.boolean_mask(cls_score, semi_filter, axis=0)
            # label = tf.Print(label, [tf.shape(semi_filter), tf.shape(label)], 'batch debug1:', first_n=000)

            sp_cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))

        else:
            sp_cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=cls_score))
        return sp_cross_entropy

    def cal_loss_by_weights(self, cls_score, label, orig_weights):
        sp_cross_entropy = tf.multiply(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=cls_score), orig_weights)
        sp_cross_entropy = tf.reduce_mean(sp_cross_entropy)
        return sp_cross_entropy

    def obtain_cbl_weights(self, tmp_label_HO, weights):
        # weights = tf.expand_dims(weights, 0)
        weights = tf.tile(weights, [tf.shape(tmp_label_HO)[0], 1]) * tmp_label_HO
        weights = tf.reduce_sum(weights, axis=1)
        weights = tf.expand_dims(weights, 1)
        weights = tf.tile(weights, [1, self.num_classes])
        return weights

    def add_objloss(self, num_stop):
        obj_cls_score = self.predictions["obj_cls_score_o"]
        if self.model_name.__contains__('_ce'):
            obj_cls_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.gt_obj_class[:num_stop], logits=obj_cls_score[:num_stop, :]))
        else:
            label_obj = tf.cast(
                tf.matmul(self.get_hoi_labels(), self.obj_to_HO_matrix, transpose_b=True) > 0,
                tf.float32)
            obj_cls_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label_obj[:tf.shape(obj_cls_score)[0], :], logits=obj_cls_score))
        self.losses["obj_cls_cross_entropy_o"] = obj_cls_cross_entropy

        model_name = self.model_name
        if model_name.__contains__('_pobjloss'):
            model_name = model_name.replace("_pobjloss", '_objloss')
        lambda1 = 0.1
        if model_name.__contains__('_objloss10'):
            lambda1 = 1.0
        elif self.model_name.__contains__('_objloss20'):
            lambda1 = 2.0
        elif model_name.__contains__('_objloss1'):
            lambda1 = 0.5
        elif model_name.__contains__('_objloss2'):
            lambda1 = 0.3
        elif model_name.__contains__('_objloss3'):
            lambda1 = 0.08
        elif model_name.__contains__('_objloss4'):
            lambda1 = 0.05
        temp = (obj_cls_cross_entropy * lambda1)

        return temp

    def add_verb_loss(self, num_stop):
        temp = 0
        if 'verb_cls_score' in self.predictions:
            vloss_num_stop = num_stop
            verb_cls_score = self.predictions["verb_cls_score"]
            verb_cls_cross_entropy = self.filter_loss(verb_cls_score[:vloss_num_stop, :],
                                                          self.gt_verb_class[:vloss_num_stop])
            self.losses["verb_cls_cross_entropy"] = verb_cls_cross_entropy
            if 'verb_cls_score_gcn' in self.predictions:
                verb_cls_score = self.predictions["verb_cls_score_gcn"]
                verb_cls_cross_entropy1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.gt_verb_class[:vloss_num_stop], logits=verb_cls_score[:vloss_num_stop, :]))
                self.losses["verb_cls_cross_entropy_gcn"] = verb_cls_cross_entropy1
                verb_cls_cross_entropy += verb_cls_cross_entropy1
            print('add vloss-------')
            # neg 0.1, negv1 0.5 negv12 0.1 1
            lambda1 = 0.1
            if self.model_name.__contains__('vloss10'):
                lambda1 = 1.0
            elif self.model_name.__contains__('vloss20'):
                lambda1 = 2.0
            elif self.model_name.__contains__('vloss1'):
                lambda1 = 0.5
            elif self.model_name.__contains__('vloss2'):
                lambda1 = 0.3
            elif self.model_name.__contains__('vloss3'):
                lambda1 = 0.08
            elif self.model_name.__contains__('vloss4'):
                lambda1 = 0.05
            temp = (verb_cls_cross_entropy * lambda1)
        if 'verb_cls_score_nvgcn_a' in self.predictions:
            vloss_num_stop = num_stop
            verb_cls_score = self.predictions["verb_cls_score_nvgcn_a"]
            verb_cls_cross_entropy = self.filter_loss(verb_cls_score[:vloss_num_stop, :],
                                                      self.gt_verb_class[:vloss_num_stop])

            self.losses["verb_cls_cross_entropy_nvgcn_a"] = verb_cls_cross_entropy
            print('add vloss===========')
            # neg 0.1, negv1 0.5 negv12 0.1 1
            lambda1 = 0.1
            if self.model_name.__contains__('_nvgcn_a10'):
                lambda1 = 1.0
            elif self.model_name.__contains__('_nvgcn_a1'):
                lambda1 = 0.5
            elif self.model_name.__contains__('_nvgcn_a2'):
                lambda1 = 0.3
            elif self.model_name.__contains__('_nvgcn_a3'):
                lambda1 = 0.08
            elif self.model_name.__contains__('_nvgcn_a4'):
                lambda1 = 0.05
            temp += (verb_cls_cross_entropy * lambda1)
        return temp

    def add_verb_ho_loss(self, num_stop):
        vloss_num_stop = num_stop
        verb_cls_score = self.predictions["verb_cls_score"]
        verb_cls_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.gt_class_HO[:vloss_num_stop], logits=verb_cls_score[:vloss_num_stop, :]))
        self.losses["verb_cls_cross_entropy"] = verb_cls_cross_entropy
        print('add vloss')
        # neg 0.1, negv1 0.5 negv12 0.1 1
        lambda1 = 1
        temp = (verb_cls_cross_entropy * lambda1)
        return temp

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = self.get_feed_dict(blobs)

        loss, _ = sess.run([self.losses['total_loss'],
                            train_op],
                           feed_dict=feed_dict)
        return loss

    # def train_step_with_summary(self, sess, blobs, lr, train_op):
    #     feed_dict = self.get_feed_dict(blobs)
    #
    #     loss, summary, _ = sess.run([self.losses['total_loss'],
    #                                  self.summary_op,
    #                                  train_op],
    #                                 feed_dict=feed_dict)
    #     return loss, summary


    def obtain_all_preds(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'], self.H_num: blobs['H_num'], self.O_mask: blobs['O_mask']}
        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:
            cls_prob_HO, pH, pO, pSp, pVerbs = sess.run(
                [self.predictions["cls_prob_HO"], self.predictions["cls_prob_H"],
                 self.predictions["cls_prob_O"], self.predictions["cls_prob_sp"],
                 self.predictions["cls_prob_verbs"]], feed_dict=feed_dict)

        except InvalidArgumentError as e:
            cls_prob_HO, pH, pO, pSp, pVerbs = sess.run(
                [self.predictions["cls_prob_HO_original"], self.predictions["cls_prob_H"],
                 self.predictions["cls_prob_O"], self.predictions["cls_prob_sp"],
                 self.predictions["cls_prob_H"]], feed_dict=feed_dict)
            print("InvalidArgumentError")

        return cls_prob_HO, pH, pO, pSp, pVerbs