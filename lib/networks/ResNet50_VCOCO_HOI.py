# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Transferable-Interactiveness-Network, Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.python.framework import ops

from ult.config import cfg
from ult.visualization import draw_bounding_boxes_HOI

import numpy as np

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc



class ResNet50():
    def __init__(self, model_name):
        self.model_name = model_name
        self.visualize = {}
        self.test_visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}

        self.image       = tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image')
        self.spatial     = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name = 'sp')
        # self.Hsp_boxes   = tf.placeholder(tf.float32, shape=[None, 5], name = 'Hsp_boxes')
        self.H_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='H_boxes')
        self.O_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'O_boxes')
        self.gt_class_H  = tf.placeholder(tf.float32, shape=[None, 24], name = 'gt_class_H')
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 24], name = 'gt_class_HO')
        self.gt_class_sp = tf.placeholder(tf.float32, shape=[None, 24], name = 'gt_class_sp')
        self.Mask_HO     = tf.placeholder(tf.float32, shape=[None, 24], name = 'HO_mask')
        self.Mask_H      = tf.placeholder(tf.float32, shape=[None, 24], name = 'H_mask')
        self.Mask_sp     = tf.placeholder(tf.float32, shape=[None, 24], name = 'sp_mask')
        self.gt_compose  = tf.placeholder(tf.float32, shape=[None, 222], name='gt_compose')
        self.gt_obj = tf.placeholder(tf.float32, shape=[None, 80], name='gt_obj')
        self.H_num       = tf.placeholder(tf.int32)
        self.image_id = tf.placeholder(tf.int32)
        self.num_classes = 24
        if self.model_name.__contains__('_t4_'):
            self.num_classes = 222
        if self.model_name.__contains__('_t5_'):
            self.verb_num_classes = 21
            self.num_classes = 222
        self.num_fc      = 1024
        self.verb_num_classes = 24
        self.obj_num_classes = 80
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        # self.lr          = tf.placeholder(tf.float32)
        if tf.__version__ == '1.1.0':
            self.blocks     = [resnet_utils.Block('block1', resnet_v1.bottleneck,[(256,   64, 1)] * 2 + [(256,   64, 2)]),
                               resnet_utils.Block('block2', resnet_v1.bottleneck,[(512,  128, 1)] * 3 + [(512,  128, 2)]),
                               resnet_utils.Block('block3', resnet_v1.bottleneck,[(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                               resnet_utils.Block('block4', resnet_v1.bottleneck,[(2048, 512, 1)] * 3),
                               resnet_utils.Block('block5', resnet_v1.bottleneck,[(2048, 512, 1)] * 3)]
        else:
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2),
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]
            if self.model_name.__contains__('unique_weights'):
                print("unique_weights2")
                self.blocks.append(resnet_v1_block('block6', base_depth=512, num_units=3, stride=1))

        # remove 3, 17 22, 23 27
        self.HO_weight = np.array([3.3510249, 3.4552405, 4.0257854, 4.088436,
                                   3.4370995, 3.85842, 4.637334, 3.5487218, 3.536237,
                                   2.5578923, 3.342811, 3.8897269, 4.70686, 3.3952892,
                                   3.9706533, 4.504736, 1.4873443, 3.700363,
                                   4.1058283, 3.6298118, 5.0808263,
                                   1.520838, 3.3888445, 3.9899964], dtype='float32').reshape(1, 24)
        self.H_weight = np.array([4.0984106, 4.102459, 4.0414762, 4.0414762,
                                  3.9768186, 4.23686, 5.3542085, 3.723717, 3.4699364,
                                  2.4587274, 3.7167964, 4.08836, 5.050695, 3.9077065,
                                  4.534647, 3.4699364, 1.8585607, 3.9433942,
                                  3.9433942, 4.3523254, 5.138182,
                                  1.7807873, 4.080392, 4.5761204], dtype='float32').reshape(1, 24)
        self.reset_classes()

    def set_ph(self, image, image_id, num_pos, sp, Human_augmented, Object_augmented,
               gt_cls_H = None, gt_cls_HO = None, gt_cls_sp = None,
               Mask_HO = None, Mask_H = None, Mask_sp = None, gt_compose = None, gt_obj=None):
        # image, image_id, H_num, spatial, H_boxes, O_boxes, gt_cls_H,
        # gt_cls_HO, gt_cls_sp, Mask_HO, Mask_H, Mask_sp, gt_compose
        if image is not None: self.image       = image
        if image_id is not None: self.image_id = image_id
        if sp is not None: self.spatial     = sp
        if Human_augmented is not None: self.H_boxes     = Human_augmented
        # self.Hsp_boxes   = Hsp_boxes
        if Object_augmented is not None: self.O_boxes     = Object_augmented
        if gt_cls_H is not None: self.gt_class_H  = gt_cls_H
        if gt_cls_HO is not None: self.gt_class_HO = gt_cls_HO
        if gt_cls_sp is not None: self.gt_class_sp = gt_cls_sp
        if Mask_HO is not None: self.Mask_HO     = Mask_HO
        if Mask_H is not None: self.Mask_H      = Mask_H
        if Mask_sp is not None: self.Mask_sp     = Mask_sp
        if num_pos is not None: self.H_num       = num_pos
        if gt_compose is not None: self.gt_compose = gt_compose
        if gt_obj is not None: self.gt_obj = gt_obj
        print("set ph:", self.image)
        if self.gt_compose is not None:
            self.reset_classes()

    def reset_classes(self):

        from networks.tools import get_convert_matrix_coco3
        if self.model_name.__contains__('_t1_'):
            raise Exception("wrong model. t1 is depressed")
        elif self.model_name.__contains__('_t2_') or self.model_name.__contains__('_t3_'):
            self.verb_num_classes = 24
            self.obj_num_classes = 80
            self.num_classes = 24
            self.compose_num_classes = 222
            verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(self.verb_num_classes, self.obj_num_classes)

            self.obj_to_HO_matrix = tf.constant(obj_to_HO_matrix, tf.float32)
            self.verb_to_HO_matrix = tf.constant(verb_to_HO_matrix, tf.float32)
            self.gt_obj_class = tf.cast(tf.matmul(self.gt_compose, self.obj_to_HO_matrix, transpose_b=True) > 0,
                                        tf.float32)
            self.gt_verb_class = tf.cast(tf.matmul(self.gt_compose, self.verb_to_HO_matrix, transpose_b=True) > 0,
                                         tf.float32)
        elif self.model_name.__contains__('_t4_'):
            self.verb_num_classes = 24
            self.obj_num_classes = 80
            self.num_classes = 222
            self.compose_num_classes = 222
            verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(self.verb_num_classes, self.obj_num_classes)

            self.obj_to_HO_matrix = tf.constant(obj_to_HO_matrix, tf.float32)
            self.verb_to_HO_matrix = tf.constant(verb_to_HO_matrix, tf.float32)
            self.gt_obj_class = tf.cast(tf.matmul(self.gt_compose, self.obj_to_HO_matrix, transpose_b=True) > 0,
                                        tf.float32)
            self.gt_verb_class = tf.cast(tf.matmul(self.gt_compose, self.verb_to_HO_matrix, transpose_b=True) > 0,
                                         tf.float32)
        elif self.model_name.__contains__('_t5_'):
            self.verb_num_classes = 21
            self.obj_num_classes = 80
            self.num_classes = 222
            self.compose_num_classes = 222
            verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco3(self.verb_num_classes, self.obj_num_classes)

            self.obj_to_HO_matrix_np = obj_to_HO_matrix
            self.verb_to_HO_matrix_np = verb_to_HO_matrix
            self.obj_to_HO_matrix = tf.constant(obj_to_HO_matrix, tf.float32)
            self.verb_to_HO_matrix = tf.constant(verb_to_HO_matrix, tf.float32)
            self.gt_obj_class = tf.cast(tf.matmul(self.gt_compose, self.obj_to_HO_matrix, transpose_b=True) > 0,
                                        tf.float32)
            self.gt_verb_class = tf.cast(tf.matmul(self.gt_compose, self.verb_to_HO_matrix, transpose_b=True) > 0,
                                         tf.float32)

            num_inst = np.asarray([485, 434, 3, 6, 6, 3, 3, 207, 1, 3, 4, 7, 1, 7, 32, 2, 160, 37, 67, 9, 126, 1, 24,
                                   6, 31, 108, 73, 292, 134, 398, 86, 28, 39, 21, 3, 60, 4, 7, 1, 61, 110, 80, 56, 56,
                                   119, 107, 96, 59, 2, 1, 4, 430, 136, 55, 1, 5, 1, 20, 165, 278, 26, 24, 1, 29, 228,
                                   1, 15, 55, 54, 1, 2, 57, 52, 93, 72, 3, 7, 12, 6, 6, 1, 11, 105, 4, 2, 1, 1, 7, 1,
                                   17, 1, 1, 2, 170, 91, 445, 6, 1, 2, 5, 1, 12, 4, 1, 1, 1, 14, 18, 7, 7, 5, 8, 4, 7,
                                   4, 1, 3, 9, 390, 45, 156, 521, 15, 4, 5, 338, 254, 3, 5, 11, 15, 12, 43, 12, 12, 2,
                                   2, 14, 1, 11, 37, 18, 134, 1, 7, 1, 29, 291, 1, 3, 4, 62, 4, 75, 1, 22, 228, 109,
                                   233, 1, 366, 86, 50, 46, 68, 1, 1, 1, 1, 8, 14, 45, 2, 5, 45, 70, 89, 9, 99, 186,
                                   50, 56, 54, 9, 120, 66, 56, 160, 269, 32, 65, 83, 67, 197, 43, 13, 26, 5, 46, 3, 6,
                                   1, 60, 67, 56, 20, 2, 78, 11, 58, 1, 350, 1, 83, 41, 18, 2, 9, 1, 466, 224, 32])
            self.num_inst = self.num_inst_all = num_inst
            tmp = np.where(num_inst > 10)[0]
            tmp1 = np.zeros(self.num_classes)
            tmp1[tmp] = 1
            self.non_rare_cls_index = tf.constant(tmp1)

            tmp = np.where(num_inst <= 10)[0]
            tmp1 = np.zeros(self.num_classes)
            tmp1[tmp] = 1
            self.rare_cls_index = tf.constant(tmp1)

        else:
            pass
            # verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix_coco(self.verb_num_classes, self.obj_num_classes)
            # self.obj_to_HO_matrix = tf.constant(obj_to_HO_matrix, tf.float32)
            # self.verb_to_HO_matrix = tf.constant(verb_to_HO_matrix, tf.float32)
            # self.gt_obj_class = tf.cast(tf.matmul(self.gt_class_HO, self.obj_to_HO_matrix, transpose_b=True) > 0,
            #                             tf.float32)
            # self.gt_verb_class = tf.cast(tf.matmul(self.gt_class_HO, self.verb_to_HO_matrix, transpose_b=True) > 0,
            #                              tf.float32)

        from networks.tools import get_word2vec
        word2vec = get_word2vec()
        self.word2vec_emb = tf.constant(word2vec)
        self.gt_class_HO_for_G_verbs = None
        self.gt_class_HO_for_D_verbs = None

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope, reuse=tf.AUTO_REUSE,):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def image_to_head(self, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                         global_pool=False,
                                         include_root_block=False,
                                         reuse=tf.AUTO_REUSE,
                                         scope=self.scope)

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            if self.model_name.__contains__('unique_weights'):
                print("unique_weights3")
                stop = -3
            else:
                stop = -2
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:stop],
                                          global_pool=False,
                                          include_root_block=False,
                                          reuse=tf.AUTO_REUSE,
                                          scope=self.scope)

        return head


    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope, reuse=tf.AUTO_REUSE,):
            ends = 2
            if self.model_name.__contains__('_spose'):
                ends = 3
            conv1_sp      = slim.conv2d(self.spatial[:,:,:,:ends], 64, [5, 5], padding='VALID', scope='conv1_sp')
            pool1_sp      = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            conv2_sp      = slim.conv2d(pool1_sp,     32, [5, 5], padding='VALID', scope='conv2_sp')
            pool2_sp      = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            pool2_flat_sp = slim.flatten(pool2_sp)

        return pool2_flat_sp


    def res5(self, pool5_H, pool5_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            fc7_H, _ = resnet_v1.resnet_v1(pool5_H,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=tf.AUTO_REUSE,
                                           scope=self.scope)

            # fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])


            fc7_O, _ = resnet_v1.resnet_v1(pool5_O,
                                       self.blocks[-1:],
                                       global_pool=False,
                                       include_root_block=False,
                                       reuse=tf.AUTO_REUSE,
                                       scope=self.scope)

            # fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])
        
        return fc7_H, fc7_O

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training), reuse=tf.AUTO_REUSE):

            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH     = tf.concat([fc7_H[:self.H_num,:], fc7_SH[:self.H_num,:]], 1)

            fc8_SH        = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH')
            fc8_SH        = slim.dropout(fc8_SH, keep_prob=0.5, is_training=is_training, scope='dropout8_SH')
            fc9_SH        = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH        = slim.dropout(fc9_SH, keep_prob=0.5, is_training=is_training, scope='dropout9_SH')  


            Concat_SO     = tf.concat([fc7_O, fc7_SO], 1)

            fc8_SO        = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO        = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO        = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO        = slim.dropout(fc9_SO,    keep_prob=0.5, is_training=is_training, scope='dropout9_SO')  


            Concat_SHsp   = tf.concat([fc7_H, sp], 1)
            Concat_SHsp   = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp')
            Concat_SHsp   = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp      = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp')
            fc7_SHsp      = slim.dropout(fc7_SHsp,  keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')


        return fc9_SH, fc9_SO, fc7_SHsp

    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:

            batch_ids    = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bboxes = self.trans_boxes_by_feats(bottom, rois)

            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def trans_boxes_by_feats(self, bottom, rois):
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        return bboxes

    def attention_pool_layer_H(self, bottom, fc7_H, is_training, name):
        with tf.variable_scope(name) as scope:

            fc1         = slim.fully_connected(fc7_H, 512, scope='fc1_b', reuse=tf.AUTO_REUSE, )
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att


    def attention_norm_H(self, att, name):
        with tf.variable_scope(name) as scope:

            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att         = tf.nn.softmax(att)
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def attention_pool_layer_O(self, bottom, fc7_O, is_training, name):
        with tf.variable_scope(name) as scope:

            fc1         = slim.fully_connected(fc7_O, 512, scope='fc1_b', reuse=tf.AUTO_REUSE)
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att


    def attention_norm_O(self, att, name):
        with tf.variable_scope(name) as scope:

            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att         = tf.nn.softmax(att)
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def region_classification(self, fc7_H, fc7_O, fc7_SHsp, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            cls_score_H  = slim.fully_connected(fc7_H, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_H')
            cls_prob_H   = tf.nn.sigmoid(cls_score_H, name='cls_prob_H') 
            tf.reshape(cls_prob_H, [1, self.num_classes])   

            cls_score_O  = slim.fully_connected(fc7_O, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_O')
            cls_prob_O  = tf.nn.sigmoid(cls_score_O, name='cls_prob_O') 
            tf.reshape(cls_prob_O, [1, self.num_classes]) 

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_sp')
            cls_prob_sp  = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp') 
            tf.reshape(cls_prob_sp, [1, self.num_classes]) 


            self.predictions["cls_score_H"] = cls_score_H
            self.predictions["cls_prob_H"]  = cls_prob_H
            self.predictions["cls_score_O"] = cls_score_O
            self.predictions["cls_prob_O"]  = cls_prob_O
            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"]  = cls_prob_sp

            self.predictions["cls_prob_HO"]  = cls_prob_sp * (cls_prob_O + cls_prob_H)

        return cls_prob_H, cls_prob_O, cls_prob_sp

    def bottleneck(self, bottom, is_training, name, reuse=False):
        with tf.variable_scope(name) as scope:

            # if reuse:
            #     scope.reuse_variables()

            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name, reuse=tf.AUTO_REUSE, )

        return head_bottleneck



    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        # ResNet Backbone
        head       = self.image_to_head(is_training)
        sp         = self.sp_to_head()
        pool5_H    = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')
        pool5_O    = self.crop_pool_layer(head, self.O_boxes,   'Crop_O')

        fc7_H, fc7_O = self.res5(pool5_H, pool5_O, sp, is_training, 'res5')
        fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])
        fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])

        # Phi 
        head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')

        # g 
        head_g   = slim.conv2d(head, 512, [1, 1], scope='head_g')

        Att_H      = self.attention_pool_layer_H(head_phi, fc7_H[:self.H_num,:], is_training, 'Att_H')
        Att_H      = self.attention_norm_H(Att_H, 'Norm_Att_H')

        att_head_H = tf.multiply(head_g, Att_H)

        Att_O      = self.attention_pool_layer_O(head_phi, fc7_O, is_training, 'Att_O')
        Att_O      = self.attention_norm_O(Att_O, 'Norm_Att_O')
        att_head_O = tf.multiply(head_g, Att_O)

        pool5_SH     = self.bottleneck(att_head_H, is_training, 'bottleneck', False)
        pool5_SO     = self.bottleneck(att_head_O, is_training, 'bottleneck', True)


        fc7_SH, fc7_SO, fc7_SHsp = self.head_to_tail(fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, 'fc_HO')

        cls_prob_H, cls_prob_O, cls_prob_sp = self.region_classification(fc7_SH, fc7_SO, fc7_SHsp, is_training, initializer, 'classification')

        self.score_summaries.update(self.predictions)

        self.visualize["attention_map_H"] = (Att_H - tf.reduce_min(Att_H[0,:,:,:])) / tf.reduce_max((Att_H[0,:,:,:] - tf.reduce_min(Att_H[0,:,:,:])))
        self.visualize["attention_map_O"] = (Att_O - tf.reduce_min(Att_O[0,:,:,:])) / tf.reduce_max((Att_O[0,:,:,:] - tf.reduce_min(Att_O[0,:,:,:])))
        return cls_prob_H, cls_prob_O, cls_prob_sp



    def create_architecture(self, is_training):

        self.build_network(is_training)

        for var in tf.trainable_variables():
            self.train_summaries.append(var)

        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            # val_summaries.append(self.add_gt_image_summary_H())
            # val_summaries.append(self.add_gt_image_summary_HO())
            # tf.summary.image('ATTENTION_MAP_H',  self.visualize["attention_map_H"], max_outputs=1)
            # tf.summary.image('ATTENTION_MAP_O',  self.visualize["attention_map_O"], max_outputs=1)
            for key, var in self.event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            # for key, var in self.score_summaries.items():
            #     self.add_score_summary(key, var)
            # for var in self.train_summaries:
            #     self.add_train_summary(var)
        
            # val_summaries.append(tf.summary.scalar('lr', self.lr))
            self.summary_op     = tf.summary.merge_all()
            self.summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output


    def add_loss(self):

        with tf.variable_scope('LOSS') as scope:
            cls_score_H  = self.predictions["cls_score_H"]
            cls_score_O  = self.predictions["cls_score_O"]
            cls_score_sp = self.predictions["cls_score_sp"]

            label_H      = self.gt_class_H
            label_HO     = self.gt_class_HO
            label_sp     = self.gt_class_sp

            H_mask       = self.Mask_H
            HO_mask      = self.Mask_HO
            sp_mask      = self.Mask_sp

            H_cross_entropy  = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_H,  logits = cls_score_H),   H_mask))
            HO_cross_entropy = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO, logits = cls_score_O),  HO_mask))
            sp_cross_entropy = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_sp, logits = cls_score_sp), sp_mask))


            self.losses['H_cross_entropy']  = H_cross_entropy
            self.losses['HO_cross_entropy'] = HO_cross_entropy
            self.losses['sp_cross_entropy'] = sp_cross_entropy

            loss = 2 * H_cross_entropy + HO_cross_entropy + sp_cross_entropy

            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)

        return loss


    def add_gt_image_summary_H(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.H_boxes, self.gt_class_H],
                      tf.float32, name="gt_boxes_H")
        return tf.summary.image('GROUND_TRUTH_H', image)

    def add_gt_image_summary_HO(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.O_boxes, self.gt_class_HO],
                      tf.float32, name="gt_boxes_HO")
        return tf.summary.image('GROUND_TRUTH_HO)', image)


    def add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)


    def add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def get_feed_dict(self, blobs):
        feed_dict = {self.image: blobs['image'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_H: blobs['gt_class_H'],
                     self.gt_class_HO: blobs['gt_class_HO'], self.Mask_H: blobs['Mask_H'],
                     self.Mask_HO: blobs['Mask_HO'], self.spatial:blobs['sp'],
                     self.Mask_sp: blobs['Mask_sp'],
                     self.gt_class_sp: blobs['gt_class_sp'], self.H_num: blobs['H_num'],
                     self.H_boxes: blobs['H_boxes']}
        return feed_dict

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = self.get_feed_dict(blobs)
        
        loss, _ = sess.run([self.losses['total_loss'],
                                                     train_op],
                                                     feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = self.get_feed_dict(blobs)

        loss, summary, _ = sess.run([
                                                              self.losses['total_loss'],
                                                              self.summary_op,
                                                              train_op],
                                                              feed_dict=feed_dict)
        return loss, summary

    def train_step_tfr(self, sess, blobs, lr, train_op):
        loss, image_id, _ = sess.run([self.losses['total_loss'], self.image_id,
                            train_op])
        return loss, image_id

    def train_step_tfr_with_summary(self, sess, blobs, lr, train_op):

        loss, summary, image_id,  _ = sess.run([self.losses['total_loss'],
                                     self.summary_op, self.image_id,
                                     train_op])
        return loss, image_id, summary
    
    def test_image_H(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.H_num: blobs['H_num']}

        cls_prob_H = sess.run([self.predictions["cls_prob_H"]], feed_dict=feed_dict)

        return cls_prob_H

    
    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], self.spatial: blobs['sp'], self.H_num: blobs['H_num']}

        cls_prob_HO = sess.run([self.predictions["cls_prob_HO"]], feed_dict=feed_dict)

        return cls_prob_HO