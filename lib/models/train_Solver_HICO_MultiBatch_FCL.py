# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.train_Solver_HICO import SolverWrapper
from ult.config import cfg
from ult.ult import get_epoch_iters
from ult.timer import Timer

import pickle
import numpy as np
import os
import sys
import glob
import time
import ipdb

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class SolverWrapperMultiBatchFCL(SolverWrapper):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, output_dir, tbdir, Restore_flag, pretrained_model):
        super(SolverWrapperMultiBatchFCL, self).__init__(sess, network, None, None, output_dir, tbdir, None, None,
                                                         Restore_flag, pretrained_model)

        self.image = None

        self.image_id = None
        self.spatial = None
        self.H_boxes = None
        self.O_boxes = None
        self.gt_class_HO = None
        self.H_num = None

    def set_data(self, image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp):
        self.image = image
        self.image_id = image_id
        self.spatial = sp
        self.H_boxes = Human_augmented
        self.O_boxes = Object_augmented
        self.gt_class_HO = action_HO
        self.H_num = num_pos

    def construct_graph(self, sess):
        print("construct_graph, SolverWrapperMultiBatchFCL")
        compose_type = self.compose_feature_helper.get_compose_type()
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            init_step = self.get_init_step()

            step_factor = self.get_step_factor()

            with tf.variable_scope('tmp', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable('global_step', shape=(), trainable=False,
                                              initializer=tf.constant_initializer(0))
                global_step1 = tf.get_variable('global_step1', shape=(), trainable=False,
                                               initializer=tf.constant_initializer(0))

            lr, self.optimizer = self.get_optimzer_lr(global_step, step_factor)
            lr1, self.optimizer_g = self.get_optimzer_lr(global_step1, step_factor)
            # self.optimizer_d = tf.train.AdamOptimizer()
            tower_grads = []
            num_stop_list = []
            tower_losses = []
            fake_tower_losses = []

            # Build the main computation graph
            layers = self.net.create_architecture(True)  # is_training flag: True

            num_stop_list.append(self.net.get_compose_num_stop())
            # Define the loss
            loss = layers['total_loss'] + layers['fake_total_loss']
            fake_loss = layers['fake_G_total_loss']
            tower_losses.append(loss)
            # fake_tower_losses.append(fake_loss)
            # variables = tf.trainable_variables()
            # grads_and_vars = self.optimizer.compute_gradients(loss, variables)
            # tower_grads.append(grads_and_vars)

            print('compose learning ====================================== batch', compose_type)

            if not self.net.model_name.__contains__('_base'):
                semi_filter = tf.reduce_sum(self.H_boxes[:self.net.get_compose_num_stop(), 1:], axis=-1)
                semi_filter = tf.cast(semi_filter, tf.bool)

                verb_feats = self.net.intermediate['fc7_verbs']
                semi_filter = tf.cast(semi_filter, tf.float32)
                if self.net.model_name.__contains__('atl1'):
                    verb_feats = tf.multiply(verb_feats, tf.expand_dims(semi_filter, axis=-1))  # remove semi data
                    verb_feats = tf.Print(verb_feats, [tf.reduce_sum(verb_feats, axis=-1)], 'message semi1====',
                                          summarize=10000,
                                          first_n=100)
                    print('============', verb_feats)
                O_features = [self.net.intermediate['fc7_O'][:self.net.pos1_idx],
                              self.net.intermediate['fc7_O'][self.net.pos1_idx:self.net.get_compose_num_stop()]]
                V_features = [verb_feats[:self.net.pos1_idx],
                              verb_feats[self.net.pos1_idx:self.net.get_compose_num_stop()]]

                if self.net.model_name.__contains__('atl1'):
                    new_loss = self.compose_feature_helper.merge_generate(O_features,
                                                                          V_features,
                                                                          [self.gt_class_HO[:self.net.pos1_idx],
                                                                           self.gt_class_HO[
                                                                           self.net.pos1_idx:self.net.get_compose_num_stop()]],
                                                                          compose_type)
                    new_loss = new_loss['vcl_loss']
                elif self.net.model_name.__contains__('atl'):
                    new_loss = self.compose_feature_helper.merge_generate(O_features,
                                                                          V_features,
                                                                          [self.gt_class_HO[:self.net.pos1_idx],
                                                                           self.gt_class_HO[
                                                                           self.net.pos1_idx:self.net.get_compose_num_stop()]],
                                                                          'semi')
                    new_loss = new_loss['vcl_loss']
                else:
                    new_loss = self.compose_feature_helper.merge_generate(O_features,
                                                                          V_features,
                                                                          [self.gt_class_HO[:self.net.pos1_idx],
                                                                           self.gt_class_HO[
                                                                           self.net.pos1_idx:self.net.get_compose_num_stop()]],
                                                                          compose_type)
                    new_loss = new_loss['vcl_loss']
                ll = self.compose_feature_helper.get_ll()
                tower_losses.append(new_loss * ll)
            variables = [v for v in tf.trainable_variables()]
            if self.net.model_name.__contains__('epic3'):
                variables = [v for v in tf.trainable_variables() if not v.name.__contains__('generator')]
            grads_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(tower_losses), variables)

            g_update_variables = [v for v in tf.trainable_variables() if
                                  v.name.__contains__('generator')]

            g_grads_and_vars = self.optimizer_g.compute_gradients(fake_loss, g_update_variables)

            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]
            g_capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in g_grads_and_vars if grad is not None]

            # self.addition_loss(capped_gvs, layers)
            train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)
            train_op_g = self.optimizer_g.apply_gradients(g_capped_gvs, global_step=global_step1)

            tf.summary.scalar('lr', lr)
            # tf.summary.scalar('merge_loss', new_loss)
            self.net.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
        return lr, train_op, tf.reduce_sum(tower_losses), train_op_g, fake_loss

    def train_model_stepwise_inner(self, D_loss, g_loss, iter, lr, max_iters, sess, timer, train_op, train_op_g):
        while iter < max_iters + 1:
            timer.tic()

            total_loss = 0
            fake_total_loss = 0
            #
            save_iters = 50000
            epoch_stride = 5
            if self.net.model_name.__contains__('_s3_'):
                epoch_stride = 3
            elif self.net.model_name.__contains__('_s1_'):
                epoch_stride = 1
            elif self.net.model_name.__contains__('_s05_'):
                epoch_stride = 0.5
            elif self.net.model_name.__contains__('_s0_'):
                epoch_stride = 0
            save_iters = get_epoch_iters(self.net.model_name)
            if iter < save_iters * epoch_stride and not self.net.model_name.__contains__('_reload'):
                if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):
                    # Compute the graph with summary
                    fake_total_loss, _, summary, image_id = sess.run(
                        [g_loss, train_op_g, self.net.summary_op, self.net.image_id, ])

                    # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                    self.writer.add_summary(summary, float(iter))
                else:
                    # Compute the graph without summary
                    fake_total_loss, _, image_id = sess.run([g_loss, train_op_g, self.net.image_id, ])
                if iter + 1 == save_iters * epoch_stride:
                    iter = save_iters - 1
            else:
                if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):

                    # Compute the graph with summary
                    total_loss, _, summary, image_id = sess.run(
                        [D_loss, train_op, self.net.summary_op, self.net.image_id, ])

                    # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                    self.writer.add_summary(summary, float(iter))

                else:
                    # Compute the graph without summary
                    total_loss, _, image_id = sess.run([D_loss, train_op, self.net.image_id, ])

            timer.toc()
            # print(image_id)
            # Display training information
            if iter % cfg.TRAIN.DISPLAY == 0:
                # if type(image_id) == tuple:
                #     image_id = image_id[0]
                # print(image_id)
                print('iter: {:d} / {:d}, im_id: {:d}, loss: {:.6f}, G: {:.6f} lr: {:f}, speed: {:.3f} s/iter'.format(
                    iter, max_iters, image_id[0], total_loss, fake_total_loss, lr.eval(), timer.average_time), end='\n',
                    flush=True)
            # print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 15765, _image_id,
            #                                                                _t['im_detect'].average_time), end='',
            #       flush=True)
            # Snapshotting
            t_iter = iter
            # if iter == 0 and self.net.model_name.__contains__('_pret'):
            #     t_iter = 1000000
            self.snapshot(sess, t_iter)

            iter += 1

    def train_model_epic3_inner(self, D_loss, g_loss, iter, lr, max_iters, sess, timer, train_op, train_op_g):
        while iter < max_iters + 1:
            timer.tic()

            total_loss = 0
            fake_total_loss = 0
            #
            save_iters = 50000
            epoch_stride = 5

            if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):
                # Compute the graph with summary
                fake_total_loss, _, summary, image_id = sess.run(
                    [g_loss, train_op_g, self.net.summary_op, self.net.image_id, ])

                # Compute the graph with summary
                total_loss, _, summary, image_id = sess.run(
                    [D_loss, train_op, self.net.summary_op, self.net.image_id, ])

                # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                self.writer.add_summary(summary, float(iter))
            else:
                # Compute the graph without summary
                fake_total_loss, _, image_id = sess.run([g_loss, train_op_g, self.net.image_id, ])
                # Compute the graph without summary
                total_loss, _, image_id = sess.run([D_loss, train_op, self.net.image_id, ])

            timer.toc()
            # print(image_id)
            # Display training information
            if iter % cfg.TRAIN.DISPLAY == 0:
                # if type(image_id) == tuple:
                #     image_id = image_id[0]
                # print(image_id)
                print('iter: {:d} / {:d}, im_id: {:d}, loss: {:.6f}, G: {:.6f} lr: {:f}, speed: {:.3f} s/iter'.format(
                    iter, max_iters, image_id[0], total_loss, fake_total_loss, lr.eval(), timer.average_time), end='\n',
                    flush=True)
            # print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 15765, _image_id,
            #                                                                _t['im_detect'].average_time), end='',
            #       flush=True)
            # Snapshotting
            t_iter = iter
            # if iter == 0 and self.net.model_name.__contains__('_pret'):
            #     t_iter = 1000000
            self.snapshot(sess, t_iter)

            iter += 1

    def train_model(self, sess, max_iters):
        lr, train_op, D_loss, train_op_g, g_loss = self.construct_graph(sess)

        self.from_snapshot(sess)

        sess.graph.finalize()

        timer = Timer()

        # Data_length = len(self.Trainval_GT)
        iter = self.get_init_step()
        self.train_model_stepwise_inner(D_loss, g_loss, iter, lr, max_iters, sess, timer, train_op, train_op_g)
        # elif self.net.model_name.__contains__('epic3'):
        #     self.train_model_epic3_inner(D_loss, g_loss, iter, lr, max_iters, sess, timer, train_op, train_op_g)

        self.writer.close()
