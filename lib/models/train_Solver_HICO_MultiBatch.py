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
from ult.timer import Timer

import numpy as np


import tensorflow as tf
from tensorflow.python import pywrap_tensorflow



class SolverWrapperMultiBatch(SolverWrapper):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, output_dir, tbdir, Restore_flag, pretrained_model):
        super(SolverWrapperMultiBatch, self).__init__(sess, network, None, None, output_dir, tbdir, None, None, Restore_flag, pretrained_model)

        self.image = None

        self.image_id = None
        self.spatial = None
        self.H_boxes  = None
        self.O_boxes  = None
        self.gt_class_HO  = None
        self.H_num  = None

    def set_data(self, image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp):
        self.image = image
        self.image_id = image_id
        self.spatial = sp
        self.H_boxes = Human_augmented
        self.O_boxes = Object_augmented
        self.gt_class_HO = action_HO
        self.H_num = num_pos

    def construct_graph(self, sess):
        print("construct_graph")
        compose_type = self.compose_feature_helper.get_compose_type()
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            init_step = self.get_init_step()

            # global_step = tf.Variable(init_step, trainable=False, name='global_step')
            with tf.variable_scope('tmp', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable('global_step', shape=(), trainable=False, initializer=tf.constant_initializer(0))

            step_factor = self.get_step_factor()

            lr, self.optimizer = self.get_optimzer_lr(global_step, step_factor)

            tower_grads = []
            tower_losses = []


            # Build the main computation graph
            layers = self.net.create_architecture(True)  # is_training flag: True

            # Define the loss
            loss = layers['total_loss']
            tower_losses.append(loss)
            if not self.net.model_name.__contains__('base'):
                semi_filter = tf.reduce_sum(self.H_boxes[:self.net.get_compose_num_stop(), 1:], axis=-1)
                semi_filter = tf.cast(semi_filter, tf.bool)
                verb_feats = self.net.intermediate['fc7_verbs']
                semi_filter = tf.cast(semi_filter, tf.float32)
                if self.net.model_name.__contains__('atl1'):
                    verb_feats = tf.multiply(verb_feats, tf.expand_dims(semi_filter,axis=-1)) # remove semi data
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
                                                                          [self.gt_class_HO[:self.net.pos1_idx], self.gt_class_HO[self.net.pos1_idx:self.net.get_compose_num_stop()]],
                                                                          'atl')
                    new_loss = new_loss['vcl_loss']
                else:
                    new_loss = self.compose_feature_helper.merge_generate(O_features,
                                                                      V_features,
                                                                      [self.gt_class_HO[:self.net.pos1_idx], self.gt_class_HO[self.net.pos1_idx:self.net.get_compose_num_stop()]],
                                                                      compose_type)
                    new_loss = new_loss['vcl_loss']
                ll = self.compose_feature_helper.get_ll()
                tower_losses.append(new_loss * ll)
            if self.net.model_name.__contains__('pret') and self.net.model_name.__contains__('classifier'):
                variables = self.get_classifier_variables()
            else:
                variables = tf.trainable_variables()
            grads_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(tower_losses), variables)
            tower_grads.append(grads_and_vars)
            # with tf.device('/gpu:%d' % 0):
            #     new_loss = self.merge_generate(O_features, V_features)
            #     tower_losses.append(new_loss)
            #     variables = tf.trainable_variables()
            #     grads_and_vars = self.optimizer.compute_gradients(new_loss, variables)
            #     tower_grads.append(grads_and_vars)
            #
            #     print('length of grads:', len(tower_grads))
            # grads_and_vars = self.average_gradients(tower_grads)
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]

            # self.addition_loss(capped_gvs, layers)

            for grad, var in capped_gvs:
                print('update: {}'.format(var.name))
            train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)
            tf.summary.scalar('lr', lr)
            self.net.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
        return lr, train_op, tf.reduce_sum(tower_losses)


    def train_model(self, sess, max_iters):
        lr, train_op, t_loss = self.construct_graph(sess)
        self.from_snapshot(sess)
        
        sess.graph.finalize()

        timer = Timer()
        import logging
        logging.basicConfig(filename='/home/zhou9878/{}.log'.format(self.net.model_name), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        iter = self.get_init_step()
        while iter < max_iters + 1:
            timer.tic()
            if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):

                # Compute the graph with summary
                # total_loss, image_id, summary = self.net.train_step_tfr_with_summary(sess, blobs, lr, train_op)
                total_loss, summary, image_id, _ = sess.run([t_loss,
                                                       self.net.summary_op, self.net.image_id,
                                                       train_op])
                # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                self.writer.add_summary(summary, float(iter))

            else:
                # Compute the graph without summary
                total_loss, image_id, _ = sess.run([t_loss, self.net.image_id,
                                                       train_op])
                # total_loss, image_id = self.net.train_step_tfr(sess, blobs, lr, train_op)

            timer.toc()
            # print(image_id)
            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                if type(image_id) == tuple or (type(image_id) != np.int32 and len(image_id) > 1):
                    image_id = image_id[0]
                # print('iter: {:d} / {:d}, im_id: {:d}, total loss: {:.6f}, lr: {:f}, speed: {:.3f} s/iter'.format(
                #       iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time), end='\n', flush=True)
                logger.info('iter: {:d} / {:d}, im_id: {:d}, total loss: {:.6f}, lr: {:f}, speed: {:.3f} s/iter'.format(
                      iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time))
            # print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 15765, _image_id,
            #                                                                _t['im_detect'].average_time), end='',
            #       flush=True)
            # Snapshotting

            self.snapshot(sess, iter)

            iter += 1

        self.writer.close()
