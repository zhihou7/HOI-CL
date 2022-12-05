# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.train_Solver_HICO import SolverWrapper
from ult.config import cfg
from ult.timer import Timer

import os


import tensorflow as tf



class SolverWrapperMultiGPU(SolverWrapper):
    """
    A wrapper class for the training process
    I do not implement this in a multi-gpu way because I suffer a wired bug when I run the code in two gpu.
    Thus, all the experiments are based on one GPU.
    Hope someone can solve the bug https://github.com/tensorflow/tensorflow/issues/32836
    """

    def __init__(self, sess, network, output_dir, tbdir, Restore_flag, pretrained_model):
        super(SolverWrapperMultiGPU, self).__init__(sess, network, None, None, output_dir, tbdir, None, None, Restore_flag, pretrained_model)

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

    def construct_graph2(self, sess):
        print("construct_graph2")
        compose_type = self.compose_feature_helper.get_compose_type()
        with sess.graph.as_default(), tf.device('/cpu:0'):
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            init_step = self.get_init_step()

            # global_step = tf.Variable(init_step, trainable=False, name='global_step')
            with tf.variable_scope('tmp', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable('global_step', shape=(), trainable=False, initializer=tf.constant_initializer(0))

            step_factor = self.get_step_factor()

            lr, self.optimizer = self.get_optimzer_lr(global_step, step_factor)

            tower_grads = []
            V_features = []
            O_features = []
            num_stop_list = []
            tower_losses = []

            for i in range(2):
                gpu_idx = i
                if 'CUDA_VISIBLE_DEVICES' not in os.environ or len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1:
                    gpu_idx = 0
                # TODO if there are multiple GPUs, the code possibly raises an Exception.
                #  But I think multiple GPUs should obtain better result.
                with tf.device('/gpu:%d' % gpu_idx):
                    with tf.name_scope('%s_%d' % ('HICO', i), ) as scope:
                        split_image = self.image[i]
                        split_image_id = self.image_id[i]
                        split_spatial = self.spatial[i]
                        split_H_boxes = self.H_boxes[i]
                        split_O_boxes = self.O_boxes[i]
                        split_gt_class_HO = self.gt_class_HO[i]
                        split_H_num = self.H_num[i]

                        self.net.set_ph(split_image, split_image_id, split_H_num,
                                        split_H_boxes, split_O_boxes, split_gt_class_HO, split_spatial)

                        # Build the main computation graph
                        layers = self.net.create_architecture(True)  # is_training flag: True

                        O_features.append(self.net.intermediate['fc7_O'][:self.net.get_compose_num_stop()])
                        V_features.append(self.net.intermediate['fc7_verbs'][:self.net.get_compose_num_stop()])
                        num_stop_list.append(self.net.get_compose_num_stop())
                        # Define the loss
                        loss = layers['total_loss']

                        tower_losses.append(loss)
                        # variables = tf.trainable_variables()
                        # grads_and_vars = self.optimizer.compute_gradients(loss, variables)
                        # tower_grads.append(grads_and_vars)

                        if i == 1 and not self.net.model_name.__contains__('_base'):
                            print('compose learning ======================================')
                            new_loss = self.compose_feature_helper.merge_generate(O_features, V_features,
                                                                              [self.gt_class_HO[j][
                                                                               :num_stop_list[j]] for j in
                                                                               range(2)],
                                                                              compose_type)
                            new_loss = new_loss['vcl_loss']
                            ll = self.compose_feature_helper.get_ll()
                            tower_losses.append(new_loss * ll)

                            variables = tf.trainable_variables()
                            grads_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(tower_losses), variables)
                            tower_grads.append(grads_and_vars)
                        elif i == 1:
                            # baseline
                            new_loss = tf.reduce_sum(tower_losses)
                            variables = tf.trainable_variables()
                            grads_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(tower_losses), variables)

            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]

            for grad, var in capped_gvs:
                print('update: {}'.format(var.name))
            train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)
            tf.summary.scalar('lr', lr)
            tf.summary.scalar('merge_loss', new_loss)
            self.net.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
        return lr, train_op, tf.reduce_sum(tower_losses)


    def train_model(self, sess, max_iters):
        if 'CUDA_VISIBLE_DEVICES' not in os.environ or len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1:
            lr, train_op, t_loss = self.construct_graph2(sess)
        else:
            lr, train_op, t_loss = self.construct_graph2(sess)
        self.from_snapshot(sess)
        
        sess.graph.finalize()

        timer = Timer()
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        # Data_length = len(self.Trainval_GT)
        iter = self.get_init_step()
        while iter < max_iters + 1:
            timer.tic()

            blobs = {}
            from tensorflow.python.framework.errors_impl import InvalidArgumentError
            if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):

                # Compute the graph with summary
                total_loss, summary, image_id, _ = sess.run([t_loss,
                                                       self.net.summary_op, self.net.image_id,
                                                       train_op])
                self.writer.add_summary(summary, float(iter))

            else:
                # Compute the graph without summary
                total_loss, image_id, _ = sess.run([t_loss, self.net.image_id,
                                                       train_op])

            timer.toc()
            # print(image_id)
            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                if type(image_id) == tuple:
                    image_id = image_id[0]
                logger.info('iter: {:d} / {:d}, im_id: {:d}, total loss: {:.6f}, lr: {:f}, speed: {:.3f} s/iter'.format(
                      iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time))
            # Snapshotting
            self.snapshot(sess, iter)

            iter += 1

        self.writer.close()
