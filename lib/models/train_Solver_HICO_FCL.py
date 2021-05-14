

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from models.train_Solver_HICO import SolverWrapper
from ult.config import cfg
from ult.timer import Timer
from ult.ult import get_epoch_iters


class SolverWrapperFCL(SolverWrapper):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, Trainval_GT, Trainval_N, output_dir, tbdir, Pos_augment, Neg_select, Restore_flag,
                 pretrained_model):
        super(SolverWrapperFCL, self).__init__(sess, network, Trainval_GT, Trainval_N, output_dir, tbdir, Pos_augment,
                                               Neg_select, Restore_flag, pretrained_model)

    def construct_graph(self, sess):
        print("construct_graph======================")
        with sess.graph.as_default():

            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            # Build the main computation graph
            layers = self.net.create_architecture(True)  # is_training flag: True

            step_factor = self.get_step_factor()

            with tf.variable_scope('tmp', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable('global_step', shape=(), trainable=False,
                                              initializer=tf.constant_initializer(0))
                global_step1 = tf.get_variable('global_step1', shape=(), trainable=False,
                                               initializer=tf.constant_initializer(0))

            lr, self.optimizer = self.get_optimzer_lr(global_step, step_factor)
            lr1, self.optimizer_g = self.get_optimzer_lr(global_step1, step_factor)

            capped_gvs = self.get_all_grads(layers)
            train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Generator
            g_loss, g_capped_gvs = self.get_generator_grads(layers, self.optimizer_g)
            train_op_g = self.optimizer_g.apply_gradients(g_capped_gvs, global_step=global_step1)

            tf.summary.scalar('lr', lr)
            self.net.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)

        return lr, train_op, train_op_g, layers['total_loss'] + layers['fake_total_loss'], g_loss

    def get_main_grads(self, layers):
        capped_gvs = []
        if 'total_loss' in layers:
            loss = layers['total_loss']
            varaibles = [v for v in tf.trainable_variables() if not v.name.__contains__('generator')]
            grads_and_vars = self.optimizer.compute_gradients(loss, varaibles)
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]

        return capped_gvs

    def get_main_all_grads(self, layers):
        capped_gvs = []
        loss = layers['total_loss'] + layers['fake_total_loss']
        varaibles = [v for v in tf.trainable_variables() if not v.name.__contains__('generator')]
        grads_and_vars = self.optimizer.compute_gradients(loss, varaibles)
        capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]

        return capped_gvs

    def get_all_grads(self, layers):
        capped_gvs = []
        loss = layers['total_loss'] + layers['fake_total_loss']
        varaibles = [v for v in tf.trainable_variables()]
        grads_and_vars = self.optimizer.compute_gradients(loss, varaibles)
        capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]

        return capped_gvs

    def get_classification_grads(self, layers):
        loss = layers['total_loss'] + layers['fake_total_loss'] + layers['fake_D_total_loss']
        variables = self.get_classifier_variables()
        grads_and_vars = self.optimizer.compute_gradients(loss, variables)
        capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]
        for grad, var in capped_gvs:
            print('d update: {}'.format(var.name))
        return loss, capped_gvs

    def get_generator_grads(self, layers, optimizer_g):
        # Generator/Fabricator
        fake_loss = layers['fake_G_total_loss']
        print('fake loss ==========', fake_loss)
        g_update_variables = [v for v in tf.trainable_variables() if v.name.__contains__('generator')]
        print(g_update_variables)
        g_grads_and_vars = optimizer_g.compute_gradients(fake_loss, g_update_variables)
        g_capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in g_grads_and_vars if grad is not None]
        for grad, var in g_capped_gvs:
            print('g update: {}'.format(var.name))
        return fake_loss, g_capped_gvs

    def train_model_stepwise_inner(self, D_loss, g_loss, iter, lr, max_iters, sess, timer, train_op, train_op_g):
        while iter < max_iters + 1:
            timer.tic()

            total_loss = 0
            fake_total_loss = 0
            #
            save_iters = 50000
            epoch_stride = 0
            if self.net.model_name.__contains__('_s1_'):
                # This is for fine-tuning the fabricator in step-wise optimization
                epoch_stride = 1
            save_iters = get_epoch_iters(self.net.model_name)

            if iter < save_iters * epoch_stride:
                if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):
                    # Compute the graph with summary
                    fake_total_loss, _, summary, image_id = sess.run(
                        [g_loss, train_op_g, self.net.summary_op, self.net.image_id, ])

                    # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                    self.writer.add_summary(summary, float(iter))
                else:
                    # Compute the graph without summary
                    fake_total_loss, _, image_id = sess.run([g_loss, train_op_g, self.net.image_id, ])
            else:
                if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):

                    # Compute the graph with summary
                    total_loss, _, summary, image_id = sess.run(
                        [D_loss, train_op, self.net.summary_op, self.net.image_id, ])

                    # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                    # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                    self.writer.add_summary(summary, float(iter))

                else:
                    # Compute the graph without summary
                    total_loss, _, image_id = sess.run([D_loss, train_op, self.net.image_id, ])

            timer.toc()
            # print(image_id)
            # Display training information
            if iter % cfg.TRAIN.DISPLAY == 0:
                if type(image_id) == tuple:
                    image_id = image_id[0]
                print('iter: {:d} / {:d}, im_id: {:d}, loss: {:.6f}, G: {:.6f} lr: {:f}, speed: {:.3f} s/iter'.format(
                    iter, max_iters, image_id, total_loss, fake_total_loss, lr.eval(), timer.average_time), end='\n',
                    flush=True)
            # Snapshotting
            t_iter = iter
            self.snapshot(sess, t_iter)

            iter += 1

    def train_model(self, sess, max_iters):
        print('train: ', self.net.model_name)
        lr, train_op, train_op_g, D_loss, g_loss = self.construct_graph(sess)
        self.from_snapshot(sess)

        sess.graph.finalize()

        timer = Timer()

        # Data_length = len(self.Trainval_GT)
        iter = 0

        self.train_model_stepwise_inner(D_loss, g_loss, iter, lr, max_iters, sess, timer, train_op, train_op_g)
        self.writer.close()

def train_net(network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag,
              pretrained_model, max_iters=300000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    tfconfig = tf.ConfigProto(device_count={"CPU": 32},
                              inter_op_parallelism_threads=16,
                              intra_op_parallelism_threads=16)
    # tfconfig = tf.ConfigProto()
    tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapperFCL(sess, network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select,
                              Restore_flag, pretrained_model)

        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(
            Neg_select) + ', Restore_flag = ' + str(Restore_flag))
        sw.train_model(sess, max_iters)
        print('done solving')
