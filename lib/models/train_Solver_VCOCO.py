# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from networks.VCL import VCL
from ult.config import cfg
from ult.timer import Timer
import os
import tensorflow as tf


class SolverWrapper(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, Trainval_GT, Trainval_N, output_dir, tbdir, Pos_augment, Neg_select, iCAN_Early_flag, Restore_flag, pretrained_model):

        self.net               = network
        self.Trainval_GT       = Trainval_GT
        self.Trainval_N        = Trainval_N
        self.output_dir        = output_dir
        self.tbdir             = tbdir
        self.Pos_augment       = Pos_augment
        self.Neg_select        = Neg_select
        self.iCAN_Early_flag   = iCAN_Early_flag
        self.Restore_flag      = Restore_flag
        self.pretrained_model  = pretrained_model

        self.compose_feature_helper = VCL(network)

    def get_optimzer_lr(self, global_step, step_factor):
        stepsize = int(cfg.TRAIN.STEPSIZE * step_factor)
        gamma = cfg.TRAIN.GAMMA
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step, stepsize,
                                        gamma, staircase=True)
        optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

        if self.net.model_name.__contains__('VCOCO') and self.net.model_name.__contains__('test'):
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE*10, global_step,
                                            int(cfg.TRAIN.STEPSIZE) * 2,
                                            gamma, staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

        return lr, optimizer

    def get_step_factor(self):
        step_factor = 5
        return step_factor

    def snapshot(self, sess, iter):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'HOI' + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def construct_graph(self, sess):
        with sess.graph.as_default():
      
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            # Build the main computation graph
            layers = self.net.create_architecture(True) # is_training flag: True

            # Define the loss
            loss = layers['total_loss']

            init_step = self.get_init_step()
            global_step = tf.Variable(init_step, trainable=False)
            lr             = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, cfg.TRAIN.STEPSIZE, cfg.TRAIN.GAMMA, staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            grads_and_vars = self.optimizer.compute_gradients(loss, tf.trainable_variables())
            capped_gvs     = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]
            
            train_op = self.optimizer.apply_gradients(capped_gvs,global_step=global_step)
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)

        return lr, train_op

    def get_init_step(self):
        if self.Restore_flag == -1:
            ckpt = tf.train.get_checkpoint_state(self.output_dir)
            print(ckpt.model_checkpoint_path)
            init_step = ckpt.model_checkpoint_path.split('/')[- 1].split('_')[- 1]
            init_step = int(init_step.replace('.ckpt', ''))
            print("Init Step:", init_step)
        else:
            init_step = 0
        return init_step

    def from_snapshot(self, sess):
        if self.Restore_flag == -1:
            ckpt = tf.train.get_checkpoint_state(self.output_dir)
            # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restoring model snapshots from {:s}'.format(ckpt.model_checkpoint_path))
        if self.Restore_flag == 0:
            saver_t = [var for var in tf.model_variables() if 'conv1' in var.name and 'conv1_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv2' in var.name and 'conv2_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv3' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv4' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv5' in var.name]
            saver_t += [var for var in tf.model_variables() if 'shortcut' in var.name]

            sess.run(tf.global_variables_initializer())
            # for var in tf.trainable_variables():
            #     print(var.name, var.eval().mean())

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))

            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

            # for var in tf.trainable_variables():
            #     print(var.name, var.eval().mean())

        if self.Restore_flag == 5 or self.Restore_flag == 6 or self.Restore_flag == 7:

            sess.run(tf.global_variables_initializer())
            for var in tf.trainable_variables():
                print(var.name, var.eval().mean())

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
            saver_t = {}
            if self.net.model_name.__contains__('res101'):
                model_backbone = 'resnet_v1_101'
            else:
                model_backbone = 'resnet_v1_50'

            # Add block0
            for ele in tf.model_variables():
                if model_backbone + '/conv1/weights' in ele.name or model_backbone + '/conv1/BatchNorm/beta' in ele.name or model_backbone + '/conv1/BatchNorm/gamma' in ele.name or model_backbone + '/conv1/BatchNorm/moving_mean' in ele.name or model_backbone + '/conv1/BatchNorm/moving_variance' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            # Add block1
            for ele in tf.model_variables():
                if 'block1' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            # Add block2
            for ele in tf.model_variables():
                if 'block2' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            # Add block3
            for ele in tf.model_variables():
                if 'block3' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            # Add block4
            for ele in tf.model_variables():
                if 'block4' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

            if self.Restore_flag >= 5:

                saver_t = {}
                # Add block5
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = \
                            [var for var in tf.model_variables() if
                             ele.name[:-2].replace('block4', 'block5') in var.name][
                                0]

                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

            if self.Restore_flag >= 6:
                saver_t = {}
                # Add block6
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = \
                            [var for var in tf.model_variables() if
                             ele.name[:-2].replace('block4', 'block6') in var.name][
                                0]

                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

            if self.Restore_flag >= 7:

                saver_t = {}
                # Add block7
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = \
                            [var for var in tf.model_variables() if
                             ele.name[:-2].replace('block4', 'block7') in var.name][
                                0]

                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)


    def train_model_tf(self, sess, max_iters):

        lr, train_op = self.construct_graph(sess)
        self.from_snapshot(sess)

        sess.graph.finalize()

        timer = Timer()

        # Data_length = len(self.Trainval_GT)
        iter = self.get_init_step()

        while iter < max_iters + 1:

            timer.tic()

            blobs = {}
            from tensorflow.python.framework.errors_impl import InvalidArgumentError
            try:
                if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):

                    # Compute the graph with summary
                    total_loss, image_id, summary = self.net.train_step_tfr_with_summary(sess, blobs, lr.eval(),
                                                                                         train_op)
                    self.writer.add_summary(summary, float(iter))

                else:
                    # Compute the graph without summary
                    total_loss, image_id = self.net.train_step_tfr(sess, blobs, lr.eval(), train_op)
            except InvalidArgumentError as e:
                print('InvalidArgumentError')
                image_id = -1
                total_loss = 0
                if self.net.model_name.__contains__('lamb'):
                    print('InvalidArgumentError', image_id)
                else:
                    raise e
            timer.toc()
            # print(image_id)
            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                if type(image_id) == tuple:
                    image_id = image_id[0]
                out_str = 'iter: %d / %d, im_id: %u, total loss: %.6f, lr: %f, speed: %.3f s/iter' % \
                          (iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time)
                print(out_str, end='\r', flush=True)


            # Snapshotting
            if (iter % cfg.TRAIN.SNAPSHOT_ITERS == 0 and iter != 0) or (iter == 10):
                # self.net.test_
                self.snapshot(sess, iter)

            iter += 1

        self.writer.close()

                                                                            
def train_net(network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, iCAN_Early_flag, Restore_flag, pretrained_model, max_iters=300000):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    if Restore_flag >= 0:
        # Remove previous events
        filelist = [ f for f in os.listdir(tb_dir)]
        for f in filelist:
            os.remove(os.path.join(tb_dir, f))
        # Remove previous snapshots

    tfconfig = tf.ConfigProto(device_count={"CPU": 32},
                              inter_op_parallelism_threads=16,
                              intra_op_parallelism_threads=16)
    # tfconfig = tf.ConfigProto()
    tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, iCAN_Early_flag, Restore_flag, pretrained_model)
        
        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(Neg_select) + ', iCAN_Early_flag = ' + str(iCAN_Early_flag) + ', Restore_flag = ' + str(Restore_flag))
        sw.train_model_tf(sess, max_iters)
        print('done solving')
