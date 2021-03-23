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
from tensorflow.python import pywrap_tensorflow

from ult.ult import get_epoch_iters


class SolverWrapper(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, Trainval_GT, Trainval_N, output_dir, tbdir, Pos_augment, Neg_select, Restore_flag, pretrained_model):

        self.net               = network
        self.Trainval_GT       = Trainval_GT
        self.Trainval_N        = Trainval_N
        self.output_dir        = output_dir
        self.tbdir             = tbdir
        self.Pos_augment       = Pos_augment
        self.Neg_select        = Neg_select
        self.Restore_flag      = Restore_flag
        self.pretrained_model  = pretrained_model

        self.compose_feature_helper = VCL(network)

    def snapshot(self, sess, iter):
        if self.net.model_name.__contains__('multi'):
            snapshot_iters = cfg.TRAIN.SNAPSHOT_ITERS * 5 // 2
        else:
            snapshot_iters = cfg.TRAIN.SNAPSHOT_ITERS * 5

        if (iter + 1) % snapshot_iters == 0 and iter != 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Store the model snapshot
            filename = 'HOI' + '_iter_{:d}'.format(iter + 1) + '.ckpt'
            filename = os.path.join(self.output_dir, filename)

            # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     print(v.name)
            self.saver.save(sess, filename)
            print('Wrote snapshot to: {:s}'.format(filename), iter / snapshot_iters)

    def get_step_factor(self):
        step_factor = 5
        if self.net.model_name.__contains__('multi'):
            step_factor = 2.5
        else:
            step_factor = 5
        return step_factor

    def get_classifier_variables(self):
        variables = [v for v in tf.trainable_variables()
                     if v.name.__contains__('Concat_verbs')
                     or v.name.__contains__('fc7_verbs')
                     or v.name.__contains__('classification/cls_score_verbs')]
        return variables


    def construct_graph(self, sess):
        with sess.graph.as_default():
      
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            # Build the main computation graph
            layers = self.net.create_architecture(True) # is_training flag: True

            # Define the loss

            init_step = self.get_init_step()

            global_step = tf.Variable(init_step, trainable=False, name='global_step')

            step_factor = self.get_step_factor()
            lr, self.optimizer = self.get_optimzer_lr(global_step, step_factor)
            capped_gvs = []

            loss = 0

            if 'total_loss' in layers:
                loss = layers['total_loss'] + loss
                for v in tf.trainable_variables():
                    print('varaibles:', v)
                grads_and_vars = self.optimizer.compute_gradients(loss, tf.trainable_variables())
                capped_gvs     = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]

            train_op = self.optimizer.apply_gradients(capped_gvs,global_step=global_step)
            tf.summary.scalar('lr', lr)
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)

        return lr, train_op

    def get_optimzer_lr(self, global_step, step_factor):
        gamma = cfg.TRAIN.GAMMA
        epoch_iters = get_epoch_iters(self.net.model_name)
        stepsize = epoch_iters * 2

        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step, stepsize,
                                        gamma, staircase=True)
        optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
        if self.net.model_name.__contains__('cosine'):
            print('cosine =========')
            first_decay_steps = epoch_iters*10 # 2 epoches
            from tensorflow.python.training.learning_rate_decay import cosine_decay_restarts
            lr = cosine_decay_restarts(cfg.TRAIN.LEARNING_RATE * 10, global_step, first_decay_steps, t_mul=2.0,
                                       m_mul=0.9, alpha=cfg.TRAIN.LEARNING_RATE*0.1)
            optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
        elif self.net.model_name.__contains__('zsrare'): #rare first
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step,
                                            int(cfg.TRAIN.STEPSIZE * 2),
                                            gamma, staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
        elif self.net.model_name.__contains__('zsnrare'): # non rare first
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step, int(cfg.TRAIN.STEPSIZE * step_factor),
                                            gamma, staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
        return lr, optimizer

    def get_init_step(self):
        if self.Restore_flag == -1:
            ckpt = tf.train.get_checkpoint_state(self.output_dir)
            init_step = ckpt.model_checkpoint_path.split('/')[- 1].split('_')[- 1]
            init_step = int(init_step.replace('.ckpt', ''))
            print("Init Step:", init_step)
        else:
            init_step = 0
        return init_step

    def switch_checkpoint_path(self, model_checkpoint_path):
        head = model_checkpoint_path.split('Weights')[0]
        model_checkpoint_path = model_checkpoint_path.replace(head, cfg.LOCAL_DATA +'/')
        return model_checkpoint_path

    def from_snapshot(self, sess):
        # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print(v.name)
        if self.Restore_flag == -7:
            restore_dirs = '/'  # TODO set your pretrained model
            print('restore_from', restore_dirs)
            ckpt = tf.train.get_checkpoint_state(restore_dirs)
            sess.run(tf.global_variables_initializer())
            variables = [v for v in tf.global_variables() if not v.name.__contains__('Momentum')]

            for v in variables:
                print('snapshot:', v)
            saver = tf.train.Saver(variables)
            ckpt.model_checkpoint_path = self.switch_checkpoint_path(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restoring model snapshots from {:s}'.format(ckpt.model_checkpoint_path))
        elif self.Restore_flag == -1:
            ckpt = tf.train.get_checkpoint_state(self.output_dir)
            saver = tf.train.Saver()
            saver.restore(sess, self.switch_checkpoint_path(ckpt.model_checkpoint_path))
            print('Restoring model snapshots from {:s}'.format(ckpt.model_checkpoint_path))


        if self.Restore_flag == 0:

            saver_t  = [var for var in tf.model_variables() if 'conv1' in var.name and 'conv1_sp' not in var.name]
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
            # for var in tf.trainable_variables():
            #     print(var.name, var.eval().mean())
            if self.net.model_name.__contains__('res101'):
                model_backbone = 'resnet_v1_101'
            else:
                model_backbone = 'resnet_v1_50'
            
            print('Restoring model snapshots from {:s}'.format(self.pretrained_model), model_backbone)
            saver_t = {}
            
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
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block5') in var.name][0]
         
                
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)
            

            if self.Restore_flag >= 6:
                saver_t = {}
                # Add block6
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block6') in var.name][0]
         
                
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)
                
            if self.Restore_flag >= 7:

                saver_t = {}
                # Add block7
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block7') in var.name][0]
         
            
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

    def train_model(self, sess, max_iters):
        lr, train_op = self.construct_graph(sess)
        self.from_snapshot(sess)
        
        sess.graph.finalize()

        timer = Timer()
        
        # Data_length = len(self.Trainval_GT)
        iter = self.get_init_step()
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        while iter < max_iters + 1:
            timer.tic()

            blobs = {}

            from tensorflow.python.framework.errors_impl import InvalidArgumentError
            if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):

                # Compute the graph with summary
                total_loss, image_id, summary = self.net.train_step_tfr_with_summary(sess, blobs, lr, train_op)
                # total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                self.writer.add_summary(summary, float(iter))

            else:
                # Compute the graph without summary
                total_loss, image_id = self.net.train_step_tfr(sess, blobs, lr, train_op)


            timer.toc()
            # print(image_id)
            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                if type(image_id) == tuple:
                    image_id = image_id[0]
                logger.info('iter: {:d} / {:d}, im_id: {:d}, total loss: {:.6f}, lr: {:f}, speed: {:.3f} s/iter'.format(
                    iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time))
            # Snapshotting
            t_iter = iter
            self.snapshot(sess, t_iter)

            iter += 1

        self.writer.close()



def train_net(network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model, max_iters=300000):
    
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
        sw = SolverWrapper(sess, network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model)
        
        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(Neg_select) + ', Restore_flag = ' + str(Restore_flag), max_iters)
        sw.train_model(sess, max_iters)
        print('done solving')
