"""
This is from VCL
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from models.train_Solver_VCOCO import SolverWrapper
from ult.config import cfg
from ult.timer import Timer


class VCOCOSolverWrapperCL(SolverWrapper):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, output_dir, tbdir, Restore_flag, pretrained_model):
        super(VCOCOSolverWrapperCL, self).__init__(sess, network, None, None, output_dir, tbdir, 0, 0, 0, Restore_flag, pretrained_model)

        self.image = None

        self.image_id = None
        self.spatial = None
        self.H_num  = None
        self.blobs =  None

    def set_coco_data(self, image, image_id, H_num, blobs):

        if image is not None: self.image       = image
        if image_id is not None: self.image_id = image_id
        self.blobs = blobs
        if H_num is not None: self.H_num       = H_num

    def construct_graph2(self, sess):
        print("construct_graph2")
        compose_type = self.compose_feature_helper.get_compose_type()
        with sess.graph.as_default(), tf.device('/cpu:0'):
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            init_step = self.get_init_step()

            global_step = tf.Variable(init_step, trainable=False, name='global_step')

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
                with tf.device('/gpu:%d' % gpu_idx):
                    with tf.name_scope('%s_%d' % ('HICO', i), ) as scope:
                        split_image = self.image[i]
                        split_image_id = self.image_id[i]
                        split_H_num = self.H_num[i]
                        blobs = self.blobs[i]
                        print(i , split_H_num, '----------------------')
                        self.net.set_ph(split_image, split_image_id, split_H_num, blobs['sp'], blobs['H_boxes'],
                                        blobs['O_boxes'], blobs['gt_class_H'], blobs['gt_class_HO'],
                                        blobs['gt_class_sp'], blobs['Mask_HO'], blobs['Mask_H'], blobs['Mask_sp'],
                                        blobs['gt_class_C'])

                        # Build the main computation graph
                        layers = self.net.create_architecture(True)  # is_training flag: True

                        O_features.append(self.net.intermediate['fc7_O'][:self.net.get_compose_num_stop()])
                        V_features.append(self.net.intermediate['fc7_verbs'][:self.net.get_compose_num_stop()])

                        num_stop_list.append(self.net.get_compose_num_stop())
                        print('num stop:', self.net.get_compose_num_stop(), num_stop_list)
                        # Define the loss
                        loss = layers['total_loss']
                        if not (self.net.model_name.__contains__('atl') and i == 1):
                            tower_losses.append(loss)
                        # variables = tf.trainable_variables()
                        # grads_and_vars = self.optimizer.compute_gradients(loss, variables)
                        # tower_grads.append(grads_and_vars)
                        # tf.get_variable_scope().reuse_variables()
                        if i == 1:
                            # with tf.device('/cpu:0'):
                            #     print(O_features[0] == O_features[1], O_features)
                            #     O_features[1] = tf.Print(O_features[1], [self.image_id, self.H_num, "res:",
                            #                                              tf.shape(O_features[0]),tf.shape(O_features[1]),
                            #                                              tf.shape(V_features[0]),tf.shape(V_features[1]),
                            #                                              tf.shape(split_gt_class_HO), num_stop_list[0],
                            #                                              num_stop_list[1]], '1dddmessage:',
                            #                              first_n=10000)
                            if not self.net.model_name.__contains__('base'):
                                key = 'gt_class_C'
                                if self.net.model_name.__contains__('atl'):
                                    new_loss = self.compose_feature_helper.merge_generate(O_features, V_features,
                                                                                          [self.blobs[j][key][
                                                                                           :num_stop_list[j]] for j in
                                                                                           range(2)],
                                                                                          'atl')
                                    new_loss = new_loss['vcl_loss']
                                    # atl1 means we jointly composing HOI images and object images
                                else:
                                    new_loss = self.compose_feature_helper.merge_generate(O_features, V_features,
                                                                                      [self.blobs[j][key][
                                                                                       :num_stop_list[j]] for j in
                                                                                       range(2)],
                                                                                      compose_type)
                                    new_loss = new_loss['vcl_loss']
                                ll = self.compose_feature_helper.get_ll()
                                tower_losses.append(new_loss * ll)
                            variables = tf.trainable_variables()
                            grads_and_vars = self.optimizer.compute_gradients(tf.reduce_sum(tower_losses), variables)
                            tower_grads.append(grads_and_vars)

            if self.net.model_name.__contains__('base') or self.net.model_name.__contains__('atl'):
                assert len(tower_losses) == 2, tower_losses
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars if grad is not None]

            # self.addition_loss(capped_gvs, layers)

            # for grad, var in capped_gvs:
            #     print('update: {}'.format(var.name))
            train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)
            tf.summary.scalar('lr', lr)
            tf.summary.scalar('merge_loss', tf.reduce_sum(tower_losses))
            self.net.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
        return lr, train_op, tf.reduce_sum(tower_losses)


    def snapshot(self, sess, iter):

        snapshot_iters = cfg.TRAIN.SNAPSHOT_ITERS
        if (iter % snapshot_iters == 0 and iter != 0):
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Store the model snapshot
            filename = 'HOI' + '_iter_{:d}'.format(iter) + '.ckpt'
            filename = os.path.join(self.output_dir, filename)

            # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     print(v.name)
            self.saver.save(sess, filename)
            print('Wrote snapshot to: {:s}'.format(filename), iter / snapshot_iters)


    def train_model(self, sess, max_iters):

        lr, train_op, t_loss = self.construct_graph2(sess)
        self.from_snapshot(sess)
        
        sess.graph.finalize()

        timer = Timer()
        
        # Data_length = len(self.Trainval_GT)
        iter = self.get_init_step()
        while iter < max_iters + 1:
            timer.tic()

            blobs = {}
            # blobs = Get_Next_Instance_HO_Neg_HICO(self.Trainval_GT, self.Trainval_N, iter, self.Pos_augment, self.Neg_select, Data_length)
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
                print('iter: {:d} / {:d}, im_id: {:d}, total loss: {:.6f}, lr: {:f}, speed: {:.3f} s/iter'.format(
                      iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time), end='\r', flush=True)

            self.snapshot(sess, iter)

            iter += 1

        self.writer.close()
