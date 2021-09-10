# --------------------------------------------------------
# Tensorflow FCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# --------------------------------------------------------

import tensorflow as tf
import os
from ult.config import cfg
from networks.tools import get_word2vec



class Fabricator(object):
    """
    this class is the Fabricator for generating object features.
    """
    def __init__(self, network, ):
        self.network = network
        self.obj_num_classes = self.network.obj_num_classes
        self.verb_num_classes = self.network.verb_num_classes
        if self.network.model_name.__contains__('var'):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                self.obj_identity_variable = tf.get_variable('identity_variable', [self.network.obj_num_classes, 2048], initializer=tf.random_normal_initializer, trainable=True)
                # if self.network.model_name.__contains__('var1') or self.network.model_name.__contains__('varv'):
                #     self.verb_identity_variable = tf.get_variable('identity_verb_variable', [self.network.verb_num_classes, 2048], initializer=tf.random_normal_initializer, trainable=True)
        elif self.network.model_name.__contains__('_ohot'):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                import numpy as np
                # example code
                self.obj_onehot_variable = tf.constant(np.eye(self.obj_num_classes), tf.float32)
        pass

    def get_obj_identity_variable(self):
        # if self.network.model_name.__contains__('_ohot'):
        #     print('onehot')
        #     return self.obj_onehot_variable
        if self.network.model_name.__contains__('_var_gan'):
            return self.obj_identity_variable
        elif self.network.model_name.__contains__('_varl_gan'):
            return self.obj_identity_variable
        else:
            print('variable')
            raise Exception("wrong model name")

    def convert_emb_feats(self, word2vec, verbs, target_dims=2048, type=0):
        # this is for ablation study
        import tensorflow.contrib.slim as slim

        with tf.variable_scope('generator'):
            print('g====================================================', word2vec)
            print(word2vec, verbs, )
            if type == 0:
                noise = tf.random_normal(shape=tf.shape(verbs), mean=0.0, stddev=1., dtype=tf.float32)
                v = tf.concat([word2vec, verbs, noise], axis=-1)
            elif type == 3:
                # won_
                v = tf.concat([word2vec, verbs, tf.zeros_like(verbs, dtype=tf.float32)], axis=-1)
            elif type == 8:
                # won1
                v = tf.concat([word2vec, verbs], axis=-1)
            elif type == 4:
                noise = tf.random_normal(shape=tf.shape(verbs), mean=0.0, stddev=1., dtype=tf.float32)
                noise = tf.nn.relu(noise)
                v = tf.concat([word2vec, verbs, noise], axis=-1)
            elif type == 2:
                noise = tf.random_normal(shape=tf.shape(verbs), mean=0.0, stddev=1., dtype=tf.float32)
                v = tf.concat([word2vec, word2vec, noise], axis=-1)
            elif type == 5:
                noise = tf.random_normal(shape=tf.shape(verbs), mean=0.0, stddev=1., dtype=tf.float32)
                v = tf.concat([word2vec, tf.zeros_like(verbs, dtype=tf.float32), noise], axis=-1)
            elif type == 6:
                noise = tf.random_normal(shape=tf.shape(verbs), mean=0.0, stddev=1., dtype=tf.float32)
                v = tf.concat([tf.zeros_like(verbs, dtype=tf.float32), verbs, noise], axis=-1)

            v = slim.fully_connected(v, 2048, reuse=tf.AUTO_REUSE, scope='fc3')
            obj = slim.fully_connected(v, target_dims,
                                       reuse=tf.AUTO_REUSE, scope='fc4')
        return obj

    def get_variable_by_all(self, fc7_verbs, fc7_objects, v_gt_class_HO):
        import numpy as np
        tmp1 = np.ones(self.network.compose_num_classes)
        word2vec_list, new_fc7_verbs, new_fc7_objects, tmp_ho_class = self.get_variable_by_cls_index(v_gt_class_HO, fc7_verbs,
                                                                                      fc7_objects,
                                                                                      tf.constant(tmp1))
        return word2vec_list, new_fc7_verbs, new_fc7_objects, tmp_ho_class


    def get_variable_by_cls_index(self, v_gt_class_HO, fc7_verbs, fc7_objects, cls_index):

        conds = self.extract_ho_conds_by_cls_index(cls_index, v_gt_class_HO)
        # tmp_ho_index = tf.reshape(tmp_ho_index, [-1])
        tmp_ho_class = tf.boolean_mask(v_gt_class_HO, conds)
        # tmp_ho_class = tf.gather(v_gt_class_HO, tmp_ho_index, axis=0)
        word2vec_list = tf.gather(self.get_obj_identity_variable(),
                                  tf.math.argmax(self.network.gt_obj_class[:self.network.get_num_stop()], axis=-1))

        word2vec_list = tf.boolean_mask(word2vec_list, conds)
        new_fc7_verbs = tf.boolean_mask(fc7_verbs, conds)
        new_fc7_objects = tf.boolean_mask(fc7_objects, conds)
        return word2vec_list, new_fc7_verbs, new_fc7_objects, tmp_ho_class

    def get_features_by_cls_index(self, v_gt_class_HO, fc7_O, fc7_verbs, cls_index):

        tmp_ho_index = self.extract_ho_index_by_cls_index(cls_index, v_gt_class_HO)
        tmp_ho_index = tf.reshape(tmp_ho_index, [-1])

        tmp_ho_class = tf.gather(v_gt_class_HO, tmp_ho_index)

        new_fc7_objects = tf.gather(fc7_O, tmp_ho_index, axis=0)
        new_fc7_verbs = tf.gather(fc7_verbs, tmp_ho_index, axis=0)

        return new_fc7_objects, new_fc7_verbs, tmp_ho_class

    def get_features_by_cls_index1(self, v_gt_class_HO, fc7_O, fc7_verbs, cls_index):

        conds = self.extract_ho_conds_by_cls_index(cls_index, v_gt_class_HO)
        tmp_ho_class = tf.boolean_mask(v_gt_class_HO, conds)

        new_fc7_objects = tf.boolean_mask(fc7_O, conds)
        new_fc7_verbs = tf.boolean_mask(fc7_verbs, conds)

        return new_fc7_objects, new_fc7_verbs, tmp_ho_class

    def extract_ho_index_by_cls_index(self, cls_index, v_gt_class_HO):
        tmp_ho_index = tf.reduce_sum(tf.multiply(v_gt_class_HO,
                                                 tf.expand_dims(tf.cast(cls_index, tf.float32), axis=0)), axis=-1)
        tmp_ho_index = tf.where(tmp_ho_index > 0.0)
        tmp_ho_index = tf.squeeze(tmp_ho_index)
        return tmp_ho_index

    def extract_ho_conds_by_cls_index(self, cls_index, v_gt_class_HO):
        tmp_ho_index = tf.reduce_sum(tf.multiply(v_gt_class_HO,
                                                 tf.expand_dims(tf.cast(cls_index, tf.float32), axis=0)), axis=-1)
        conds = tmp_ho_index > 0.0
        return conds

    def fabricate_model(self, fc7_O, fc7_O_raw, fc7_verbs, fc7_verbs_raw, initializer, is_training, v_gt_class_HO):
        if is_training and self.network.model_name.__contains__('_var_gan'):
            # this is helpful for visualization.
            fc7_O, fc7_verbs = self.var_fabricate_gen(fc7_O, fc7_O_raw, fc7_verbs, fc7_verbs_raw, initializer, is_training,
                                                      v_gt_class_HO)
        elif is_training and self.network.model_name.__contains__('_varl_gan'):
            # this has similar result to var_gan.
            fc7_O, fc7_verbs = self.var_fabricate_gen_lite(fc7_O, fc7_O_raw, fc7_verbs, fc7_verbs_raw, initializer, is_training,
                                                           v_gt_class_HO)
        else:
            # this has similar result to var_gan.
            fc7_O, fc7_verbs = self.var_fabricate_gen_lite(fc7_O, fc7_O_raw, fc7_verbs, fc7_verbs_raw, initializer,
                                                           is_training,
                                                           v_gt_class_HO)

        return fc7_O, fc7_verbs

    def obtain_last_verb_dim(self):
        last_dim = 2048
        return last_dim

    def get_ll(self):
        ll = 0.5
        return ll

    def obtain_gen_type(self):
        """
        This is for ablation study
        :return:
        """
        noise_type = 0
        if self.network.model_name.__contains__('_woa_'):
            # with verb
            noise_type = 2
        elif self.network.model_name.__contains__('_won_'):
            # no noise, with empty variable to keep dimension of FC unchanged
            noise_type = 3
        elif self.network.model_name.__contains__('_won1_'):
            # no noise
            noise_type = 8
        elif self.network.model_name.__contains__('_n1_'):
            # we use positive noise. Because the verb representation is after relu.
            # this is useless. I can not understand why I tried this.
            noise_type = 4
        elif self.network.model_name.__contains__('_woa1_'):
            # without verb, but we add a placeholder variable to make sure the dimension of FC unchanged.
            noise_type = 5
        elif self.network.model_name.__contains__('_woa2_'):
            # without verb, but we add a duplicate word embedding to make sure the dimension of FC unchanged.
            # However, this is a little bug because the dimensions of word embedding and Verb representation are different.
            noise_type = 7
        elif self.network.model_name.__contains__('_woo_'):
            # without object, this is for verb fabricator.
            noise_type = 6
        return noise_type

    def var_fabricate_gen(self, fc7_O, fc7_O_raw, fc7_verbs, fc7_verbs_raw, initializer, is_training, v_gt_class_HO):
        print('var_gan_gen ======================')
        if 'fake_total_loss' not in self.network.losses:
            self.network.losses['fake_total_loss'] = 0
        if 'fake_G_total_loss' not in self.network.losses:
            self.network.losses['fake_G_total_loss'] = 0
        # noise_type = self.obtain_gen_type()  # for ablated study
        noise_type = 0
        gan_fc7_O = fc7_O
        gan_fc7_verbs = fc7_verbs
        gan_v_gt_class_HO = v_gt_class_HO


        word2vec_list_G, new_fc7_verbs_G, new_fc7_objects_G, new_ho_class_G = \
            self.get_variable_by_all(gan_fc7_verbs, gan_fc7_O, gan_v_gt_class_HO)
        fake_obj_list = self.convert_emb_feats(word2vec_list_G, new_fc7_verbs_G, type= noise_type)
        # with tf.device('/cpu:0'):
        #     tf.summary.histogram('fake_obj_g', fake_obj_list)
        #     tf.summary.histogram('real_obj_g', new_fc7_objects_G)
        fc7_O_G = fake_obj_list
        fc7_verbs_G = new_fc7_verbs_G
        gt_class_HO_for_G_verbs = new_ho_class_G
        print('===========', fc7_verbs, gt_class_HO_for_G_verbs)

        # this does not work and even affects the performance.
        # tmp_fc7_O = tf.stop_gradient(new_fc7_objects_G)
        # ll = tf.losses.mean_squared_error(tmp_fc7_O, fake_obj_list)
        #
        # self.network.losses['disl_g'] = ll
        # self.network.losses['fake_G_total_loss'] += ll * 0.3

        cos_los = tf.losses.cosine_distance(tf.nn.l2_normalize(new_fc7_objects_G, axis=-1),
                                            tf.nn.l2_normalize(fake_obj_list, axis=-1), axis=-1)
        self.network.losses['cos_g'] = cos_los
        if self.network.model_name.__contains__('_cosg_'):
            self.network.losses['fake_G_total_loss'] += tf.where(cos_los > 0.15, cos_los, 0)


        tmp_gan_v_gt_class_HO = gan_v_gt_class_HO
        fc7_O_G = tf.concat([fc7_O, fc7_O_G], axis=0)
        fc7_verbs_G = tf.concat([fc7_verbs, fc7_verbs_G], axis=0)
        print('gan_v_gt_class_HO:', gan_v_gt_class_HO, gt_class_HO_for_G_verbs)
        gt_class_HO_for_G_verbs = tf.concat([tmp_gan_v_gt_class_HO, gt_class_HO_for_G_verbs], axis=0)

        self.network.set_gt_class_HO_for_G_verbs(gt_class_HO_for_G_verbs)
        fc7_vo = self.network.head_to_tail_ho(fc7_O_G, fc7_verbs_G, None, None, is_training, 'fc_HO')
        cls_prob_verbs = self.network.region_classification_ho(fc7_vo, is_training, initializer, 'classification',
                                                                nameprefix='fake_G_')


        # generate balanced objects
        obj_labels = tf.tile(tf.expand_dims(tf.one_hot(tf.range(self.obj_num_classes), self.obj_num_classes), axis=0), [tf.shape(gan_fc7_verbs)[0], 1, 1])
        obj_labels = tf.reshape(obj_labels, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, self.obj_num_classes])
        tmp_word2vec_list = tf.tile(tf.expand_dims(self.get_obj_identity_variable(), axis=0), [tf.shape(gan_fc7_verbs)[0], 1, 1])
        new_fc7_verbs_D_1 = tf.tile(tf.expand_dims(gan_fc7_verbs, axis=1), [1, self.obj_num_classes, 1])

        verbs_labels = tf.cast(tf.matmul(gan_v_gt_class_HO, self.network.verb_to_HO_matrix, transpose_b=True) > 0,
                                    tf.float32)
        verbs_labels = tf.tile(tf.expand_dims(verbs_labels, axis=1), [1, self.obj_num_classes, 1])
        verbs_labels = tf.reshape(verbs_labels, [tf.shape(gan_fc7_verbs)[0]*self.obj_num_classes, self.verb_num_classes])

        old_obj_labels = tf.cast(tf.matmul(gan_v_gt_class_HO, self.network.obj_to_HO_matrix, transpose_b=True) > 0,
                               tf.float32)
        old_obj_labels = tf.tile(tf.expand_dims(old_obj_labels, axis=1), [1, self.obj_num_classes, 1])
        old_obj_labels = tf.reshape(old_obj_labels, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, self.obj_num_classes])

        # construct_new ho_classes
        tmp_ho_class_from_obj = tf.greater(tf.matmul(obj_labels, self.network.obj_to_HO_matrix), 0.)
        tmp_ho_class_from_vb = tf.greater(tf.matmul(verbs_labels, self.network.verb_to_HO_matrix), 0.)
        new_gt_class_HO_1 = tf.cast(tf.logical_and(tmp_ho_class_from_vb, tmp_ho_class_from_obj), tf.float32)
        last_dim = self.obtain_last_verb_dim()
        obj_dim = self.obtain_last_obj_dim()
        new_fc7_verbs_D_1 = tf.reshape(new_fc7_verbs_D_1, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, last_dim])
        word2vec_list_D_1 = tf.reshape(tmp_word2vec_list, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, obj_dim])

        # all
        all_new_fc7_verbs_D_1, all_new_ho_class_D_1, all_word2vec_list_D_1 = self.conds_zeros(None, new_fc7_verbs_D_1, new_gt_class_HO_1, word2vec_list_D_1)

        # we also tried to add contrastive learning loss: the faked objects and its corresponding real objects. However, it is useless.

        new_fc7_verbs_D_1, new_ho_class_D_1, word2vec_list_D_1, _, _ = self.sample_instances(gan_fc7_verbs, new_fc7_verbs_D_1,
                                                                                       new_gt_class_HO_1,
                                                                                       word2vec_list_D_1, old_obj = old_obj_labels,
                                                                                       new_obj = obj_labels)


        fake_obj_list = self.convert_emb_feats(word2vec_list_D_1, new_fc7_verbs_D_1, type= noise_type)
        # if self.network.model_name.__contains__('_costive1'):
        #     self.add_contrastive_loss(fake_obj_list, new_ho_class_D_1, gan_fc7_O, gan_v_gt_class_HO)

        print(fake_obj_list, new_fc7_verbs_D_1, new_ho_class_D_1, '===========')
        self.cal_dax_loss(fake_obj_list, initializer, is_training, new_fc7_verbs_D_1, new_ho_class_D_1, nameprefix='fake_tmp_')

        ll = self.get_ll()
        self.network.losses['fake_total_loss'] += self.network.losses['fake_tmp_verbs_cross_entropy'] * ll

        gll = 1.
        self.network.losses['fake_G_total_loss'] += (self.network.losses['fake_tmp_verbs_cross_entropy'] * gll)

        with tf.device('/cpu:0'):
            fc7_O = tf.Print(fc7_O,
                             [1, tf.shape(fc7_O), tf.shape(fc7_verbs), tf.shape(word2vec_list_D_1),
                              tf.shape(new_ho_class_D_1),
                              tf.shape(new_fc7_verbs_D_1)],
                             "_dax shape:", first_n=100)

        word2vec_list_D, new_fc7_verbs_D, new_fc7_objects_D, new_ho_class_D = \
            self.get_variable_by_all(gan_fc7_verbs, gan_fc7_O, gan_v_gt_class_HO)


        fake_obj_list_D = self.convert_emb_feats(word2vec_list_D, new_fc7_verbs_D, type= noise_type)
        tmp_fc7_O = tf.stop_gradient(new_fc7_objects_D)

        # This is for mean squared loss. However it does not work.
        # We use this to illustrate the similarity between fake objects and real objects.
        ll = tf.losses.mean_squared_error(tmp_fc7_O, fake_obj_list_D)
        self.network.losses['disl_d'] = ll
        if self.network.model_name.__contains__('_disl_'):
            self.network.losses['fake_total_loss'] += tf.where(ll > 1., ll * 0.1, 0)

        tmp_cos_fc7_objs = new_fc7_objects_D
        cos_los = tf.losses.cosine_distance(tf.nn.l2_normalize(tmp_cos_fc7_objs, axis=-1),
                                            tf.nn.l2_normalize(fake_obj_list_D, axis=-1), axis=-1)
        self.network.losses['cos_d'] = cos_los
        self.network.losses['fake_total_loss'] += tf.where(cos_los > 0.2, cos_los, 0)

        fc7_O = tf.concat([fc7_O, fake_obj_list_D], axis=0)
        fc7_verbs = tf.concat([fc7_verbs, new_fc7_verbs_D], axis=0)
        gt_class_HO_for_D_verbs = tf.concat([v_gt_class_HO, new_ho_class_D], axis=0)

        with tf.device('/cpu:0'):
            fc7_O = tf.Print(fc7_O,
                             [tf.shape(fc7_O), tf.shape(fc7_verbs), tf.shape(gt_class_HO_for_D_verbs),
                              tf.shape(new_ho_class_D),
                              tf.shape(fake_obj_list_D), tf.shape(fc7_O), tf.shape(v_gt_class_HO)],
                             "D shape:", first_n=100)
        self.network.set_gt_class_HO_for_D_verbs(gt_class_HO_for_D_verbs)
        return fc7_O, fc7_verbs

    def var_fabricate_gen_lite(self, fc7_O, fc7_O_raw, fc7_verbs, fc7_verbs_raw, initializer, is_training, v_gt_class_HO):
        """
        This is similar to var_gan_gen. We just simplify the code.
        :param fc7_O:
        :param fc7_O_raw:
        :param fc7_verbs:
        :param fc7_verbs_raw:
        :param initializer:
        :param is_training:
        :param v_gt_class_HO:
        :return:
        """
        print('wemb_gan_gen ======================')
        if 'fake_total_loss' not in self.network.losses:
            self.network.losses['fake_total_loss'] = 0
        if 'fake_G_total_loss' not in self.network.losses:
            self.network.losses['fake_G_total_loss'] = 0
        noise_type = self.obtain_gen_type()

        gan_fc7_O = fc7_O
        gan_fc7_verbs = fc7_verbs
        gan_v_gt_class_HO = v_gt_class_HO

        # word2vec_list_G, new_fc7_verbs_G, new_fc7_objects_G, new_ho_class_G = \
        #         self.get_variable_by_all(gan_fc7_verbs, gan_fc7_O, gan_v_gt_class_HO)

        # fake_obj_list = self.convert_emb_feats(word2vec_list_G, new_fc7_verbs_G, type= noise_type)
        # with tf.device('/cpu:0'):
        #     tf.summary.histogram('fake_obj_g', fake_obj_list)
        #     tf.summary.histogram('real_obj_g', new_fc7_objects_G)
        # self.add_discriminator_loss(fake_obj_list, new_fc7_objects, initializer, is_training)
        # fc7_O_G = fake_obj_list
        # fc7_verbs_G = new_fc7_verbs_G
        # gt_class_HO_for_G_verbs = new_ho_class_G
        # print('===========', fc7_verbs, gt_class_HO_for_G_verbs)

        obj_labels = tf.tile(tf.expand_dims(tf.one_hot(tf.range(self.obj_num_classes), self.obj_num_classes), axis=0), [tf.shape(gan_fc7_verbs)[0], 1, 1])
        obj_labels = tf.reshape(obj_labels, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, self.obj_num_classes])
        tmp_word2vec_list = tf.tile(tf.expand_dims(self.get_obj_identity_variable(), axis=0), [tf.shape(gan_fc7_verbs)[0], 1, 1])
        new_fc7_verbs_D_1 = tf.tile(tf.expand_dims(gan_fc7_verbs, axis=1), [1, self.obj_num_classes, 1])

        verbs_labels = tf.cast(tf.matmul(gan_v_gt_class_HO, self.network.verb_to_HO_matrix, transpose_b=True) > 0,
                                    tf.float32)
        verbs_labels = tf.tile(tf.expand_dims(verbs_labels, axis=1), [1, self.obj_num_classes, 1])
        verbs_labels = tf.reshape(verbs_labels, [tf.shape(gan_fc7_verbs)[0]*self.obj_num_classes, self.verb_num_classes])

        old_obj_labels = tf.cast(tf.matmul(gan_v_gt_class_HO, self.network.obj_to_HO_matrix, transpose_b=True) > 0,
                               tf.float32)
        old_obj_labels = tf.tile(tf.expand_dims(old_obj_labels, axis=1), [1, self.obj_num_classes, 1])
        old_obj_labels = tf.reshape(old_obj_labels, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, self.obj_num_classes])

        # construct_new ho_classes
        tmp_ho_class_from_obj = tf.greater(tf.matmul(obj_labels, self.network.obj_to_HO_matrix), 0.)
        tmp_ho_class_from_vb = tf.greater(tf.matmul(verbs_labels, self.network.verb_to_HO_matrix), 0.)
        new_gt_class_HO_1 = tf.cast(tf.logical_and(tmp_ho_class_from_vb, tmp_ho_class_from_obj), tf.float32)
        last_dim = self.obtain_last_verb_dim()
        obj_dim = self.obtain_last_obj_dim()
        new_fc7_verbs_D_1 = tf.reshape(new_fc7_verbs_D_1, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, last_dim])
        word2vec_list_D_1 = tf.reshape(tmp_word2vec_list, [tf.shape(gan_fc7_verbs)[0] * self.obj_num_classes, obj_dim])

        # all
        # all_new_fc7_verbs_D_1, all_new_ho_class_D_1, all_word2vec_list_D_1 = self.conds_zeros(None, new_fc7_verbs_D_1, new_gt_class_HO_1, word2vec_list_D_1)

        # In fact, we also tried use constrastive loss, however, it is useless.

        new_fc7_verbs_D_1, new_ho_class_D_1, word2vec_list_D_1, new_old_obj, new_obj_labels = self.sample_instances(gan_fc7_verbs, new_fc7_verbs_D_1,
                                                                                       new_gt_class_HO_1,
                                                                                       word2vec_list_D_1, old_obj = old_obj_labels,
                                                                                       new_obj = obj_labels)
        item_weights = None
        fake_obj_list = self.convert_emb_feats(word2vec_list_D_1, new_fc7_verbs_D_1, type= noise_type)

        # self.add_discriminator_loss(fake_obj_list, new_fc7_objects, initializer, is_training)
        print(fake_obj_list, new_fc7_verbs_D_1, new_ho_class_D_1, '===========')
        self.cal_dax_loss(fake_obj_list, initializer, is_training, new_fc7_verbs_D_1, new_ho_class_D_1, item_weights=item_weights, nameprefix='fake_tmp_')

        ll = self.get_ll()
        self.network.losses['fake_total_loss'] += self.network.losses['fake_tmp_verbs_cross_entropy'] * ll

        # This is only for step-wise optimization
        self.network.losses['fake_G_total_loss'] += (self.network.losses['fake_tmp_verbs_cross_entropy'])

        with tf.device('/cpu:0'):
            fc7_O = tf.Print(fc7_O,
                             [1, tf.shape(fc7_O), tf.shape(fc7_verbs), tf.shape(word2vec_list_D_1),
                              tf.shape(new_ho_class_D_1),
                              tf.shape(new_fc7_verbs_D_1)],
                             "_dax shape:", first_n=100)

        return fc7_O, fc7_verbs

    def cal_dax_loss(self, fake_obj_list, initializer, is_training, new_fc7_verbs_D_1, new_ho_class_D_1, item_weights=None, nameprefix='fake_tmp_'):
        """
        This includes all objects. ie for each verb, fabricate all kinds of objects.
        :param fake_obj_list:
        :param initializer:
        :param is_training:
        :param new_fc7_verbs_D_1:
        :param new_ho_class_D_1:
        :param item_weights:
        :param nameprefix:
        :return:
        """
        print('before', new_ho_class_D_1)

        fc7_vo = self.network.head_to_tail_ho(fake_obj_list, new_fc7_verbs_D_1, None, None, is_training, 'fc_HO')
        cls_prob_verbs = self.network.region_classification_ho(fc7_vo, is_training, initializer, 'classification',
                                                                nameprefix=nameprefix)
        fake_cls_score_verbs = self.network.predictions[nameprefix + "cls_score_hoi"]
        print('lllll', fake_cls_score_verbs)
        import numpy as np
        # reweights = np.log(1 / (self.network.num_inst_all / np.sum(self.network.num_inst_all)))
        reweights = np.log(1 / (self.network.num_inst / np.sum(self.network.num_inst))) # This is same as network.HO_weights.
        if self.network.model_name.__contains__('_zs'):
            # this is only for zero shot. Following previous work.
            from ult.ult import get_zero_shot_type
            zero_shot_type = get_zero_shot_type(self.network.model_name)
            from ult.ult import get_unseen_index
            unseen_idx = get_unseen_index(zero_shot_type)
            import numpy as np
            reweights[unseen_idx] = 20.
            print(reweights)

        loss = tf.multiply(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=new_ho_class_D_1, logits=fake_cls_score_verbs), reweights)
        loss = tf.reduce_mean(loss)
        self.network.losses[nameprefix + 'verbs_cross_entropy'] = (loss / 2)

    def obtain_last_obj_dim(self):
        # for ablated study, you can ignore
        obj_dim = 2048
        if self.network.model_name.__contains__('_ohot'):
            obj_dim = self.obj_num_classes
        elif self.network.model_name.__contains__('_wemb_gan'):
            obj_dim = 300
        elif self.network.model_name.__contains__('_xemb_gan'):
            obj_dim = 300
        elif self.network.model_name.__contains__('_vemb_gan'):
            obj_dim = 300
        return obj_dim

    def get_wemb_mask(self, gt_obj_class, gt_obj_class_new, neighbor_num = 5):
        from networks.tools import get_neighborhood_matrix
        matrix = get_neighborhood_matrix(neighbor_num)
        matrix = tf.constant(matrix)
        available_obj_label = tf.matmul(gt_obj_class_new, matrix)
        conds = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(available_obj_label, tf.bool),
                                                     tf.cast(gt_obj_class, tf.bool)), tf.int32), axis=-1) > 0
        return conds

    def conds_zeros(self, fc7_verbs, new_fc7_verbs_D_1, new_gt_class_HO_1, word2vec_list_D_1,):
        """
        remove infeasible HOIs
        :param fc7_verbs:
        :param new_fc7_verbs_D_1:
        :param new_gt_class_HO_1:
        :param word2vec_list_D_1:
        :return:
        """
        conds = tf.greater(tf.reduce_sum(new_gt_class_HO_1, axis=-1), 0)
        word2vec_list_D_1 = tf.boolean_mask(word2vec_list_D_1, conds)
        new_fc7_verbs_D_1 = tf.boolean_mask(new_fc7_verbs_D_1, conds)
        new_ho_class_D_1 = tf.boolean_mask(new_gt_class_HO_1, conds)
        return new_fc7_verbs_D_1, new_ho_class_D_1, word2vec_list_D_1

    def sample_instances(self, fc7_verbs, new_fc7_verbs_D_1, new_gt_class_HO_1, word2vec_list_D_1,
                         old_obj=None, new_obj=None, gen_num_clsses = 80,):
        with tf.device('/cpu:0'):
            word2vec_list_D_1 = tf.Print(word2vec_list_D_1,
                                         [tf.shape(word2vec_list_D_1), tf.shape(new_fc7_verbs_D_1), tf.reduce_sum(new_gt_class_HO_1)],
                                         'begin indexes:', first_n=100)
        conds = None
        new_ho_class_D_1 = new_gt_class_HO_1
        if self.network.model_name.__contains__('rands'):

            tmp_gt = tf.reshape(new_gt_class_HO_1, [-1, gen_num_clsses, self.network.compose_num_classes])
            print('==============', gen_num_clsses, self.obtain_last_verb_dim())
            new_fc7_verbs_D_1 = tf.reshape(new_fc7_verbs_D_1, [-1, gen_num_clsses, self.obtain_last_verb_dim()])
            word2vec_list_D_1 = tf.reshape(word2vec_list_D_1, [-1, gen_num_clsses, self.obtain_last_obj_dim()])


            noise = tf.random.uniform(shape=[gen_num_clsses, ], minval=0.00001, maxval=1.)
            noise = tf.expand_dims(noise, axis=0)
            noise = tf.expand_dims(noise, axis=-1)
            # N O HOI
            noise = tf.tile(noise, [tf.shape(tmp_gt)[0], 1, self.network.compose_num_classes])
            ll = tf.reduce_sum(tmp_gt, axis=-1)  # the number of labels for each composite HOI
            ll = tf.where(tf.equal(ll, 0), tf.ones_like(ll), ll)

            if self.network.model_name.__contains__('select'):
                # for ablated study
                # old_obj = tf.boolean_mask(old_obj, conds)
                # new_obj = tf.boolean_mask(new_obj, conds)
                neighbor_num = 40
                if self.network.model_name.__contains__('select10'):
                    neighbor_num = 10
                    pass
                elif self.network.model_name.__contains__('select1'):
                    neighbor_num = 1
                    pass
                elif self.network.model_name.__contains__('select5'):
                    neighbor_num = 5
                    pass
                elif self.network.model_name.__contains__('select20'):
                    neighbor_num = 20
                    pass
                elif self.network.model_name.__contains__('select40'):
                    neighbor_num = 40
                    pass
                elif self.network.model_name.__contains__('select80'):
                    neighbor_num = 80 #
                    pass
                tmp_conds_sim = self.get_wemb_mask(old_obj, new_obj, neighbor_num)
                sim_gt = tf.reshape(tmp_conds_sim, [-1, gen_num_clsses])
                sim_gt = tf.expand_dims(sim_gt, axis=-1)
                sim_gt = tf.tile(sim_gt, [1, 1, self.network.compose_num_classes])
                sim_gt = tf.cast(sim_gt, tf.float32)
                sim_gt = tf.multiply(sim_gt, tmp_gt) # filter out Dissimilar composite HOIs to original HOI
                noise = tf.reduce_sum(tf.multiply(sim_gt, noise), axis=-1) / ll
            else:
                print(tmp_gt, noise, '=========')
                noise = tf.reduce_sum(tf.multiply(tmp_gt, noise), axis=-1) / ll # remove zeros
                # is equal to : noise = tf.multiply(tf.cast(tf.reduce_sum(tmp_gt, axis=-1) > 0, tf.float32), tf.reduce_sum(noise, axis=-1)) / 600
                # here 600 does not matter
                # remove the noise whose corresponding gt label is zero.
            # N O
            select_num = 3
            indexes = tf.argsort(noise, axis=-1,  direction='DESCENDING', stable=True)[:, :select_num]
            print('=========DEBUG:indexes', indexes, )

            word2vec_list_D_1 = tf.batch_gather(word2vec_list_D_1, indexes)
            new_fc7_verbs_D_1 = tf.batch_gather(new_fc7_verbs_D_1, indexes)
            tmp_gt = tf.batch_gather(tmp_gt, indexes)

            new_fc7_verbs_D_1 = tf.reshape(new_fc7_verbs_D_1, [-1, self.obtain_last_verb_dim()])
            word2vec_list_D_1 = tf.reshape(word2vec_list_D_1, [-1, self.obtain_last_obj_dim()])
            new_ho_class_D_1 = tf.reshape(tmp_gt, [-1, self.network.compose_num_classes])

            new_gt_class_HO_1 = new_ho_class_D_1
            conds = tf.greater(tf.reduce_sum(new_ho_class_D_1, axis=-1), 0)

            with tf.device('/cpu:0'):
                word2vec_list_D_1 = tf.Print(word2vec_list_D_1,
                                             [tf.shape(noise), tf.shape(fc7_verbs), tf.shape(word2vec_list_D_1)],
                                             'randso indexes:', first_n=100)

        if conds is not None:
            print(conds, new_gt_class_HO_1, '==============================')
            word2vec_list_D_1 = tf.boolean_mask(word2vec_list_D_1, conds)
            new_fc7_verbs_D_1 = tf.boolean_mask(new_fc7_verbs_D_1, conds)
            new_ho_class_D_1 = tf.boolean_mask(new_gt_class_HO_1, conds)
            if old_obj is not None and new_obj is not None:
                old_obj = tf.boolean_mask(old_obj, conds)
                new_obj = tf.boolean_mask(new_obj, conds)
        with tf.device('/cpu:0'):
            word2vec_list_D_1 = tf.Print(word2vec_list_D_1,
                                         [tf.shape(word2vec_list_D_1), tf.shape(new_ho_class_D_1), tf.shape(new_gt_class_HO_1)],
                                         'begin2 indexes:', first_n=100)
        return new_fc7_verbs_D_1, new_ho_class_D_1, word2vec_list_D_1, old_obj, new_obj
