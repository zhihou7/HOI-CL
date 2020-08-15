# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# --------------------------------------------------------

import tensorflow as tf

from ult.ult import get_unseen_index, get_zero_shot_type, get_augment_type, get_aug_params


class VCL(object):

    def __init__(self, net):
        self.net = net
        self.additional_loss = None
        pass

    def compose_ho_between_images(self, O_features, V_features, cur_gt_class_HO, type):
        return self.compose_ho_inner(O_features, V_features, cur_gt_class_HO, type)


    def compose_ho_inner(self, O_features, V_features, cur_gt_class_HO, type, is_single=False):
        """

        :param O_features:
        :param V_features:
        :param cur_gt_class_HO:
        :param type:
        :param is_single:
        :return: fc7_O, fc7_V, gt_verb_class, new_gt_class_HO, gt_obj_class, gt_obj_class_orig
        gt_obj_class_orig is corresponding to gt_verb_class
        """
        fc7_O_0 = O_features[0]
        fc7_O_1 = O_features[1]
        fc7_V_0 = V_features[0]
        fc7_V_1 = V_features[1]
        shape_0 = tf.shape(fc7_O_0)[0]
        shape_1 = tf.shape(fc7_O_1)[0]
        gt_obj_class_l = []
        gt_verb_class_l = []
        for i in range(2):
            gt_obj_class0 = tf.cast(
                tf.matmul(cur_gt_class_HO[i], self.net.obj_to_HO_matrix, transpose_b=True) > 0,
                tf.float32)
            gt_verb_class0 = tf.cast(
                tf.matmul(cur_gt_class_HO[i], self.net.verb_to_HO_matrix, transpose_b=True) > 0,
                tf.float32)
            gt_obj_class_l.append(gt_obj_class0)
            gt_verb_class_l.append(gt_verb_class0)
        # with tf.device('/cpu:0'):
        #     fc7_O_0 = tf.Print(fc7_O_0,
        #                                    [tf.shape(fc7_O_0), tf.shape(fc7_O_1) ], 'input:', first_n=100)
        # TODO this function needs to return the gt_obj_class and gt_verb_class
        # TODO this function will update gt_obj_class_l and gt_verb_class_l
        fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1 = self.select_compose_candidate_elements(fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1,
                                                                                    gt_obj_class_l, gt_verb_class_l,
                                                                                    type)
        if self.net.model_name.__contains__('_def1') and not is_single:
            # this is similar to the default. This operation will compose relations from the single image.
            len = tf.maximum(tf.shape(fc7_O_0)[0], tf.shape(fc7_O_1)[0])
            fc7_O_0 = fc7_O_0[:len]
            fc7_O_1 = fc7_O_1[:len]
            fc7_V_1 = fc7_V_1[:len]
            fc7_V_0 = fc7_V_0[:len]
            with tf.device('/cpu:0'):
                fc7_O_0 = tf.Print(fc7_O_0, [tf.shape(fc7_O_0), tf.shape(fc7_O_1), tf.shape(fc7_V_0), tf.shape(fc7_V_1)], 'shape c2:', first_n=100)

            gt_obj_class_l[0] = gt_obj_class_l[0][:len]
            gt_obj_class_l[1] = gt_obj_class_l[1][:len]
            gt_verb_class_l[0] = gt_verb_class_l[0][:len]
            gt_verb_class_l[1] = gt_verb_class_l[1][:len]

        elif self.net.model_name.__contains__('_def2') and not is_single:
            # do not compose HOIs within HOIs.
            len = tf.minimum(tf.shape(fc7_O_0)[0], tf.shape(fc7_O_1)[0])
            fc7_O_0 = fc7_O_0[:len]
            fc7_O_1 = fc7_O_1[:len]
            fc7_V_1 = fc7_V_1[:len]
            fc7_V_0 = fc7_V_0[:len]
            with tf.device('/cpu:0'):
                fc7_O_0 = tf.Print(fc7_O_0, [tf.shape(fc7_O_0), tf.shape(fc7_O_1), tf.shape(fc7_V_0), tf.shape(fc7_V_1)], 'shape c2:', first_n=100)

            gt_obj_class_l[0] = gt_obj_class_l[0][:len]
            gt_obj_class_l[1] = gt_obj_class_l[1][:len]
            gt_verb_class_l[0] = gt_verb_class_l[0][:len]
            gt_verb_class_l[1] = gt_verb_class_l[1][:len]
        elif self.net.model_name.__contains__('_def3') and not is_single:
            # tile the short part. When I use tf.tile, it will throw an Exception, which I do not how to fix.
            # I do not know whether this will affect the performance or not.
            # In our experiment, this is apparently worse ``def''.
            fc7_O_1 = tf.concat([fc7_O_1, fc7_O_1, fc7_O_1, fc7_O_1, fc7_O_1, fc7_O_1], axis=0)
            fc7_V_1 = tf.concat([fc7_V_1, fc7_V_1, fc7_V_1, fc7_V_1, fc7_O_1, fc7_O_1], axis=0)
            gt_obj_class_l[1] = tf.concat([gt_obj_class_l[1], gt_obj_class_l[1], gt_obj_class_l[1], gt_obj_class_l[1],
                                           gt_obj_class_l[1], gt_obj_class_l[1]], axis=0)
            gt_verb_class_l[1] = tf.concat([gt_verb_class_l[1], gt_verb_class_l[1], gt_verb_class_l[1], gt_verb_class_l[1],
                                            gt_verb_class_l[1], gt_verb_class_l[1]], axis=0)

            len = tf.minimum(tf.shape(fc7_O_0)[0], tf.shape(fc7_O_1)[0])
            fc7_O_0 = fc7_O_0[:len]
            fc7_O_1 = fc7_O_1[:len]
            fc7_V_1 = fc7_V_1[:len]
            fc7_V_0 = fc7_V_0[:len]
            with tf.device('/cpu:0'):
                fc7_O_0 = tf.Print(fc7_O_0, [tf.shape(fc7_O_0), tf.shape(fc7_O_1), tf.shape(fc7_V_0), tf.shape(fc7_V_1)], 'shape c2:', first_n=100)

            gt_obj_class_l[0] = gt_obj_class_l[0][:len]
            gt_obj_class_l[1] = gt_obj_class_l[1][:len]
            gt_verb_class_l[0] = gt_verb_class_l[0][:len]
            gt_verb_class_l[1] = gt_verb_class_l[1][:len]

        # just 1 composition
        fc7_O = tf.concat([fc7_O_0, fc7_O_1], axis=0)
        fc7_V = tf.concat([fc7_V_1, fc7_V_0], axis=0)
        gt_obj_class = tf.concat([gt_obj_class_l[0], gt_obj_class_l[1]], axis=0)
        gt_verb_class = tf.concat([gt_verb_class_l[1], gt_verb_class_l[0]], axis=0)
        # based on verb,
        # the composite verb items are same with the original verb items
        gt_obj_class_orig = tf.concat([gt_obj_class_l[1], gt_obj_class_l[0]], axis=0)

        tmp_ho_class_from_obj = tf.matmul(gt_obj_class, self.net.obj_to_HO_matrix) > 0
        tmp_ho_class_from_vb = tf.matmul(gt_verb_class, self.net.verb_to_HO_matrix) > 0
        new_gt_class_HO = tf.cast(tf.logical_and(tmp_ho_class_from_obj, tmp_ho_class_from_vb), tf.float32)

        return fc7_O, fc7_V, gt_verb_class, new_gt_class_HO, gt_obj_class, gt_obj_class_orig

    def select_composited_hois(self, fc7_O, fc7_V, gt_obj_class, gt_obj_class_orig, gt_verb_class, is_single,
                               new_gt_class_HO, base_compose_item_weights = None):
        conds = None
        compose_item_weights = None
        # select composited hois according to the similarity of objects.
        if compose_item_weights is None:
            compose_item_weights = tf.ones([tf.shape(fc7_O)[0]])
        return compose_item_weights, conds, fc7_O, fc7_V, gt_verb_class, new_gt_class_HO

    def compose_ho_single_images(self, O_features, V_features, cur_gt_class_HO, type):
        return self.compose_ho_inner(O_features, V_features, cur_gt_class_HO, type, is_single=True)


    def select_compose_candidate_elements(self, fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1, gt_obj_class_l, gt_verb_class_l,
                                          type):
        """
        select candidate elements by similarity or other rules.
        I have tried various kinds of strategies, but I do not find any apparent improvement.
        :param fc7_O_0:
        :param fc7_O_1:
        :param fc7_V_0:
        :param fc7_V_1:
        :param gt_obj_class_l:
        :param gt_verb_class_l:
        :param type:
        :return:
        """
        if self.net.model_name.__contains__('_c2_'):
            # ignore the augmentation data
            fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1 = self.create_new_HO_features_c2(fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1,
                                                                                gt_obj_class_l, gt_verb_class_l)
            with tf.device('/cpu:0'):
                fc7_O_0 = tf.Print(fc7_O_0,
                                   [tf.shape(fc7_O_0), tf.shape(fc7_O_1), tf.shape(fc7_V_0), tf.shape(fc7_V_1)],
                                   'shape c1:', first_n=100)

        elif not type.__contains__('default'): # all
            fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1 = self.create_new_HO_features(fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1,
                                                                             gt_obj_class_l, gt_verb_class_l)
            with tf.device('/cpu:0'):
                fc7_O_0 = tf.Print(fc7_O_0,
                                   [tf.shape(fc7_O_0), tf.shape(fc7_O_1), tf.shape(fc7_V_0), tf.shape(fc7_V_1)],
                                   'shape c1:', first_n=100)
        return fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1

    def get_ll(self):
        """
        the super-parameter, 0.5 is best in our experiment. 1.0 is also ok.
        :return:
        """
        ll = 1.0
        if self.net.model_name.__contains__('_ml2_'):
            ll = 0.2
        elif self.net.model_name.__contains__('_ml5_'):
            ll = 0.5
        elif self.net.model_name.__contains__('_ml8_'):
            ll = 0.8
        elif self.net.model_name.__contains__('_ml05_'):
            ll = 0.05
        elif self.net.model_name.__contains__('_ml01_'):
            ll = 0.01
        elif self.net.model_name.__contains__('_ml1_'):
            ll = 0.1
        elif self.net.model_name.__contains__('_ml10_'):
            ll = 1.
        elif self.net.model_name.__contains__('_ml0_'):
            ll = 0
        elif self.net.model_name.__contains__('_ml15_'):
            ll = 1.5
        elif self.net.model_name.__contains__('_ml20_'):
            ll = 2.
        return ll

    def merge_generate(self, O_features, V_features, cur_gt_class_HO, type = 'default'):
        # assert type == 0 or type == 1 or type == 2 or type == 3
        compose_item_weights = 1.
        # if type.__contains__('compose_all_hos'):
        #     fc7_O, fc7_V, gt_obj_class, gt_obj_class_orig, gt_verb_class, new_gt_class_HO = self.compose_ho_all(O_features, V_features,
        #                                                                        cur_gt_class_HO)
        # else:
        fc7_O, fc7_V, gt_verb_class, new_gt_class_HO, gt_obj_class, gt_obj_class_orig = self.compose_ho_between_images(
            O_features, V_features,
            cur_gt_class_HO, type)

        compose_item_weights, conds, fc7_O, fc7_V, gt_verb_class, new_gt_class_HO = self.select_composited_hois(
                fc7_O,
                fc7_V,
                gt_obj_class,
                gt_obj_class_orig,
                gt_verb_class,
                False,
                new_gt_class_HO,
                base_compose_item_weights=compose_item_weights)

        new_gt_class_HO, compose_item_weights, fc7_O, fc7_V, gt_obj_class, gt_obj_class_orig, gt_verb_class = self.conds_zeros(
            new_gt_class_HO, compose_item_weights, fc7_O, fc7_V, gt_obj_class, gt_obj_class_orig, gt_verb_class)

        # with tf.device('/cpu:0'):
        #     new_gt_class_HO = tf.Print(new_gt_class_HO, [tf.shape(fc7_O), tf.shape(fc7_V), tf.shape(gt_verb_class), tf.shape(new_gt_class_HO)], 'message:', first_n=100)
        # print('degut===============', fc7_O, fc7_V, gt_verb_class, new_gt_class_HO, compose_item_weights)

        new_loss = self.cal_loss_with_new_composing_features(fc7_O, fc7_V, gt_verb_class, new_gt_class_HO, type, compose_item_weights=compose_item_weights, orig_len=tf.shape(O_features[0])[0] + tf.shape(O_features[1])[0])
        return new_loss

    def conds_zeros(self, new_gt_class_HO, *args):

        label_conds = tf.reduce_sum(tf.abs(new_gt_class_HO), axis=-1) > 0
        print(label_conds, new_gt_class_HO)
        new_gt_class_HO = tf.boolean_mask(new_gt_class_HO, label_conds, axis=0)
        result = [new_gt_class_HO]
        for item in args:
            if item is not None:
                item = tf.boolean_mask(item, label_conds, axis=0)
                result.append(item)
            else:
                result.append(item)
        return result

    def cal_loss_with_new_composing_features(self, fc7_O, fc7_V, gt_verb_class, new_gt_class_HO, type, compose_item_weights=None, orig_len=50):
        with tf.device('/cpu:0'):
            new_gt_class_HO = tf.Print(new_gt_class_HO, [tf.shape(fc7_O), tf.shape(fc7_V), tf.shape(new_gt_class_HO)],
                                       '1message:', first_n=100)
        if self.net.model_name.__contains__('VCOCO') and self.net.model_name.__contains__('t3'):
            fc7_vo = self.net.head_to_tail_ho(fc7_O, fc7_V, None, None, True, 'fc_HO_vcoco')
            orig_num_classes = self.net.num_classes
            self.net.num_classes = self.net.compose_num_classes
            self.net.region_classification_ho(fc7_vo, True, tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                              'classification_aux', nameprefix='merge_')
            self.net.num_classes = orig_num_classes
        else:
            fc7_vo = self.net.head_to_tail_ho(fc7_O, fc7_V, None, None, True, 'fc_HO')
            self.net.region_classification_ho(fc7_vo, True, tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                              'classification', nameprefix='merge_')
        cls_score_verbs = self.net.predictions["merge_cls_score_hoi"]
        if self.net.model_name.__contains__('VCOCO') and self.net.model_name.__contains__('t3'):
            reweights = tf.matmul(self.net.HO_weight, self.net.verb_to_HO_matrix)
        else:
            reweights = self.net.HO_weight

        if self.net.model_name.__contains__('rew2'):
            # this means that we also apply the reweight strategy for the generated HO relation
            # TODO I simply and empirically set the weights for VCL. I think there should be a better solution.
            #  Noticeably, our method is orthogonal to re-weighting strategy.
            #  Moreover, completely copying from previous work, we multiply the weights at the logits.
            #  I think this is also an important reason why baseline of zero-shot has some values!
            #  This can help the network learn from the known factors (i.e. verb and object)
            #  It might be because the non-linear sigmoid function.
            #  After this kind of re-weighting, the small value (e.g. 0.1) will further tend 0 where the gradient
            #  is larger. It is interesting! We do not mention this in paper since our method is orthogonal to this.
            #  But I do not understand the reason very good. Hope someone can explain.
            cls_score_verbs = tf.multiply(self.net.predictions["merge_cls_score_hoi"], reweights / 10)
        elif self.net.model_name.__contains__('_rew51'):
            # this is for zero-shot, we simply emphasize the weights of zero-shot categories.
            zero_shot_type = get_zero_shot_type(self.net.model_name)
            unseen_idx = get_unseen_index(zero_shot_type)
            import numpy as np
            new_HO_weight = np.asarray(self.net.HO_weight).reshape(-1)
            # use the maximum value for the weight of unseen HOIs
            new_HO_weight[unseen_idx] = 20.
            new_HO_weight = new_HO_weight.reshape(1, 600)
            cls_score_verbs = tf.multiply(self.net.predictions["merge_cls_score_hoi"],
                                          new_HO_weight / 10)
        if self.net.model_name.__contains__('VCOCO') and (self.net.model_name.__contains__('_t1_')
                                                          or self.net.model_name.__contains__('_t2_')):
            loss_func = self.calculate_loss_by_removing_useless_coco
        else:
            loss_func = self.calculate_loss_by_removing_useless
        new_loss = loss_func(cls_score_verbs, new_gt_class_HO, weights=compose_item_weights)
        return new_loss


    def compose_ho_all(self, O_features, V_features, cur_gt_class_HO):

        _fc7_O = tf.concat([O_features[0], O_features[1]], axis=0)
        _fc7_V = tf.concat([V_features[0], V_features[1]], axis=0)
        _gt_class_HO = tf.concat([cur_gt_class_HO[0], cur_gt_class_HO[1]], axis=0)

        fc7_O, fc7_V, gt_obj_class, gt_obj_class_orig, gt_verb_class, new_gt_class_HO = self.compose_ho_all_inner(_fc7_O, _fc7_V, _gt_class_HO)
        return fc7_O, fc7_V, gt_obj_class, gt_obj_class_orig, gt_verb_class, new_gt_class_HO


    def compose_ho_all_inner(self, fc7_O, fc7_V, gt_class_HO):
        _fc7_O = fc7_O
        gt_obj_class = tf.cast(
            tf.matmul(gt_class_HO, self.net.obj_to_HO_matrix, transpose_b=True) > 0,
            tf.float32)
        gt_obj_class_orig = gt_obj_class
        gt_verb_class = tf.cast(
            tf.matmul(gt_class_HO, self.net.verb_to_HO_matrix, transpose_b=True) > 0,
            tf.float32)
        last_obj_dim = 2048
        with tf.device('/cpu:0'):
            fc7_V = tf.Print(fc7_V,
                              [tf.shape(fc7_V)],
                              'fc7_V================:', first_n=100)
        if not self.net.model_name.__contains__('pose'):
            last_vb_dim = 2048
        elif self.net.model_name.__contains__('posesp'):
            last_vb_dim = 7456
        else:
            last_vb_dim = tf.shape(fc7_V)[-1]
        shape_O = tf.shape(fc7_O)[0]
        shape_V = tf.shape(fc7_V)[0]
        fc7_O = tf.expand_dims(fc7_O, dim=0)
        fc7_V = tf.expand_dims(fc7_V, dim=1)
        fc7_O = tf.reshape(tf.tile(fc7_O, [shape_V, 1, 1]),
                           [shape_V * shape_O, last_obj_dim])
        fc7_V = tf.reshape(tf.tile(fc7_V, [1, shape_O, 1]),
                           [shape_V * shape_O, last_vb_dim])
        gt_obj_class = tf.expand_dims(gt_obj_class, dim=0)
        gt_verb_class = tf.expand_dims(gt_verb_class, dim=1)
        gt_obj_class_orig = tf.expand_dims(gt_obj_class_orig, dim=1)
        gt_obj_class = tf.reshape(tf.tile(gt_obj_class, [shape_V, 1, 1]),
                                  [shape_V * shape_O,
                                   self.net.obj_num_classes])
        gt_verb_class = tf.reshape(tf.tile(gt_verb_class, [1, shape_O, 1]),
                                   [shape_V * shape_O,
                                    self.net.verb_num_classes])
        gt_obj_class_orig = tf.reshape(tf.tile(gt_obj_class_orig, [1, shape_O, 1]),
                                   [shape_V * shape_O,
                                    self.net.obj_num_classes])
        tmp_ho_class_from_obj = tf.cast(tf.matmul(gt_obj_class, self.net.obj_to_HO_matrix) > 0, tf.float32)
        tmp_ho_class_from_vb = tf.cast(tf.matmul(gt_verb_class, self.net.verb_to_HO_matrix) > 0,
                                       tf.float32)
        new_gt_class_HO = tf.cast(tmp_ho_class_from_obj + tmp_ho_class_from_vb > 1., tf.float32)
        return fc7_O, fc7_V, gt_obj_class, gt_obj_class_orig, gt_verb_class, new_gt_class_HO

    def create_new_HO_features(self, fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1, gt_obj_class_l, gt_verb_class_l):
        last_obj_dim = 2048

        if not self.net.model_name.__contains__('pose'):
            last_vb_dim = 2048
        else:
            last_vb_dim = tf.shape(fc7_V_0)[-1]
        shape_0 = tf.shape(fc7_O_0)[0]
        shape_1 = tf.shape(fc7_O_1)[0]
        fc7_O_0 = tf.expand_dims(fc7_O_0, dim=0)
        fc7_V_0 = tf.expand_dims(fc7_V_0, dim=0)
        fc7_O_1 = tf.expand_dims(fc7_O_1, dim=1)
        fc7_V_1 = tf.expand_dims(fc7_V_1, dim=1)
        fc7_O_0 = tf.reshape(tf.tile(fc7_O_0, [shape_1, 1, 1]),
                             [shape_1 * shape_0, last_obj_dim])
        fc7_V_1 = tf.reshape(tf.tile(fc7_V_1, [1, shape_0, 1]),
                             [shape_1 * shape_0, last_vb_dim])
        fc7_O_1 = tf.reshape(tf.tile(fc7_O_1, [1, shape_0, 1]),
                             [shape_1 * shape_0, last_obj_dim])
        fc7_V_0 = tf.reshape(tf.tile(fc7_V_0, [shape_1, 1, 1]),
                             [shape_1 * shape_0, last_vb_dim])
        gt_obj_class_l[0] = tf.expand_dims(gt_obj_class_l[0], dim=0)
        gt_verb_class_l[0] = tf.expand_dims(gt_verb_class_l[0], dim=0)
        gt_obj_class_l[1] = tf.expand_dims(gt_obj_class_l[1], dim=1)
        gt_verb_class_l[1] = tf.expand_dims(gt_verb_class_l[1], dim=1)
        gt_obj_class_l[0] = tf.reshape(tf.tile(gt_obj_class_l[0], [shape_1, 1, 1]),
                                       [shape_1 * shape_0,
                                        80])
        gt_verb_class_l[1] = tf.reshape(tf.tile(gt_verb_class_l[1], [1, shape_0, 1]),
                                        [shape_1 * shape_0,
                                         117])
        gt_obj_class_l[1] = tf.reshape(tf.tile(gt_obj_class_l[1], [1, shape_0, 1]),
                                       [shape_1 * shape_0,
                                        80])
        gt_verb_class_l[0] = tf.reshape(tf.tile(gt_verb_class_l[0], [shape_1, 1, 1]),
                                        [shape_1 * shape_0,
                                         117])
        return fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1


    def create_new_HO_features_c2(self, fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1, gt_obj_class_l, gt_verb_class_l):
        last_obj_dim = 2048

        augment_type = get_augment_type(self.net.model_name)
        Neg_select1, Pos_augment1, inters_per_img = get_aug_params(0, 0, augment_type)
        per_aug_nums = Pos_augment1 + 1
        if not self.net.model_name.__contains__('pose'):
            last_vb_dim = 2048
        else:
            last_vb_dim = tf.shape(fc7_V_0)[-1]
        shape_0 = tf.shape(fc7_O_0)[0]
        shape_1 = tf.shape(fc7_O_1)[0]
        with tf.device('/cpu:0'):
            fc7_O_0 = tf.Print(fc7_O_0, [tf.shape(fc7_O_0), tf.shape(fc7_O_1), tf.shape(fc7_O_1)],
                                       '2message:', first_n=100)
        print(fc7_O_0, per_aug_nums, augment_type, '-----------------debug---------------')
        fc7_O_0 = tf.reshape(fc7_O_0, [shape_0 // per_aug_nums, per_aug_nums, last_obj_dim])
        fc7_O_1 = tf.reshape(fc7_O_1, [shape_1 // per_aug_nums, per_aug_nums, last_obj_dim])

        fc7_V_0 = tf.reshape(fc7_V_0, [shape_0 // per_aug_nums, per_aug_nums, last_vb_dim])
        fc7_V_1 = tf.reshape(fc7_V_1, [shape_1 // per_aug_nums, per_aug_nums, last_vb_dim])

        gt_obj_class_l[0] = tf.reshape(gt_obj_class_l[0], [shape_0 // per_aug_nums, per_aug_nums, 80])
        gt_obj_class_l[1] = tf.reshape(gt_obj_class_l[1], [shape_1 // per_aug_nums, per_aug_nums, 80])

        gt_verb_class_l[0] = tf.reshape(gt_verb_class_l[0], [shape_0 // per_aug_nums, per_aug_nums, 117])
        gt_verb_class_l[1] = tf.reshape(gt_verb_class_l[1], [shape_1 // per_aug_nums, per_aug_nums, 117])

        shape_0 = tf.shape(fc7_O_0)[0]
        shape_1 = tf.shape(fc7_O_1)[0]

        fc7_O_0 = tf.expand_dims(fc7_O_0, dim=0)
        fc7_V_0 = tf.expand_dims(fc7_V_0, dim=0)
        fc7_O_1 = tf.expand_dims(fc7_O_1, dim=1)
        fc7_V_1 = tf.expand_dims(fc7_V_1, dim=1)
        fc7_O_0 = tf.reshape(tf.tile(fc7_O_0, [shape_1, 1, 1, 1]),
                             [shape_1 * shape_0 * per_aug_nums, last_obj_dim])
        fc7_V_1 = tf.reshape(tf.tile(fc7_V_1, [1, shape_0, 1, 1]),
                             [shape_1 * shape_0 * per_aug_nums, last_vb_dim])
        fc7_O_1 = tf.reshape(tf.tile(fc7_O_1, [1, shape_0, 1, 1]),
                             [shape_1 * shape_0 * per_aug_nums, last_obj_dim])
        fc7_V_0 = tf.reshape(tf.tile(fc7_V_0, [shape_1, 1, 1, 1]),
                             [shape_1 * shape_0 * per_aug_nums, last_vb_dim])
        gt_obj_class_l[0] = tf.expand_dims(gt_obj_class_l[0], dim=0)
        gt_verb_class_l[0] = tf.expand_dims(gt_verb_class_l[0], dim=0)
        gt_obj_class_l[1] = tf.expand_dims(gt_obj_class_l[1], dim=1)
        gt_verb_class_l[1] = tf.expand_dims(gt_verb_class_l[1], dim=1)
        gt_obj_class_l[0] = tf.reshape(tf.tile(gt_obj_class_l[0], [shape_1, 1, 1, 1]),
                                       [shape_1 * shape_0 * per_aug_nums,
                                        80])
        gt_verb_class_l[1] = tf.reshape(tf.tile(gt_verb_class_l[1], [1, shape_0, 1, 1]),
                                        [shape_1 * shape_0 * per_aug_nums,
                                         117])
        gt_obj_class_l[1] = tf.reshape(tf.tile(gt_obj_class_l[1], [1, shape_0, 1, 1]),
                                       [shape_1 * shape_0 * per_aug_nums,
                                        80])
        gt_verb_class_l[0] = tf.reshape(tf.tile(gt_verb_class_l[0], [shape_1, 1, 1, 1]),
                                        [shape_1 * shape_0 * per_aug_nums,
                                         117])
        return fc7_O_0, fc7_O_1, fc7_V_0, fc7_V_1

    def calculate_loss_by_removing_useless(self, cls_score_verbs, new_gt_class_HO, weights=None):
        tmp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=new_gt_class_HO, logits=cls_score_verbs), axis=-1)
        if weights is not None:
            print('calculate_loss_by_removing_useless======= not none')
            tmp = tf.multiply(tmp, weights)
        new_loss = tf.reduce_mean(tmp)
        return new_loss

    def calculate_loss_by_removing_useless_coco(self, cls_score_verbs, new_gt_class_HO, weights=None):
        new_gt_class_verbs = tf.cast(
            tf.matmul(new_gt_class_HO, self.net.verb_to_HO_matrix, transpose_b=True) > 0,
            tf.float32)
        tmp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=new_gt_class_verbs, logits=cls_score_verbs), axis=-1)
        new_loss = tf.reduce_mean(tmp)
        # with tf.device('/cpu:0'):
        #     new_loss = tf.Print(new_loss, [tf.shape(O_features[0]), tf.shape(new_gt_class_HO), new_loss, entropy1],
        #                            'after message:', first_n=10000)
        return new_loss

    def get_compose_type(self):
        compose_type = 'default'  # 0

        return compose_type
