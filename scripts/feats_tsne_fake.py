#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 上午10:01
# @Author  : Zhi Hou
# @Site    : 
# @File    : feats_tsne.py
# @Software: PyCharm
#
#
#                  .--,       .--,
#                 ( (  \.---./  ) )
#                  '.__/o   o\__.'
#                     {=  ^  =}
#                      >  -  <
#                     /       \
#                    //       \\
#                   //|   .   |\\
#                   "'\       /'"_.-~^`'-.
#                      \  _  /--'         `
#                    ___)( )(___
#                   (((__) (__)))
#              高山仰止,景行行止.虽不能至,心向往之。
import os
import numpy as np
from matplotlib import gridspec
# import matplotlib.pyplot as plt
#
# fig2 = plt.figure(constrained_layout=True)
# plt.rcParams['figure.figsize'] = (18.0, 9.0)
# spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2)
# f2_ax1 = fig2.add_subplot(spec2[0, 0])
# f2_ax1.tick_params(direction='out', length=6, width=2)
# f2_ax2 = fig2.add_subplot(spec2[0, 1])
# f2_ax3 = fig2.add_subplot(spec2[1, 0])
# f2_ax4 = fig2.add_subplot(spec2[1, 1])
# plt.show()
# exit()
import _init_paths
from networks.tools import obtain_hoi_to_obj, visual_tsne, obtain_hoi_to_verb, visual_tsne1, \
    visual_tsne_multi

if __name__ == '__main__':
    import matplotlib
    # print(matplotlib.get_backend())

    import pickle

    # model_name = 'iCAN_R_union_multi_ml1_rew_aug5_3_x5new'
    # model_name = 'iCAN_R_union_multi_base_rew_aug5_3_x5new'
    model_name = 'iCAN_R_union_l2_zs7_epic2_s0_7_vloss2_gall_gc_var_gan_dax_rew49_zws_randso2_aug5_3_x5new_res101'
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    print(model_name)
    # _train_
    # feats = pickle.load(open('/opt/data/private/{}_train_HICO_HO_feats.pkl'.format(model_name), 'rb'))
    if not os.path.exists('/project/ZHIHOU/feats/{}_train_HICO_HO_feats.pkl'.format(model_name)):
        print('you should first extract the feature by extract_HO_feature.py')
    feats = pickle.load(open('/project/ZHIHOU/feats/{}_train_HICO_HO_feats.pkl'.format(model_name), 'rb'))

    hoi_to_obj, obj_names = obtain_hoi_to_obj()
    hoi_to_verb, verb_names = obtain_hoi_to_verb()
    feats['O_list'] = [feats['O_list'][i] for i in range(len(feats['A_list'])) if np.sum(feats['A_list'][i]) == 1]
    feats['fake_O_list'] = [feats['fake_O_list'][i] for i in range(len(feats['A_list'])) if np.sum(feats['A_list'][i]) == 1]
    feats['A_list'] = [feats['A_list'][i] for i in range(len(feats['A_list'])) if
                            np.sum(feats['A_list'][i]) == 1]
    gt_ho = np.asarray(feats['A_list'])

    action_lists = [np.argwhere(a)[0] for a in gt_ho]

    # print(action_lists)
    obj_lists = [hoi_to_obj[action[0]] for action in action_lists]
    from collections import Counter
    obj_counter = Counter(obj_lists)
    obj_ids = sorted(obj_counter, key=lambda x: obj_counter[x])
    obj_ids = obj_ids[::-1]

    verb_lists = [hoi_to_verb[action[0]] for action in action_lists]
    # print(obj_lists)
    print(len(obj_lists), 'successfully load')
    new_feats = []
    new_fake_feats = []
    n_obj_list = []
    zs_obj_ids = [4, 23, 28, 30, 33, 50, 52, 58, 59, 67, 71, 79]
    count_obj = {}
    obj_num_selected = 10
    if len(sys.argv) > 2:
        obj_num_selected = int(sys.argv[2])
    selected_nums = 100
    if len(sys.argv) > 3:
        selected_nums = int(sys.argv[3])
    selected_obj_ids = obj_ids[:obj_num_selected]
    verb_lists_list = []
    zs_obj_lists = []
    for ii in range(len(obj_lists)):
        if obj_lists[ii] in selected_obj_ids:
            if obj_lists[ii] in count_obj:
                count_obj[obj_lists[ii]] += 1
            else:
                count_obj[obj_lists[ii]] = 1
            if count_obj[obj_lists[ii]] > 100:
                continue
            new_feats.append(feats['O_list'][ii])
            new_fake_feats.append(feats['fake_O_list'][ii])
            n_obj_list.append(selected_obj_ids.index(obj_lists[ii]))
            verb_lists_list.append(verb_lists[ii])
            if obj_lists[ii] in zs_obj_ids:
                zs_obj_lists.append(1)
            else:
                zs_obj_lists.append(0)
    length = 200000
    verb_list_ids = list(set(verb_lists_list))
    verb_lists_list = [verb_list_ids.index(item) for item in verb_lists_list]
    y2 = ['o'] * min(len(new_feats), length) + ['D'] * min(len(new_fake_feats), length)
    import matplotlib.pyplot as plt

    f2_ax_list = []
    if model_name.__contains__('zs'):
        # plt.rcParams['figure.figsize'] = (8.0, 6.0)
        plt.rcParams['figure.figsize'] = (18.0, 15.0)
        fig2 = plt.figure(constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax1.set_title('(a-1)', fontsize=20)
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        f2_ax2.set_title('(a-2)', fontsize=20)
        f2_ax3 = fig2.add_subplot(spec2[0, 2])
        f2_ax3.set_title('(a-3)', fontsize=20)

        f2_ax7 = fig2.add_subplot(spec2[1, 0])
        f2_ax7.set_title('(b-1)', fontsize=20)
        f2_ax8 = fig2.add_subplot(spec2[1, 1])
        f2_ax8.set_title('(b-2)', fontsize=20)
        f2_ax9 = fig2.add_subplot(spec2[1, 2])
        f2_ax9.set_title('(b-3)', fontsize=20)

        f2_ax4 = fig2.add_subplot(spec2[2, 0])
        f2_ax4.set_title('(c-1)', fontsize=20)
        f2_ax5 = fig2.add_subplot(spec2[2, 1])
        f2_ax5.set_title('(c-2)', fontsize=20)
        f2_ax6 = fig2.add_subplot(spec2[2, 2])
        f2_ax6.set_title('(c-3)', fontsize=20)
        f2_ax_list = [f2_ax1, f2_ax2, f2_ax3, f2_ax4, f2_ax5, f2_ax6,
                      f2_ax7, f2_ax8,f2_ax9]
    else:
        # plt.rcParams['figure.figsize'] = (8.0, 4.0)
        plt.rcParams['figure.figsize'] = (18.0, 10.0)
        fig2 = plt.figure(constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax1.set_title('(a-1)', fontsize=20)
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        f2_ax2.set_title('(a-2)', fontsize=20)
        f2_ax3 = fig2.add_subplot(spec2[0, 2])
        f2_ax3.set_title('(a-3)', fontsize=20)
        f2_ax4 = fig2.add_subplot(spec2[1, 0])
        f2_ax4.set_title('(b-1)', fontsize=20)
        f2_ax5 = fig2.add_subplot(spec2[1, 1])
        f2_ax5.set_title('(b-2)', fontsize=20)
        f2_ax6 = fig2.add_subplot(spec2[1, 2])
        f2_ax6.set_title('(b-3)', fontsize=20)
        f2_ax_list = [f2_ax1, f2_ax2, f2_ax3, f2_ax4, f2_ax5, f2_ax6]
    for ax in f2_ax_list:
        ax.tick_params(labelsize=15)
    # plt.rcParams['figure.figsize'] = (8.0, 4.0)
    # plt.clf()
    visual_tsne(np.asarray(new_feats[:length]), np.asarray(n_obj_list[:length]), obj_num_selected,
                model_name + 'O_list' + str(obj_num_selected), save_fig=True, old_plt=f2_ax1)
    visual_tsne(np.asarray(new_fake_feats[:length]), np.asarray(n_obj_list[:length]), obj_num_selected,
                model_name + 'fake_O_list' + str(obj_num_selected), save_fig=True, old_plt=f2_ax2)
    visual_tsne1(np.asarray(new_feats[:length] + new_fake_feats[:length]),
                 np.asarray(n_obj_list[:length] + n_obj_list[:length]), y2, obj_num_selected,
                 model_name + 'joint_O_list_' + str(obj_num_selected), save_fig=True, old_plt=f2_ax3)


    if model_name.__contains__('zs11'):
        visual_tsne_multi(np.asarray(new_feats[:length]), np.asarray(n_obj_list[:length]),
                          np.asarray(zs_obj_lists[:length]),
                          2, obj_num_selected,
                          model_name + 'O_list_zs_' + str(obj_num_selected), save_fig=True, old_plt=f2_ax7)
        visual_tsne_multi(np.asarray(new_fake_feats[:length]), np.asarray(n_obj_list[:length]),
                          np.asarray(zs_obj_lists[:length]),
                          2, obj_num_selected,
                    model_name + 'fake_O_list_zs_' + str(obj_num_selected), save_fig=True, old_plt=f2_ax8)
        # visual_tsne_multi(np.asarray(new_fake_feats[:length]), np.asarray(n_obj_list[:length]),
        #                   np.asarray(zs_obj_lists[:length]),
        #                   2, obj_num_selected, model_name + 'Multi_list_zs_' + str(obj_num_selected),
        #                   save_fig=True)
        visual_tsne_multi(np.asarray(new_feats[:length] + new_fake_feats[:length]),
                          np.asarray(n_obj_list[:length] + n_obj_list[:length]),
                          np.asarray(zs_obj_lists[:length] + zs_obj_lists[:length]), 2, obj_num_selected,
                          model_name + 'joint_O_list_zs_' + str(obj_num_selected), save_fig=True, old_plt=f2_ax9)

    visual_tsne_multi(np.asarray(new_feats[:length]), np.asarray(n_obj_list[:length]), np.asarray(verb_lists_list[:length]),
                      len(verb_list_ids), obj_num_selected, model_name + 'Multi_list'+str(obj_num_selected), save_fig=True, old_plt=f2_ax4)
    visual_tsne_multi(np.asarray(new_fake_feats[:length]), np.asarray(n_obj_list[:length]),
                      np.asarray(verb_lists_list[:length]),
                      len(verb_list_ids), obj_num_selected, model_name + 'Multi_list_fake_' + str(obj_num_selected),
                      save_fig=True, old_plt=f2_ax5)
    visual_tsne_multi(np.asarray(new_feats[:length] + new_fake_feats[:length]),
                      np.asarray(n_obj_list[:length] + n_obj_list[:length]),
                      np.asarray(verb_lists_list[:length] + verb_lists_list[:length]),
                      len(verb_list_ids),
                      obj_num_selected,
                      model_name + 'Multi_joint_O_list_' + str(obj_num_selected), save_fig=True, old_plt=f2_ax6)

    fig2.savefig('/project/ZHIHOU/jpg_test/{}.pdf'.format(model_name), dpi=300)
    exit()
    num_inst = np.load("/data1/zhihou/num_inst.npy")
    rare_idx = np.where(num_inst <= 10)[0].tolist()
    num_threhold = 50
    rare_feats = []
    action_gt = []
    num_classes = 80
    count_key = [0 for i in range(num_classes)]
    for i in range(len(action_lists)):
        objs = set([hoi_to_obj[action_lists[i][j]] for j in range(len(action_lists[i]))])
        if len(objs) > 1:
            continue
        if hoi_to_obj[action_lists[i][0]] < num_classes and count_key[hoi_to_obj[action_lists[i][0]]] < num_threhold:
            rare_feats.append(feats['O_list'][i])
            action_gt.append(hoi_to_obj[action_lists[i][0]])
            count_key[hoi_to_obj[action_lists[i][0]]] += 1
            # action_gt.append(rare_idx.index(action_lists[i][0]))

    old_gt = action_gt
    old_feats = rare_feats
    action_gt = []
    rare_feats = []
    label_select = [ll for ll in list(range(num_classes)) if count_key[ll] >= num_threhold]
    label_select = [ll for ll in list(range(num_classes))]

    for i in range(len(old_gt)):
        if old_gt[i] in label_select:
            action_gt.append(label_select.index(old_gt[i]))
            rare_feats.append(old_feats[i])

    print(count_key, len(action_lists))
    print(count_key, len(action_lists), len(count_key), label_select, len(label_select))
    print(np.argwhere(np.asarray(count_key) < num_threhold))
    print(len(np.asarray(rare_feats)), '========', len(np.asarray(action_gt)))
    visual_tsne(np.asarray(rare_feats), np.asarray(action_gt),
                label_nums=len(label_select), title='o/'+model_name + 'O1_list_act_'+str(num_threhold)+'_117', save_fig=True)
