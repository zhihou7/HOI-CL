# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os

os.environ['DATASET'] = 'HICO'

import numpy as np
import argparse
import pickle

from ult.timer import Timer
from ult.config import cfg
from networks.tools import get_convert_matrix as get_cooccurence_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='convert feats to verb bank on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=160000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='model', type=str)
    parser.add_argument('--num', dest='num', default=100,
                        type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print(args)

    print('Pre-trained weights loaded.')
    detection = {}
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0
    _t['im_detect'].tic()
    feats = pickle.load(open(cfg.LOCAL_DATA + '/feats/'+'{}_train_HOI_verb_feats.pkl'.format(args.model), 'rb'))
    verb_feats = feats['V_list']
    action_list = feats['A_list']
    print(feats['img_id_list'][:10])

    num_hoi_classes = 600
    new_verb_feats = []
    verb_label_list = []
    new_action_list = []
    verb_to_HO_matrix, obj_to_HO_matrix = get_cooccurence_matrix()

    no_interactions = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107,
                       111, 129, 146, 160, 170, 174, 186, 194, 198,
                       208, 214, 224, 232, 235, 239, 243, 247, 252, 257,
                       264, 273, 283, 290, 295, 305, 313, 325, 330, 336,
                       342, 348, 352, 356, 363, 368, 376, 383, 389, 393,
                       397, 407, 414, 418, 429, 434, 438, 445, 449, 453,
                       463, 474, 483, 488, 502, 506, 516, 528, 533, 538,
                       546, 550, 558, 562, 567, 576, 584, 588, 595, 600]
    no_interactions = set([item - 1 for item in no_interactions])

    select_nums = np.asarray([0]*117)

    for i in range(len(verb_feats)):
        hoi = action_list[i]
        verb_labels = np.matmul(hoi, verb_to_HO_matrix.transpose())
        has_no_inter = len(set(np.argwhere(hoi).reshape(-1).tolist()).intersection(no_interactions)) > 0
        if has_no_inter:
            continue
        verb_ids = np.argwhere(verb_labels).reshape(-1)
        if 57 in verb_ids:
            continue
        if min(select_nums[verb_ids]) > args.num:
            continue
        select_nums[verb_ids] += 1
        verb_label_list.append(np.argwhere(verb_labels).reshape(-1).tolist())
        new_verb_feats.append(verb_feats[i])
        new_action_list.append(action_list[i])

    action_list = new_action_list
    verb_feats = new_verb_feats
    num_hoi_classes = 600
    new_feats = {}
    new_feats['V_list'] = new_verb_feats
    new_feats['A_list'] = new_action_list
    assert len(new_verb_feats) == len(new_action_list)
    print(len(new_action_list))

    pickle.dump(new_feats, open(cfg.LOCAL_DATA + '/feats/' + '{}_train_HOI_verb_feats.pkl'.format(args.model), 'wb'))
