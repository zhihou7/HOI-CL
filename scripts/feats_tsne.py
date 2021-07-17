#!/usr/bin/env python
# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# ---------
import _init_paths
import numpy as np

from networks.tools import visual_tsne_multi
from ult.tools import visual_tsne, obtain_hoi_to_verb, obtain_hoi_to_obj
from ult.config import cfg

if __name__ == '__main__':

    import pickle


    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    print(model_name)
    # feats = pickle.load(open('/opt/data/private/{}_train_HICO_HO_feats.pkl'.format(model_name), 'rb'))
    feats = pickle.load(open(cfg.LOCAL_DATA + '/{}_train_HICO_HO_feats.pkl'.format(model_name), 'rb'))

    hoi_to_obj, obj_names = obtain_hoi_to_obj()
    hoi_to_verb, verb_names = obtain_hoi_to_verb()
    gt_ho = np.asarray(feats['A_list'])

    action_lists = [np.argwhere(a)[0] for a in gt_ho]

    obj_lists = [hoi_to_obj[action[0]] for action in action_lists]
    verb_lists = [hoi_to_verb[action[0]] for action in action_lists]
    length = 20000
    visual_tsne(np.asarray(feats['O_list'])[:length], np.asarray(obj_lists)[:length], 80, model_name + 'O_list', save_fig=True)

    visual_tsne(np.asarray(feats['V_list'])[:length], np.asarray(verb_lists)[:length], 117, model_name + 'V_list', save_fig=True)

    visual_tsne_multi(np.asarray(feats['V_list'][:length]), np.asarray(verb_lists[:length]), np.asarray(obj_lists)[:length],
                  80, 117, model_name + 'Multi_verb_list'+str(80), save_fig=True, old_plt=None)
