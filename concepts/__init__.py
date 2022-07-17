from scripts import get_id_convert_dicts

DATA_DIR = './Data/'
import os

def cal_ap(gt_labels, affordance_probs_new, mask):
    from sklearn.metrics import average_precision_score

    # ap = average_precision_score(gt_labels.reshape(-1), affordance_stat_tmp1.reshape(-1))
    # print(ap)
    # exit()
    #
    mask = mask.reshape([-1])
    gt_labels = gt_labels.reshape(-1).tolist()
    affordance_probs_new = affordance_probs_new.reshape(-1).tolist()
    assert len(mask) == len(gt_labels) == len(affordance_probs_new)
    gt_labels = [gt_labels[i] for i in range(len(mask)) if mask[i] == 0]
    affordance_probs_new = [affordance_probs_new[i] for i in range(len(mask)) if mask[i] == 0]

    ap = average_precision_score(gt_labels, affordance_probs_new)
    return ap


def stat_ap(v_o_matrix, log=True):
    id_vb, id_obj, id_hoi, hoi_to_obj, hoi_to_verbs = get_id_convert_dicts()
    gt_label_file = open(DATA_DIR + 'HICO_concepts.csv')
    import numpy as np
    gt_labels = np.zeros([117, 80], np.float)
    concept_gt_pairs = []
    for line in gt_label_file.readlines():
        arrs = line.split(' ')
        v = arrs[1]
        o = arrs[2]
        gt_labels[int(v)][int(o)] = 1.
        concept_gt_pairs.append((int(v), int(o)))
    # stat AP
    hico_id_pairs = []
    zs_id_pairs = []
    for i in range(600):
        hico_id_pairs.append((hoi_to_verbs[i], hoi_to_obj[i]))
    mask = np.zeros([117, 80], np.float32)
    for v, o in hico_id_pairs:
        mask[v][o] = 1.
    ap_new = cal_ap(gt_labels, v_o_matrix, mask)
    ap_all = cal_ap(gt_labels, v_o_matrix, np.zeros([117, 80], np.float32))
    affordance_stat_tmp = v_o_matrix.reshape([117, 80])
    gt_labels_known = np.zeros([117, 80], np.float32)
    for v, o in hico_id_pairs:
        gt_labels_known[v][o] = 1.
    ap_all_know = cal_ap(gt_labels_known, affordance_stat_tmp, np.zeros([117, 80], np.float32))


    tmp_max_v = np.max(affordance_stat_tmp)
    for v, o in hico_id_pairs:
        affordance_stat_tmp[v][o] = 1. + tmp_max_v
    ap_all_fix = cal_ap(gt_labels, affordance_stat_tmp, np.zeros([117, 80], np.float32))
    # f = open(DATA_DIR + '/hico_concepts.txt', 'a')
    # f.write('{} {}\t{:.4}\t{:.4}\t{:.4}\n'.format('discover', 'concept', ap_new, ap_all, ap_all_fix))
    # f.close()
    if log:
        print(ap_new, ap_all, ap_all_fix, 'discover', np.sum(affordance_stat_tmp > 0), np.sum(affordance_stat_tmp == 0),
              np.sum(affordance_stat_tmp == -1), np.max(affordance_stat_tmp),)
    # pass
    return ap_new, ap_all, ap_all_fix, ap_all_know



def cal_loss(preds, hoi_concept_conds, hoi_labels, v_label_, ):
    # assert hoi_concept_labels.sum() == len(feats)
    hoi_concept_labels_not = torch.logical_not(hoi_concept_conds)
    # feats_not = feats[hoi_concept_labels_not]
    hoi_labels_not = hoi_labels[hoi_concept_labels_not]
    v_label_not = v_label_[hoi_concept_labels_not]
    preds_not = preds[hoi_concept_labels_not]
    if AFFORDANCE_RUNNING:
        label_not = label[hoi_concept_labels_not]
        concept_prob = concept_prob * running_affordance_stat
        concept_prob = torch.mean(concept_prob, dim=[-1, -2])
    # feats = feats[hoi_concept_labels]
    preds = preds[hoi_concept_conds]
    hoi_labels = hoi_labels[hoi_concept_conds]
    v_label_ = v_label_[hoi_concept_conds]
    if NEGATIVE_PREDS and len(v_label_not) > 0:
        import random
        # TODO 6 vs 10, 8
        selected_zeros = torch.tensor(random.sample(range(len(v_label_not)), min(len(preds), len(v_label_not)))).to(device)
        # selected_zeros = torch.randperm(len(feats_not), len(feats)//6) - 1
        # feats_not = feats_not[selected_zeros]
        hoi_labels_not = hoi_labels_not[selected_zeros]
        v_label_not = v_label_not[selected_zeros]
        v_label_not = torch.zeros_like(v_label_not).to(v_label_not.device)
        preds_not = preds_not[selected_zeros]

        # feats = torch.cat([feats, feats_not], dim=0)
        hoi_labels = torch.cat([hoi_labels, hoi_labels_not], dim=0)
        v_label_ = torch.cat([v_label_, v_label_not], dim=0)
        preds = torch.cat([preds, preds_not], dim=0)
    # preds = model(feats)
    if HOI_PREDICTION:
        loss = criteria(preds, hoi_labels)
    else:
        loss = criteria(preds, v_label_)

    # import ipdb;ipdb.set_trace()
    pt = torch.exp(-loss)
    alpha = 1.
    # gamma = torch.sqrt(torch.sum(verb_nums) / verb_nums)
    gamma = 4.
    F_loss = alpha * (1 - pt) ** gamma * loss
    return F_loss, preds, hoi_labels, v_label_

