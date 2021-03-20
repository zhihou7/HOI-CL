
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ult.config import cfg


def obtain_hoi_to_obj():
    fobj = open(cfg.DATA_DIR + '/hico_list_obj.txt')
    fhoi = open(cfg.DATA_DIR + '/hico_list_hoi.txt')
    fvb = open(cfg.DATA_DIR + '/hico_list_vb.txt')
    hoi_to_obj = {}
    obj_id = {}
    obj_name_lists = ['']*80

    for line in fobj.readlines()[2:]:
        line = line.strip()
        cid, obj = [item for item in line.split(' ') if item != '']
        obj_id[obj] = int(cid) - 1
        obj_name_lists[int(cid) - 1] = obj

    for line in fhoi.readlines()[2:]:
        line = line.strip()
        # print(line.split(' '), [item for item in line.split(' ') if item != ''])

        cid, obj, vb = [item for item in line.split(' ') if item != '']
        hoi_to_obj[int(cid) - 1] = obj_id[obj]

    return hoi_to_obj, obj_name_lists


def obtain_hoi_to_verb():
    # fobj = open(cfg.DATA_DIR + '/hico_list_obj.txt')
    fhoi = open(cfg.DATA_DIR + '/hico_list_hoi.txt')
    fvb = open(cfg.DATA_DIR + '/hico_list_vb.txt')
    hoi_to_verb = {}
    verb_id = {}
    verb_name_lists = ['']*117

    for line in fvb.readlines()[2:]:
        line = line.strip()
        cid, verb = [item for item in line.split(' ') if item != '']
        verb_id[verb] = int(cid) - 1
        # print(verb, verb_id[verb])
        verb_name_lists[int(cid) - 1] = verb

    for line in fhoi.readlines()[2:]:
        line = line.strip()
        # print(line.split(' '), [item for item in line.split(' ') if item != ''])

        cid, obj, vb = [item for item in line.split(' ') if item != '']
        hoi_to_verb[int(cid) - 1] = verb_id[vb]


    return hoi_to_verb, verb_name_lists


def get_word2vec():
    import pickle
    word2vec_pkl = cfg.LOCAL_DATA +  '/Data/coco_glove_word2vec.pkl'
    if not os.path.exists(word2vec_pkl):
        word2vec_pkl = cfg.LOCAL_DATA +  '/coco_glove_word2vec.pkl'
    # index is sorted by alphabet
    with open(word2vec_pkl, 'rb') as f:
        word2vec = pickle.load(f)
    return word2vec


def get_neighborhood_matrix(neighbor_num=5):
    word2vec_base = get_word2vec()
    word2vec = np.expand_dims(word2vec_base, axis=0)
    t1 = np.tile(word2vec, [80, 1, 1])

    word2vec1 = np.expand_dims(word2vec_base, axis=1)
    t2 = np.tile(word2vec1, [1, 80, 1])
    means = np.mean(np.square(t1 - t2), axis=-1)
    neighbors = np.argsort(means, axis=-1)[:, :neighbor_num + 1]
    matrix = np.zeros([80, 80], np.float32)
    for i in range(len(neighbors)):
        for j in range(len(neighbors[i])):
            matrix[i][neighbors[i][j]] = 1
    return matrix


def get_convert_matrix(verb_class_num=117, obj_class_num=80):
    import pickle
    import numpy as np
    verb_to_HO_matrix = np.zeros((600, verb_class_num), np.float32)
    hoi_to_vb = pickle.load(open(cfg.DATA_DIR + '/hoi_to_vb.pkl', 'rb'))
    for k, v in hoi_to_vb.items():
        verb_to_HO_matrix[k][v] = 1
    verb_to_HO_matrix = np.transpose(verb_to_HO_matrix)

    obj_to_HO_matrix = np.zeros((600, obj_class_num), np.float32)
    hoi_to_obj = pickle.load(open(cfg.DATA_DIR + '/hoi_to_obj.pkl', 'rb'))
    for k, v in hoi_to_obj.items():
        obj_to_HO_matrix[k][v] = 1
    obj_to_HO_matrix = np.transpose(obj_to_HO_matrix)
    return verb_to_HO_matrix, obj_to_HO_matrix


if __name__ == '__main__':
    pass