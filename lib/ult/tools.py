# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# --------------------------------------------------------
import numpy as np

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

    # import json
    # cat_idx = json.load(open('/data1/zhihou/dataset/data/category.json'))

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

    # import json
    # cat_idx = json.load(open('/data1/zhihou/dataset/data/category.json'))

    return hoi_to_verb, verb_name_lists

def visual_tsne(X, y, label_nums = 80, title=  'tsne', save_fig=False):
    hoi_to_obj, obj_names = obtain_hoi_to_obj()



    import numpy as np

    from sklearn import manifold
    from time import time
    import matplotlib.cm as cm

    n_components = 2

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=10)
    Y = tsne.fit_transform(X)
    print(Y.shape)
    t1 = time()
    print("circles, perplexity=%d in %.2g sec" % (30, t1 - t0))
    # ax.set_title("Perplexity=30")
    colors = cm.rainbow(np.linspace(0, 1, label_nums))
    handles = []
    import matplotlib.pyplot as plt

    for i in range(label_nums):
        label = y == i
        # print(label, colors[i])
        p = plt.scatter(Y[label, 0], Y[label, 1], c=colors[i])
        # p = plt.plot(Y[label, 0], Y[label, 1], c=colors[i], label=obj_names[i])
        handles.append(p)
    # plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0., fontsize = 'xx-small')
    # plt.legend(handles[:3], obj_names[:3], loc=10)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis('tight')
    plt.title(title)
    if save_fig:
        plt.savefig(cfg.LOCAL_DATA + '/{}.jpg'.format(title))
    else:
        plt.show()


def get_word2vec():
    import pickle
    word2vec_pkl = cfg.LOCAL_DATA0 +  '/coco_glove_word2vec.pkl'
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


def get_neighborhood_matrix2(sorted_nums=480):
    """
    This is for word embedding similarity between objects
    :param sorted_nums:
    :return:
    """
    word2vec_base = get_word2vec()
    word2vec = np.expand_dims(word2vec_base, axis=0)
    t1 = np.tile(word2vec, [80, 1, 1])

    word2vec1 = np.expand_dims(word2vec_base, axis=1)
    t2 = np.tile(word2vec1, [1, 80, 1])
    means = np.mean(np.square(t1 - t2), axis=-1)
    # print(means.shape)
    a = np.reshape(means, (-1))
    b = np.sort(a)[sorted_nums]
    # print(b, np.sort(a))
    matrix = np.asarray(means < b, np.float32)
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


def get_convert_matrix_coco2(verb_class_num=29, obj_class_num=80):
    set_list = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (4, 1), (4, 19), (4, 28), (4, 46), (4, 47),
                (4, 48),
                (4, 49), (4, 51), (4, 52), (4, 54), (4, 55), (4, 56), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8),
                (5, 9), (5, 18), (5, 21), (6, 68), (7, 33), (8, 64), (9, 47), (9, 48), (9, 49), (9, 50), (9, 51),
                (9, 52),
                (9, 53), (9, 54), (9, 55), (9, 56), (10, 2), (10, 4), (10, 14), (10, 18), (10, 21), (10, 25), (10, 27),
                (10, 29), (10, 57), (10, 58), (10, 60), (10, 61), (10, 62), (10, 64), (11, 31), (11, 32), (11, 37),
                (11, 38), (12, 14), (12, 57), (12, 58), (12, 60), (12, 61), (13, 40), (13, 41), (13, 42), (13, 46),
                (14, 1),
                (14, 25), (14, 26), (14, 27), (14, 29), (14, 30), (14, 31), (14, 32), (14, 33), (14, 34), (14, 35),
                (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 47), (14, 50), (14, 68), (14, 74),
                (14, 75), (14, 78), (15, 30), (15, 33), (16, 43), (16, 44), (16, 45), (18, 1), (18, 2), (18, 3),
                (18, 4),
                (18, 5), (18, 6), (18, 7), (18, 8), (18, 11), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18),
                (18, 19),
                (18, 20), (18, 21), (18, 24), (18, 25), (18, 26), (18, 27), (18, 28), (18, 29), (18, 30), (18, 31),
                (18, 32), (18, 33), (18, 34), (18, 35), (18, 36), (18, 37), (18, 38), (18, 39), (18, 40), (18, 41),
                (18, 42), (18, 43), (18, 44), (18, 45), (18, 46), (18, 47), (18, 48), (18, 49), (18, 51), (18, 53),
                (18, 54), (18, 55), (18, 56), (18, 57), (18, 61), (18, 62), (18, 63), (18, 64), (18, 65), (18, 66),
                (18, 67), (18, 68), (18, 73), (18, 74), (18, 75), (18, 77), (19, 35), (19, 39), (20, 33), (21, 31),
                (21, 32), (23, 1), (23, 11), (23, 19), (23, 20), (23, 24), (23, 28), (23, 34), (23, 49), (23, 53),
                (23, 56),
                (23, 61), (23, 63), (23, 64), (23, 67), (23, 68), (23, 73), (24, 74), (25, 1), (25, 2), (25, 4),
                (25, 8),
                (25, 9), (25, 14), (25, 15), (25, 16), (25, 17), (25, 18), (25, 19), (25, 21), (25, 25), (25, 26),
                (25, 27),
                (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37),
                (25, 38), (25, 39), (25, 40), (25, 41), (25, 42), (25, 43), (25, 44), (25, 45), (25, 46), (25, 47),
                (25, 48), (25, 49), (25, 50), (25, 51), (25, 52), (25, 53), (25, 54), (25, 55), (25, 56), (25, 57),
                (25, 64), (25, 65), (25, 66), (25, 67), (25, 68), (25, 73), (25, 74), (25, 77), (25, 78), (25, 79),
                (25, 80), (26, 32), (26, 37), (28, 30), (28, 33)]
    # there are some duplicate verbs since v-coco contains two kinds of verb (v-instru  v-obj) where v might be similar.
    # But this do not affect the evaluation.
    import numpy as np
    verb_to_HO_matrix = np.zeros((238, verb_class_num))
    for i in range(len(set_list)):
        item = set_list[i]
        verb_to_HO_matrix[i][item[0]] = 1
    verb_to_HO_matrix = np.transpose(verb_to_HO_matrix)

    obj_to_HO_matrix = np.zeros((238, obj_class_num))
    for i in range(len(set_list)):
        item = set_list[i]
        obj_to_HO_matrix[i][item[1] - 1] = 1
    obj_to_HO_matrix = np.transpose(obj_to_HO_matrix)
    return verb_to_HO_matrix, obj_to_HO_matrix

if __name__ == '__main__':
    pass