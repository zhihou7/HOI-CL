
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


def visual_tsne(X, y, label_nums = 80, title=  'tsne', save_fig=False, old_plt=None):
    import numpy as np
    import pickle
    hoi_to_obj, obj_names = obtain_hoi_to_obj()



    import numpy as np

    from matplotlib.ticker import NullFormatter
    from sklearn import manifold, datasets
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
    if old_plt is None:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (8.0, 4.0)
        plt.clf()
    else:
        plt = old_plt
    for i in range(label_nums):
        label = y == i
        # print(label, colors[i])
        print('label1:', i, colors[i])
        p = plt.scatter(Y[label, 0], Y[label, 1], c=colors[i])
        # p = plt.plot(Y[label, 0], Y[label, 1], c=colors[i], label=obj_names[i])
        handles.append(p)
    # plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0., fontsize = 'xx-small')
    # plt.legend(handles[:3], obj_names[:3], loc=10)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis('tight')
    if old_plt is not None:
        return
    plt.title(title)
    if save_fig:
        print('save')
        plt.savefig('/project/ZHIHOU/jpg_test/{}.eps'.format(title), dpi=300)
        print('save')
    else:
        plt.show()


def visual_tsne_multi(X, y, y2, label2_nums=80, label_nums = 80, title= 'tsne', save_fig=False, old_plt=None):
    import numpy as np
    from sklearn import manifold, datasets
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
    colors = cm.rainbow(np.linspace(0, 1, label_nums))
    colors_2 = cm.rainbow(np.linspace(0, 1, label2_nums))
    if label2_nums == 2:
        print(colors_2[0], colors_2[1])
    handles = []
    if old_plt is None:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (8.0, 4.0)
        plt.clf()
    else:
        plt = old_plt
    area = (30 * np.arange(0, label2_nums)) ** 2  # 0 to 15 point radii
    for i in range(label_nums):
        label = y == i
        # label1 = y2 == 1
        size = area[y2]
        print('label1:', i, colors[i])
        for j in range(label2_nums):
            label2 = y2 == j
            merge_label = np.logical_and(label, label2)
            p = plt.scatter(Y[merge_label, 0], Y[merge_label, 1], c=colors[i], edgecolors=colors_2[j])
            print('label2:', j, colors_2[j])
            # p = plt.plot(Y[label, 0], Y[label, 1], c=colors[i], label=obj_names[i])
            handles.append(p)
    if old_plt is not None:
        return
    plt.title(title)
    if save_fig:
        print('save')
        plt.savefig('/project/ZHIHOU/jpg_test/{}.eps'.format(title), dpi=300)
        print('save')
    else:
        plt.show()


def visual_tsne1(X, y, y2, label_nums = 80, title=  'tsne', save_fig=False, old_plt=None):
    import numpy as np
    import pickle
    hoi_to_obj, obj_names = obtain_hoi_to_obj()



    import numpy as np

    from matplotlib.ticker import NullFormatter
    from sklearn import manifold, datasets
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
    if old_plt is None:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (8.0, 4.0)
        plt.clf()
    else:
        plt = old_plt
    halp_len = int(len(y) / 2)
    for i in range(label_nums):
        # import ipdb;ipdb.set_trace()
        label = y[:halp_len] == i
        # label = label[:halp_len]
        # label2 = y2 == i
        # print(label, colors[i])
        p = plt.scatter(Y[:halp_len][label, 0], Y[:halp_len][label, 1], c=colors[i], marker = 'o')
        # p = plt.plot(Y[label, 0], Y[label, 1], c=colors[i], label=obj_names[i])
        handles.append(p)


    for i in range(label_nums):
        label = y[halp_len:] == i

        # label2 = y2 == i
        # print(label, colors[i])
        p = plt.scatter(Y[halp_len:][label, 0], Y[halp_len:][label, 1], c=colors[i], marker = 'D')
        # p = plt.plot(Y[label, 0], Y[label, 1], c=colors[i], label=obj_names[i])
        handles.append(p)
    # plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0., fontsize = 'xx-small')
    # plt.legend(handles[:3], obj_names[:3], loc=10)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis('tight')
    if old_plt is not None:
        return
    plt.title(title)
    if save_fig:
        plt.savefig('/project/ZHIHOU/jpg_test/{}.eps'.format(title), dpi=300)
    else:
        plt.show()


def get_convert_matrix_coco3(verb_class_num=24, obj_class_num=80):
    if verb_class_num == 24:
        set_list = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (3, 1), (3, 19), (3, 28), (3, 46), (3, 47),
                    (3, 48), (3, 49), (3, 51), (3, 52), (3, 54), (3, 55), (3, 56), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7),
                    (4, 8), (4, 9), (4, 18), (4, 21), (5, 68), (6, 33), (7, 64), (8, 47), (8, 48), (8, 49), (8, 50),
                    (8, 51), (8, 52), (8, 53), (8, 54), (8, 55), (8, 56), (9, 2), (9, 4), (9, 14), (9, 18), (9, 21),
                    (9, 25), (9, 27), (9, 29), (9, 57), (9, 58), (9, 60), (9, 61), (9, 62), (9, 64), (10, 31), (10, 32),
                    (10, 37), (10, 38), (11, 14), (11, 57), (11, 58), (11, 60), (11, 61), (12, 40), (12, 41), (12, 42),
                    (12, 46), (13, 1), (13, 25), (13, 26), (13, 27), (13, 29), (13, 30), (13, 31), (13, 32), (13, 33),
                    (13, 34), (13, 35), (13, 37), (13, 38), (13, 39), (13, 40), (13, 41), (13, 42), (13, 47), (13, 50),
                    (13, 68), (13, 74), (13, 75), (13, 78), (14, 30), (14, 33), (15, 43), (15, 44), (15, 45), (16, 1),
                    (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 11), (16, 14), (16, 15), (16, 16),
                    (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 24), (16, 25), (16, 26), (16, 27), (16, 28),
                    (16, 29), (16, 30), (16, 31), (16, 32), (16, 33), (16, 34), (16, 35), (16, 36), (16, 37), (16, 38),
                    (16, 39), (16, 40), (16, 41), (16, 42), (16, 43), (16, 44), (16, 45), (16, 46), (16, 47), (16, 48),
                    (16, 49), (16, 51), (16, 53), (16, 54), (16, 55), (16, 56), (16, 57), (16, 61), (16, 62), (16, 63),
                    (16, 64), (16, 65), (16, 66), (16, 67), (16, 68), (16, 73), (16, 74), (16, 75), (16, 77), (17, 35),
                    (17, 39), (18, 33), (19, 31), (19, 32), (20, 74), (21, 1), (21, 2), (21, 4), (21, 8), (21, 9), (21, 14),
                    (21, 15), (21, 16), (21, 17), (21, 18), (21, 19), (21, 21), (21, 25), (21, 26), (21, 27), (21, 28),
                    (21, 29), (21, 30), (21, 31), (21, 32), (21, 33), (21, 34), (21, 35), (21, 36), (21, 37), (21, 38),
                    (21, 39), (21, 40), (21, 41), (21, 42), (21, 43), (21, 44), (21, 45), (21, 46), (21, 47), (21, 48),
                    (21, 49), (21, 50), (21, 51), (21, 52), (21, 53), (21, 54), (21, 55), (21, 56), (21, 57), (21, 64),
                    (21, 65), (21, 66), (21, 67), (21, 68), (21, 73), (21, 74), (21, 77), (21, 78), (21, 79), (21, 80),
                    (22, 32), (22, 37), (23, 30), (23, 33)]
    elif verb_class_num == 21:
        set_list = [(0, 38), (1, 31), (1, 32), (2, 1), (2, 19), (2, 28), (2, 43), (2, 44), (2, 46), (2, 47), (2, 48),
                    (2, 49),
                    (2, 51), (2, 52), (2, 54), (2, 55), (2, 56), (2, 77), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7),
                    (3, 8),
                    (3, 9), (3, 18), (3, 21), (4, 68), (5, 33), (6, 64), (7, 43), (7, 44), (7, 45), (7, 47), (7, 48),
                    (7, 49),
                    (7, 50), (7, 51), (7, 52), (7, 53), (7, 54), (7, 55), (7, 56), (8, 2), (8, 4), (8, 14), (8, 18),
                    (8, 21),
                    (8, 25), (8, 27), (8, 29), (8, 57), (8, 58), (8, 60), (8, 61), (8, 62), (8, 64), (9, 31), (9, 32),
                    (9, 37),
                    (9, 38), (10, 14), (10, 57), (10, 58), (10, 60), (10, 61), (11, 40), (11, 41), (11, 42), (11, 46),
                    (12, 1),
                    (12, 25), (12, 26), (12, 27), (12, 29), (12, 30), (12, 31), (12, 32), (12, 33), (12, 34), (12, 35),
                    (12, 37), (12, 38), (12, 39), (12, 40), (12, 41), (12, 42), (12, 47), (12, 50), (12, 68), (12, 74),
                    (12, 75), (12, 78), (13, 30), (13, 33), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6),
                    (14, 7),
                    (14, 8), (14, 11), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21),
                    (14, 24),
                    (14, 25), (14, 26), (14, 27), (14, 28), (14, 29), (14, 30), (14, 31), (14, 32), (14, 33), (14, 34),
                    (14, 35), (14, 36), (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 43), (14, 44),
                    (14, 45), (14, 46), (14, 47), (14, 48), (14, 49), (14, 51), (14, 53), (14, 54), (14, 55), (14, 56),
                    (14, 57), (14, 61), (14, 62), (14, 63), (14, 64), (14, 65), (14, 66), (14, 67), (14, 68), (14, 73),
                    (14, 74), (14, 75), (14, 77), (15, 33), (15, 35), (15, 39), (16, 31), (16, 32), (17, 74), (18, 1),
                    (18, 2),
                    (18, 4), (18, 8), (18, 9), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18), (18, 19), (18, 21),
                    (18, 25),
                    (18, 26), (18, 27), (18, 28), (18, 29), (18, 30), (18, 31), (18, 32), (18, 33), (18, 34), (18, 35),
                    (18, 36), (18, 37), (18, 38), (18, 39), (18, 40), (18, 41), (18, 42), (18, 43), (18, 44), (18, 45),
                    (18, 46), (18, 47), (18, 48), (18, 49), (18, 50), (18, 51), (18, 52), (18, 53), (18, 54), (18, 55),
                    (18, 56), (18, 57), (18, 64), (18, 65), (18, 66), (18, 67), (18, 68), (18, 73), (18, 74), (18, 77),
                    (18, 78), (18, 79), (18, 80), (19, 32), (19, 37), (20, 30), (20, 33)]
    else:
        return
    import pickle
    import numpy as np
    verb_to_HO_matrix = np.zeros((len(set_list), verb_class_num))
    for i in range(len(set_list)):
        item = set_list[i]
        verb_to_HO_matrix[i][item[0]] = 1
    verb_to_HO_matrix = np.transpose(verb_to_HO_matrix)

    obj_to_HO_matrix = np.zeros((len(set_list), obj_class_num))
    for i in range(len(set_list)):
        item = set_list[i]
        obj_to_HO_matrix[i][item[1] - 1] = 1
    obj_to_HO_matrix = np.transpose(obj_to_HO_matrix)
    return verb_to_HO_matrix, obj_to_HO_matrix


def get_convert_matrix_coco(verb_class_num=29, obj_class_num=1):
    import pickle
    import numpy as np
    verb_to_HO_matrix = np.zeros((29, verb_class_num))
    hoi_to_vb = pickle.load(open(cfg.DATA_DIR + '/vcoco_to_vb.pkl', 'rb'))
    for k, v in hoi_to_vb.items():
        verb_to_HO_matrix[k][v] = 1
    verb_to_HO_matrix = np.transpose(verb_to_HO_matrix)

    obj_to_HO_matrix = np.zeros((29, obj_class_num))
    hoi_to_obj = pickle.load(open(cfg.DATA_DIR + '/vcoco_to_obj.pkl', 'rb'))
    for k, v in hoi_to_obj.items():
        obj_to_HO_matrix[k][v] = 1
    obj_to_HO_matrix = np.transpose(obj_to_HO_matrix)
    return verb_to_HO_matrix, obj_to_HO_matrix


if __name__ == '__main__':
    pass