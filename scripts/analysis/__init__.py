DATA_DIR = './Data/'


def obtain_config(file_name):
    if file_name.__contains__('VCOCO') and file_name.__contains__('CL_24'):
        num_classes = 222
        verb_class_num = 24
        obj_class_num = 80

        gt_label_file = open(DATA_DIR + 'vcoco_concepts.csv')
        import numpy as np

        gt_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        gt_known_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        concept_gt_pairs = []
        for line in gt_label_file.readlines():
            arrs = line.split(' ')
            v = arrs[1]
            o = arrs[2]
            gt_labels[int(v)][int(o)] = 1.

            concept_gt_pairs.append((int(v), int(o)))
            if line.startswith('yes'):
                gt_known_labels[int(v)][int(o)] = 1.
    elif file_name.__contains__('VCOCO') and file_name.__contains__('CL_21'):
        convert_24_21 = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 7,
         16: 14, 17: 15, 18: 15, 19: 16, 20: 17, 21: 18, 22: 19, 23: 20}
        num_classes = 222
        verb_class_num = 21
        obj_class_num = 80

        gt_label_file = open(DATA_DIR + 'vcoco_concepts.csv')
        import numpy as np

        gt_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        gt_known_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        concept_gt_pairs = []
        for line in gt_label_file.readlines():
            arrs = line.split(' ')
            v = arrs[1]
            o = arrs[2]
            gt_labels[convert_24_21[int(v)]][int(o)] = 1.
            if line.startswith('yes'): gt_known_labels[convert_24_21[int(v)]][int(o)] = 1.
            concept_gt_pairs.append((convert_24_21[int(v)], int(o)))
    else:
        num_classes = 600
        verb_class_num = 117
        obj_class_num = 80

        gt_label_file = open(DATA_DIR + 'HICO_concepts.csv')
        import numpy as np

        gt_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        gt_known_labels = np.zeros([verb_class_num, obj_class_num], np.float)
        concept_gt_pairs = []
        for line in gt_label_file.readlines():
            arrs = line.split(' ')
            v = arrs[1]
            o = arrs[2]
            gt_labels[int(v)][int(o)] = 1.
            concept_gt_pairs.append((int(v), int(o)))
            if line.startswith('yes'):
                gt_known_labels[int(v)][int(o)] = 1.
    return num_classes, verb_class_num, obj_class_num, gt_labels, concept_gt_pairs, gt_known_labels
