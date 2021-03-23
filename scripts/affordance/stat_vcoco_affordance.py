#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Site    : 
# @File    : stat_vcoco_affordance.py
# @Software: PyCharm

import _init_paths
from ult.config import cfg

# dataset = 'gtobj365_coco'
dataset = 'gthico'
# dataset = 'gtobj365'
# dataset = 'gtval2017'

import sys
num_thres = 10
model_name = 'ATL_union_multi_atl_ml5_l05_t5_def2_aug5_new_VCOCO_coco_CL_21'
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    # num_thres = int(sys.argv[1])
if len(sys.argv) > 2:
    model_name = sys.argv[2]

print(dataset, model_name)
obj = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
       6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
       12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
       19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
       26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
       33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
       39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
       47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
       54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
       61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
       68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
       75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

vcoco = {0: 'surf_instr', 1: 'ski_instr', 2: 'cut_instr', 3: 'ride_instr', 4: 'talk_on_phone_instr',
         5: 'kick_obj', 6: 'work_on_computer_instr', 7: 'eat_obj', 8: 'sit_instr', 9: 'jump_instr',
         10: 'lay_instr', 11: 'drink_instr', 12: 'carry_obj', 13: 'throw_obj', 14: 'look_obj',
         15: 'hit_instr', 16: 'snowboard_instr', 17: 'read_obj', 18: 'hold_obj',
         19: 'skateboard_instr', 20: 'catch_obj'}

obj365 = {224: 'scale', 220: 'tape', 217: 'chicken', 244: 'hurdle', 354: 'game board', 334: 'baozi', 360: 'target', 26: 'plants pot/vase', 209: 'toothbrush', 190: 'projector', 300: 'cheese', 166: 'candy', 352: 'durian', 279: 'dumbbell', 136: 'gas stove', 335: 'lion', 251: 'french fries', 27: 'bench', 83: 'power outlet', 58: 'faucet', 25: 'storage box', 330: 'crab', 237: 'helicopter', 362: 'chainsaw', 288: 'antelope', 280: 'hamimelon', 294: 'jellyfish', 200: 'kettle', 215: 'marker', 204: 'clutch', 283: 'lettuce', 138: 'toilet', 115: 'oven', 170: 'baseball', 85: 'drum', 88: 'hanger', 236: 'toaster', 22: 'bracelet', 261: 'cherry', 159: 'tissue ', 225: 'watermelon', 183: 'basketball', 128: 'cleaning products', 123: 'tent', 188: 'fire hydrant', 81: 'truck', 304: 'rice cooker', 331: 'microscope', 262: 'tablet', 73: 'stuffed animal', 228: 'golf ball', 247: 'CD', 273: 'eggplant', 44: 'bowl', 12: 'desk', 351: 'eagle', 43: 'slippers', 252: 'horn', 40: 'carpet', 234: 'notepaper', 232: 'peach', 346: 'saw', 144: 'surfboard', 210: 'facial cleanser', 265: 'corn', 169: 'folder', 214: 'violin', 64: 'watch', 10: 'glasses', 124: 'shampoo/shower gel', 131: 'pizza', 357: 'asparagus', 295: 'mushroom', 322: 'steak', 178: 'suitcase', 347: 'table tennis  paddle', 211: 'mango', 29: 'boots', 56: 'necklace', 327: 'noodles', 272: 'volleyball', 141: 'baseball bat', 264: 'nuts', 139: 'stroller', 155: 'pumpkin', 171: 'strawberry', 181: 'pear', 111: 'luggage', 54: 'sandals', 150: 'liquid soap', 13: 'handbag', 365: 'flashlight', 291: 'trombone', 116: 'remote', 140: 'shovel', 180: 'ladder', 74: 'cake', 292: 'pomegranate', 84: 'clock', 162: 'vent', 104: 'cymbal', 364: 'iron', 348: 'okra', 359: 'pasta', 126: 'lantern', 269: 'broom', 192: 'fire extinguisher', 177: 'snowboard', 277: 'rice', 245: 'swing', 82: 'cow', 63: 'van', 305: 'tuba', 15: 'book', 249: 'swan', 5: 'lamp', 303: 'race car', 213: 'egg', 253: 'avocado', 92: 'guitar', 246: 'radio', 2: 'sneakers', 342: 'eraser', 320: 'measuring cup', 312: 'sushi', 212: 'deer', 318: 'parrot', 168: 'scissors', 102: 'balloon', 317: 'tortoise/turtle', 285: 'meat balls', 148: 'cat', 315: 'electric drill', 341: 'comb', 191: 'sausage', 223: 'bar soap', 201: 'hamburger', 174: 'pepper', 227: 'router/modem', 316: 'spring rolls', 182: 'american football', 299: 'egg tart', 278: 'tape measure/ruler', 109: 'banana', 146: 'gun', 187: 'billiards', 11: 'picture/frame', 118: 'paper towel', 87: 'bus', 284: 'goldfish', 133: 'computer box', 21: 'potted plant', 216: 'ship', 356: 'ambulance', 99: 'dog', 286: 'medal', 298: 'butterfly', 308: 'hair dryer', 268: 'globe', 355: 'french horn', 275: 'board eraser', 94: 'tea pot', 106: 'telephone', 328: 'mop', 137: 'broccoli', 311: 'dolphin', 3: 'chair', 4: 'hat', 96: 'tripod', 51: 'traffic light', 208: 'hot dog', 90: 'pot/pan', 9: 'car', 30: 'dining table', 306: 'crosswalk sign', 121: 'tomato', 45: 'barrel/bucket', 161: 'washing machine', 337: 'polar bear', 49: 'tie', 350: 'monkey', 238: 'green beans', 203: 'cucumber', 163: 'cookies', 47: 'suv', 239: 'brush', 160: 'carrot', 165: 'tennis racket', 17: 'helmet', 66: 'sink', 36: 'stool', 23: 'flower', 157: 'radiator', 260: 'fishing rod', 147: 'Life saver', 338: 'lighter', 60: 'bread', 326: 'radish', 1: 'human', 93: 'traffic cone', 78: 'knife', 179: 'grapes', 79: 'cellphone', 274: 'trophy', 313: 'urinal', 8: 'cup', 185: 'paint brush', 105: 'mouse', 113: 'soccer', 164: 'cutting/chopping board', 221: 'wheelchair', 156: 'Accordion/keyboard/piano', 189: 'goose', 336: 'red cabbage', 16: 'plate', 254: 'saxophone', 77: 'laptop', 194: 'facial mask', 218: 'onion', 75: 'motorbike/motorcycle', 55: 'canned', 363: 'lobster', 135: 'toiletries', 242: 'earphone', 33: 'flag', 333: 'Bread/bun', 255: 'trumpet', 248: 'parking meter', 250: 'garlic', 143: 'skateboard', 198: 'pie', 332: 'barbell', 329: 'yak', 281: 'stapler', 130: 'tangerine', 151: 'zebra', 70: 'traffic sign', 6: 'bottle', 361: 'hotair balloon', 129: 'sailboat', 325: 'llama', 101: 'blackboard/whiteboard', 175: 'coffee machine', 319: 'flute', 345: 'pencil case', 219: 'ice cream', 65: 'combine with bowl', 132: 'kite', 53: 'microphone', 86: 'fork', 358: 'hoverboard', 205: 'blender', 167: 'skating and skiing shoes', 89: 'nightstand', 287: 'toothpaste', 323: 'poker card', 98: 'fan', 108: 'orange', 196: 'chopsticks', 302: 'pig', 176: 'bathtub', 20: 'glove', 202: 'golf club', 119: 'refrigerator', 290: 'rickshaw', 72: 'candle', 57: 'mirror', 142: 'microwave', 158: 'converter', 110: 'airplane', 149: 'lemon', 125: 'head phone', 235: 'tricycle', 259: 'bear', 37: 'backpack', 69: 'apple', 114: 'trolley', 206: 'tong', 307: 'papaya', 233: 'cello', 282: 'camel', 324: 'binoculars', 226: 'cabbage', 31: 'umbrella', 241: 'cigar', 301: 'pomelo', 7: 'cabinet/shelf', 95: 'keyboard', 67: 'horse', 152: 'duck', 117: 'combine with glove', 229: 'pine apple', 184: 'potato', 103: 'air conditioner', 270: 'pliers', 231: 'fire truck', 97: 'hockey stick', 134: 'elephant', 153: 'sports car', 48: 'toy', 339: 'mangosteen', 353: 'rabbit', 59: 'bicycle', 154: 'giraffe', 267: 'screwdriver', 100: 'spoon', 91: 'sheep', 266: 'key', 28: 'wine glass', 297: 'treadmill', 193: 'extension cord', 289: 'shrimp', 62: 'ring', 32: 'boat', 263: 'green vegetables', 46: 'coffee table', 343: 'pitaya', 321: 'shark', 41: 'basket', 76: 'wild bird', 240: 'carriage', 207: 'slide', 68: 'fish', 199: 'frisbee', 271: 'hammer', 186: 'printer', 222: 'plum', 42: 'towel/napkin', 71: 'camera', 34: 'speaker', 107: 'pickup truck', 61: 'high heels', 172: 'bow tie', 173: 'pigeon', 293: 'coconut', 122: 'machinery vehicle', 38: 'sofa', 50: 'bed', 195: 'tennis ball', 276: 'dates', 14: 'street lights', 80: 'paddle', 296: 'calculator', 349: 'starfish', 310: 'chips', 120: 'train', 258: 'kiwi fruit', 39: 'belt', 24: 'monitor', 112: 'skis', 18: 'leather shoes', 256: 'sandwich', 197: 'Electronic stove and gas stove', 243: 'penguin', 145: 'surveillance camera', 257: 'cue', 344: 'scallop', 309: 'green onion', 340: 'seal', 230: 'crane', 314: 'donkey', 52: 'pen/pencil', 127: 'donut', 19: 'pillow', 35: 'trash bin/can'}

# [20, 53, 182, 171, 365, 220, 334, 352, 29, 216, 23, 183]
# ['glove', 'microphone', 'american football', 'strawberry', 'flashlight', 'tape',
# 'baozi', 'durian', 'boots', 'ship', 'flower', 'basketball']
obj365_set_list = [ (12, 20), (13, 20), (18, 20),
                    (4, 53), (12, 53), (13, 53), (14, 53), (18, 53),
                    (5,182), (12,182), (13,182),(14,182),(15,182), (18,182),
(2, 171), (7, 171), (12, 171), (13, 171), (18, 171),
(12, 365), (13, 365), (18, 365),
(12, 220), (13, 220), (18, 220),
(7, 334), (12, 334), (14, 334), (18, 334),
(7, 352), (12, 352), (18, 352),
(12, 29), (18, 29),
(3, 216), (8, 216), (10, 216), (14, 216),
(14, 23), (18, 23),
(13, 183),  (18, 183)

#                     29: [8, 37, 51, 68, 115, ],  # boots
#                         216: [1, 5],  # ship
# 23: [8, 37, 39, 51, 68],  # flower
# 183: [3, 37, 45, 51, 68, 105, ],  # basketball
# 300: [8, 16, 24, 37, 51, 55, 68],  # cheese
# 225: [8, 16, 24, 37, 51],  # watermelon
# 282: [27, 37, 77, 88, 111, 112, 113],  # camel
# 335: [27, 37, 77, 88, 111, 112, 113]  # lion
]


set_list = [(0, 38), (1, 31), (1, 32), (2, 1), (2, 19), (2, 28), (2, 43), (2, 44), (2, 46), (2, 47), (2, 48), (2, 49),
            (2, 51), (2, 52), (2, 54), (2, 55), (2, 56), (2, 77), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8),
            (3, 9), (3, 18), (3, 21), (4, 68), (5, 33), (6, 64), (7, 43), (7, 44), (7, 45), (7, 47), (7, 48), (7, 49),
            (7, 50), (7, 51), (7, 52), (7, 53), (7, 54), (7, 55), (7, 56), (8, 2), (8, 4), (8, 14), (8, 18), (8, 21),
            (8, 25), (8, 27), (8, 29), (8, 57), (8, 58), (8, 60), (8, 61), (8, 62), (8, 64), (9, 31), (9, 32), (9, 37),
            (9, 38), (10, 14), (10, 57), (10, 58), (10, 60), (10, 61), (11, 40), (11, 41), (11, 42), (11, 46), (12, 1),
            (12, 25), (12, 26), (12, 27), (12, 29), (12, 30), (12, 31), (12, 32), (12, 33), (12, 34), (12, 35),
            (12, 37), (12, 38), (12, 39), (12, 40), (12, 41), (12, 42), (12, 47), (12, 50), (12, 68), (12, 74),
            (12, 75), (12, 78), (13, 30), (13, 33), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7),
            (14, 8), (14, 11), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (14, 24),
            (14, 25), (14, 26), (14, 27), (14, 28), (14, 29), (14, 30), (14, 31), (14, 32), (14, 33), (14, 34),
            (14, 35), (14, 36), (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 43), (14, 44),
            (14, 45), (14, 46), (14, 47), (14, 48), (14, 49), (14, 51), (14, 53), (14, 54), (14, 55), (14, 56),
            (14, 57), (14, 61), (14, 62), (14, 63), (14, 64), (14, 65), (14, 66), (14, 67), (14, 68), (14, 73),
            (14, 74), (14, 75), (14, 77), (15, 33), (15, 35), (15, 39), (16, 31), (16, 32), (17, 74), (18, 1), (18, 2),
            (18, 4), (18, 8), (18, 9), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18), (18, 19), (18, 21), (18, 25),
            (18, 26), (18, 27), (18, 28), (18, 29), (18, 30), (18, 31), (18, 32), (18, 33), (18, 34), (18, 35),
            (18, 36), (18, 37), (18, 38), (18, 39), (18, 40), (18, 41), (18, 42), (18, 43), (18, 44), (18, 45),
            (18, 46), (18, 47), (18, 48), (18, 49), (18, 50), (18, 51), (18, 52), (18, 53), (18, 54), (18, 55),
            (18, 56), (18, 57), (18, 64), (18, 65), (18, 66), (18, 67), (18, 68), (18, 73), (18, 74), (18, 77),
            (18, 78), (18, 79), (18, 80), (19, 32), (19, 37), (20, 30), (20, 33)]


num_gt_hois = [485., 434., 3., 6., 6., 3., 3., 207., 1., 3., 4.,
               7., 1., 7., 32., 2., 160., 37., 67., 9., 126., 1.,
               24., 6., 31., 108., 73., 292., 134., 398., 86., 28., 39.,
               21., 3., 60., 4., 7., 1., 61., 110., 80., 56., 56.,
               119., 107., 96., 59., 2., 1., 4., 430., 136., 55., 1.,
               5., 1., 20., 165., 278., 26., 24., 1., 29., 228., 1.,
               15., 55., 54., 1., 2., 57., 52., 93., 72., 3., 7.,
               12., 6., 6., 1., 11., 105., 4., 2., 1., 1., 7.,
               1., 17., 1., 1., 2., 170., 91., 445., 6., 1., 2.,
               5., 1., 12., 4., 1., 1., 1., 14., 18., 7., 7.,
               5., 8., 4., 7., 4., 1., 3., 9., 390., 45., 156.,
               521., 15., 4., 5., 338., 254., 3., 5., 11., 15., 12.,
               43., 12., 12., 2., 2., 14., 1., 11., 37., 18., 134.,
               1., 7., 1., 29., 291., 1., 3., 4., 62., 4., 75.,
               1., 22., 228., 109., 233., 1., 366., 86., 50., 46., 68.,
               1., 1., 1., 1., 8., 14., 45., 2., 5., 45., 70.,
               89., 9., 99., 186., 50., 56., 54., 9., 120., 66., 56.,
               160., 269., 32., 65., 83., 67., 197., 43., 13., 26., 5.,
               46., 3., 6., 1., 60., 67., 56., 20., 2., 78., 11.,
               58., 1., 350., 1., 83., 41., 18., 2., 9., 1., 466.,
               224., 32.]

num_gt_verbs = [0]*21


num_objs = 80

hoi_verb = {}
hoi_obj = {}
obj_verb = {}
for i in range(80 + 1):
    obj_verb[i] = []
for i in range(len(set_list)):
    # print(i, len(set_list), set_list[i])
    hoi_verb[i] = set_list[i][0]
    hoi_obj[i] = set_list[i][1]

    obj_verb[set_list[i][1]].append(vcoco[set_list[i][0]])

for i in range(len(set_list)):
    num_gt_verbs[hoi_verb[i]] += num_gt_hois[i]


if dataset == 'gtobj365':
    obj_verb = {}
    for i in range(365 + 1):
        obj_verb[i] = []
    for i in range(len(obj365_set_list)):
        # print(obj365_set_list[i][1])
        obj_verb[obj365_set_list[i][1]].append(vcoco[obj365_set_list[i][0]])


import pickle
if dataset == 'gtobj365_coco':

    obj_verb_map_list1 = pickle.load(open(cfg.LOCAL_DATA + "/obj_hoi_map_new/{}_{}_obj_hoi_map_list.pkl".format(
        'gtobj365_coco_1', model_name), 'rb'))
    obj_verb_map_list2 = pickle.load(open(
        cfg.LOCAL_DATA + "/obj_hoi_map_new/{}_{}_obj_hoi_map_list.pkl".format(
            'gtobj365_coco_2', model_name), 'rb'))

    obj_verb_map_list = obj_verb_map_list1 + obj_verb_map_list2
    assert len(obj_verb_map_list1) + len(obj_verb_map_list2) == len(obj_verb_map_list)
else:
    f_name = cfg.LOCAL_DATA + "/obj_hoi_map_new/{}_{}_obj_hoi_map_list.pkl".format(dataset, model_name)
    print(f_name)
    obj_verb_map_list = pickle.load(open(f_name, 'rb'))


max_num_thres = 0
for item in obj_verb_map_list:
    if max(item[1]) > max_num_thres:
        max_num_thres = max(item[1])


def cal_prec(idx, map_list, num_thres):
    verb_stat = [0]* 21
    for k in range(len(map_list[idx][1])):
        verb_stat[hoi_verb[k]] += map_list[idx][1][k]
        pass
    verb_probs = [verb_stat[k] / num_gt_verbs[k] for k in range(len(verb_stat))]
    # print(verb_probs)
    gt_verbs = set(obj_verb[map_list[idx][0]])
    if len(gt_verbs) == 0:
        print(gt_verbs)
        import ipdb;ipdb.set_trace()
    verbs = [vcoco[kk] for kk in range(21) if verb_probs[kk] > 0.5]
    # print(map_list[idx][0], gt_verbs)
    # if dataset == 'gtobj365':
    #     print(idx, gt_verbs, verbs)
    #     return 1, 1
    if len(set(verbs)) == 0:
        prec = 0
    else:
        prec = len(gt_verbs.intersection(set(verbs))) / len(set(verbs))
    recall = len(gt_verbs.intersection(set(verbs))) / len(set(gt_verbs))
    # print(prec, recall, gt_verbs, verbs)
    # print(gt_verbs, set(verbs))
    return prec, recall

def stat_prec_recall(num_thres):
    prec_list = []
    base_prec_list = []
    recall_list = []
    F1_list = []
    base_recall_list = []
    for idx in range(len(obj_verb_map_list)):
        if len(obj_verb[obj_verb_map_list[idx][0]]) == 0:
            continue
        prec, recall = cal_prec(idx, obj_verb_map_list, num_thres)
        # print(idx, '====', base_prec, base_recall, prec, recall)
        # base_prec_list.append(base_prec)
        if prec + recall == 0:
            F1_list.append(0)
        else:
            F1_list.append(2*prec*recall / (prec+recall))
        prec_list.append(prec)
        recall_list.append(recall)
        # base_recall_list.append(base_recall)
    return 0, sum(prec_list) / len(prec_list), 0, sum(recall_list)/ len(recall_list), 0, sum(F1_list)/len(F1_list)

b_f1_list = []
f1_list = []
base_prec, prec, base_recall, recall, base_F1_, F1_ = stat_prec_recall(100);


print('{} {} {}'.format("%.2f " % ( recall  * 100), "%.2f " % (prec  * 100), "%.2f " % (F1_ * 100)))

print()
print()
exit()