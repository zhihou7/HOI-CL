#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Site    : 
# @File    : stat_vcoco_affordance.py
# @Software: PyCharm

import _init_paths

from networks.tools import get_convert_matrix as get_cooccurence_matrix
from ult.config import cfg

dataset = 'gtobj365_coco'
dataset = 'gtobj365'

import sys
num_thres = 10
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    # num_thres = int(sys.argv[1])
model_name = 'iCAN_R_union_vloss2_l2_rew_aug5_3_x5new_res101'
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

# from pycocotools.coco import COCO
# coco_annotation_path = cfg.LOCAL_DATA + '/Data/v-coco/coco/annotations/instances_train2014.json'
# coco = COCO(coco_annotation_path)
# cats = coco.loadCats(coco.getCatIds())
coco_annos_map = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}
coco_id_map_90_2_80 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27:
24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46
, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68,
79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

# [0, 4, 7, 8, 10, 12, 14, 15, 17, 21, 23, 25, 26, 36, 37, 38, 39, 41, 43, 45, 48, 49, 52, 54, 57, 58, 59, 62, 64, 65, 66, 67, 73, 75, 76, 85, 87, 89, 93, 94, 96, 98, 99, 104, 110, 111, 112, 114]

# [20,       53,          182,                 171,          365,         220,     334,     352,      29,      216,    23,       183,          300,      225,          282]
# ['glove', 'microphone', 'american football', 'strawberry', 'flashlight', 'tape', 'baozi', 'durian', 'boots', 'ship', 'flower', 'basketball', 'cheese', 'watermelon', 'camel']
obj365 = {224: 'scale', 220: 'tape', 217: 'chicken', 244: 'hurdle', 354: 'game board', 334: 'baozi', 360: 'target', 26: 'plants pot/vase', 209: 'toothbrush', 190: 'projector', 300: 'cheese', 166: 'candy', 352: 'durian', 279: 'dumbbell', 136: 'gas stove', 335: 'lion', 251: 'french fries', 27: 'bench', 83: 'power outlet', 58: 'faucet', 25: 'storage box', 330: 'crab', 237: 'helicopter', 362: 'chainsaw', 288: 'antelope', 280: 'hamimelon', 294: 'jellyfish', 200: 'kettle', 215: 'marker', 204: 'clutch', 283: 'lettuce', 138: 'toilet', 115: 'oven', 170: 'baseball', 85: 'drum', 88: 'hanger', 236: 'toaster', 22: 'bracelet', 261: 'cherry', 159: 'tissue ', 225: 'watermelon', 183: 'basketball', 128: 'cleaning products', 123: 'tent', 188: 'fire hydrant', 81: 'truck', 304: 'rice cooker', 331: 'microscope', 262: 'tablet', 73: 'stuffed animal', 228: 'golf ball', 247: 'CD', 273: 'eggplant', 44: 'bowl', 12: 'desk', 351: 'eagle', 43: 'slippers', 252: 'horn', 40: 'carpet', 234: 'notepaper', 232: 'peach', 346: 'saw', 144: 'surfboard', 210: 'facial cleanser', 265: 'corn', 169: 'folder', 214: 'violin', 64: 'watch', 10: 'glasses', 124: 'shampoo/shower gel', 131: 'pizza', 357: 'asparagus', 295: 'mushroom', 322: 'steak', 178: 'suitcase', 347: 'table tennis  paddle', 211: 'mango', 29: 'boots', 56: 'necklace', 327: 'noodles', 272: 'volleyball', 141: 'baseball bat', 264: 'nuts', 139: 'stroller', 155: 'pumpkin', 171: 'strawberry', 181: 'pear', 111: 'luggage', 54: 'sandals', 150: 'liquid soap', 13: 'handbag', 365: 'flashlight', 291: 'trombone', 116: 'remote', 140: 'shovel', 180: 'ladder', 74: 'cake', 292: 'pomegranate', 84: 'clock', 162: 'vent', 104: 'cymbal', 364: 'iron', 348: 'okra', 359: 'pasta', 126: 'lantern', 269: 'broom', 192: 'fire extinguisher', 177: 'snowboard', 277: 'rice', 245: 'swing', 82: 'cow', 63: 'van', 305: 'tuba', 15: 'book', 249: 'swan', 5: 'lamp', 303: 'race car', 213: 'egg', 253: 'avocado', 92: 'guitar', 246: 'radio', 2: 'sneakers', 342: 'eraser', 320: 'measuring cup', 312: 'sushi', 212: 'deer', 318: 'parrot', 168: 'scissors', 102: 'balloon', 317: 'tortoise/turtle', 285: 'meat balls', 148: 'cat', 315: 'electric drill', 341: 'comb', 191: 'sausage', 223: 'bar soap', 201: 'hamburger', 174: 'pepper', 227: 'router/modem', 316: 'spring rolls', 182: 'american football', 299: 'egg tart', 278: 'tape measure/ruler', 109: 'banana', 146: 'gun', 187: 'billiards', 11: 'picture/frame', 118: 'paper towel', 87: 'bus', 284: 'goldfish', 133: 'computer box', 21: 'potted plant', 216: 'ship', 356: 'ambulance', 99: 'dog', 286: 'medal', 298: 'butterfly', 308: 'hair dryer', 268: 'globe', 355: 'french horn', 275: 'board eraser', 94: 'tea pot', 106: 'telephone', 328: 'mop', 137: 'broccoli', 311: 'dolphin', 3: 'chair', 4: 'hat', 96: 'tripod', 51: 'traffic light', 208: 'hot dog', 90: 'pot/pan', 9: 'car', 30: 'dining table', 306: 'crosswalk sign', 121: 'tomato', 45: 'barrel/bucket', 161: 'washing machine', 337: 'polar bear', 49: 'tie', 350: 'monkey', 238: 'green beans', 203: 'cucumber', 163: 'cookies', 47: 'suv', 239: 'brush', 160: 'carrot', 165: 'tennis racket', 17: 'helmet', 66: 'sink', 36: 'stool', 23: 'flower', 157: 'radiator', 260: 'fishing rod', 147: 'Life saver', 338: 'lighter', 60: 'bread', 326: 'radish', 1: 'human', 93: 'traffic cone', 78: 'knife', 179: 'grapes', 79: 'cellphone', 274: 'trophy', 313: 'urinal', 8: 'cup', 185: 'paint brush', 105: 'mouse', 113: 'soccer', 164: 'cutting/chopping board', 221: 'wheelchair', 156: 'Accordion/keyboard/piano', 189: 'goose', 336: 'red cabbage', 16: 'plate', 254: 'saxophone', 77: 'laptop', 194: 'facial mask', 218: 'onion', 75: 'motorbike/motorcycle', 55: 'canned', 363: 'lobster', 135: 'toiletries', 242: 'earphone', 33: 'flag', 333: 'Bread/bun', 255: 'trumpet', 248: 'parking meter', 250: 'garlic', 143: 'skateboard', 198: 'pie', 332: 'barbell', 329: 'yak', 281: 'stapler', 130: 'tangerine', 151: 'zebra', 70: 'traffic sign', 6: 'bottle', 361: 'hotair balloon', 129: 'sailboat', 325: 'llama', 101: 'blackboard/whiteboard', 175: 'coffee machine', 319: 'flute', 345: 'pencil case', 219: 'ice cream', 65: 'combine with bowl', 132: 'kite', 53: 'microphone', 86: 'fork', 358: 'hoverboard', 205: 'blender', 167: 'skating and skiing shoes', 89: 'nightstand', 287: 'toothpaste', 323: 'poker card', 98: 'fan', 108: 'orange', 196: 'chopsticks', 302: 'pig', 176: 'bathtub', 20: 'glove', 202: 'golf club', 119: 'refrigerator', 290: 'rickshaw', 72: 'candle', 57: 'mirror', 142: 'microwave', 158: 'converter', 110: 'airplane', 149: 'lemon', 125: 'head phone', 235: 'tricycle', 259: 'bear', 37: 'backpack', 69: 'apple', 114: 'trolley', 206: 'tong', 307: 'papaya', 233: 'cello', 282: 'camel', 324: 'binoculars', 226: 'cabbage', 31: 'umbrella', 241: 'cigar', 301: 'pomelo', 7: 'cabinet/shelf', 95: 'keyboard', 67: 'horse', 152: 'duck', 117: 'combine with glove', 229: 'pine apple', 184: 'potato', 103: 'air conditioner', 270: 'pliers', 231: 'fire truck', 97: 'hockey stick', 134: 'elephant', 153: 'sports car', 48: 'toy', 339: 'mangosteen', 353: 'rabbit', 59: 'bicycle', 154: 'giraffe', 267: 'screwdriver', 100: 'spoon', 91: 'sheep', 266: 'key', 28: 'wine glass', 297: 'treadmill', 193: 'extension cord', 289: 'shrimp', 62: 'ring', 32: 'boat', 263: 'green vegetables', 46: 'coffee table', 343: 'pitaya', 321: 'shark', 41: 'basket', 76: 'wild bird', 240: 'carriage', 207: 'slide', 68: 'fish', 199: 'frisbee', 271: 'hammer', 186: 'printer', 222: 'plum', 42: 'towel/napkin', 71: 'camera', 34: 'speaker', 107: 'pickup truck', 61: 'high heels', 172: 'bow tie', 173: 'pigeon', 293: 'coconut', 122: 'machinery vehicle', 38: 'sofa', 50: 'bed', 195: 'tennis ball', 276: 'dates', 14: 'street lights', 80: 'paddle', 296: 'calculator', 349: 'starfish', 310: 'chips', 120: 'train', 258: 'kiwi fruit', 39: 'belt', 24: 'monitor', 112: 'skis', 18: 'leather shoes', 256: 'sandwich', 197: 'Electronic stove and gas stove', 243: 'penguin', 145: 'surveillance camera', 257: 'cue', 344: 'scallop', 309: 'green onion', 340: 'seal', 230: 'crane', 314: 'donkey', 52: 'pen/pencil', 127: 'donut', 19: 'pillow', 35: 'trash bin/can'}
obj365_set_list = { 20: [8, 9, 37, 51, 68, 115], # glove
                    53: [9, 37, 51, 68], # microphone
                    182: [3, 9, 10, 37, 45, 51, 68, 105], # american football
                    171:[8, 24, 37, 51, 57], # strawberry
                    365: [8, 37, 51, 68], # flashlight
                    220: [8, 37, 51, 68], # tape
                    334: [8, 24, 37, 51, 68], # baozi
                    352: [8, 37, 51, 68], # durian
                    29: [8, 37, 51, 68, 115,], # boots
                    216: [1, 5], # ship
                    23: [8, 37, 39, 51, 68], # flower
                    183: [3, 37, 45, 51, 68, 105, ], # basketball
                    # 300: [8, 16, 24, 37, 51, 55, 68], # cheese
                    # 225: [8, 16, 24, 37, 51], # watermelon
                    # 282: [27, 37, 77, 88, 111, 112, 113], # camel
                    # 335: [27, 37, 77, 88, 111, 112, 113] # lion
                    #51
}


unseen_idx = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                      294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                      338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                      429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                      463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                      537, 558, 559, 560, 561, 595, 596, 597, 598, 599]

obj_id = []

num_gt_hois = 0
new_verb_feats = []
# verb_label_list = []
# new_action_list = []
verb_to_HO_matrix, obj_to_HO_matrix = get_cooccurence_matrix()
#
# no_interactions = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107,
#                    111, 129, 146, 160, 170, 174, 186, 194, 198,
#                    208, 214, 224, 232, 235, 239, 243, 247, 252, 257,
#                    264, 273, 283, 290, 295, 305, 313, 325, 330, 336,
#                    342, 348, 352, 356, 363, 368, 376, 383, 389, 393,
#                    397, 407, 414, 418, 429, 434, 438, 445, 449, 453,
#                    463, 474, 483, 488, 502, 506, 516, 528, 533, 538,
#                    546, 550, 558, 562, 567, 576, 584, 588, 595, 600]
#

num_gt_verbs = [67.0, 43.0, 101.0, 101.0, 108.0, 50.0, 101.0, 28.0, 342.0, 115.0, 49.0, 102.0, 26.0, 78.0, 101.0, 109.0, 106.0, 101.0, 101.0, 103.0, 120.0, 108.0, 3.0, 120.0, 101.0, 101.0, 107.0, 79.0, 101.0, 1.0, 132.0, 19.0, 101.0, 47.0, 102.0, 109.0, 1649.0, 52.0, 97.0, 129.0, 61.0, 217.0, 13.0, 213.0, 139.0, 101.0, 51.0, 103.0, 101.0, 101.0, 103.0, 30.0, 103.0, 6.0, 101.0, 32.0, 4.0, 0.0, 113.0, 82.0, 30.0, 10.0, 101.0, 24.0, 59.0, 151.0, 57.0, 104.0, 62.0, 38.0, 120.0, 101.0, 123.0, 161.0, 16.0, 108.0, 608.0, 101.0, 121.0, 101.0, 101.0, 68.0, 20.0, 101.0, 2.0, 108.0, 113.0, 285.0, 5.0, 44.0, 18.0, 7.0, 5.0, 215.0, 106.0, 69.0, 37.0, 25.0, 291.0, 108.0, 1.0, 101.0, 101.0, 101.0, 117.0, 72.0, 75.0, 101.0, 101.0, 101.0, 151.0, 149.0, 174.0, 1.0, 137.0, 177.0, 1.0]

# print(num_gt_verbs)
# print(num_gt_verbs.shape)
# print(num_gt_verbs.tolist())
obj_verb = {}
for i in range(81):
    obj_verb[i] = []

fobj = open('../../Data/hico_list_obj.txt')
fhoi = open('../../Data/hico_list_hoi.txt')
fvb = open('../../Data/hico_list_vb.txt')
id_obj = {}
hoi_to_obj = {}
hoi_to_verbs = {}
obj_id = {}
id_hoi = {}
vb_id = {}
obj_to_hoi = {}
id_vb = {}
coco_to_hico_obj = {}
hico_obj_to_coco = {}
for line in fobj.readlines()[2:]:
    line = line.strip()
    # print(line.split(' '), [item for item in line.split(' ') if item != ''])

    cid, obj = [item for item in line.split(' ') if item != '']
    obj_id[obj] = int(cid) - 1
    id_obj[int(cid) - 1] = obj
    coco_obj = obj.replace('_', ' ')
    # print(coco_annos_map[coco_obj], obj, cid)
    coco_to_hico_obj[coco_id_map_90_2_80[coco_annos_map[coco_obj]] + 1] = int(cid) - 1
    hico_obj_to_coco[int(cid) - 1] = coco_id_map_90_2_80[coco_annos_map[coco_obj]] + 1

for line in fvb.readlines()[2:]:
    line = line.strip()
    cid, vb = [item for item in line.split(' ') if item != '']
    vb_id[vb] = int(cid) - 1
    id_vb[int(cid) - 1] = vb

for line in fhoi.readlines()[2:]:
    line = line.strip()
    cid, obj, vb = [item for item in line.split(' ') if item != '']
    hoi_to_obj[int(cid) - 1] = obj_id[obj]
    id_hoi[int(cid) - 1] = obj + ' ' + vb
    hoi_to_verbs[int(cid) - 1] = vb_id[vb]
    obj_verb[obj_id[obj]].append(vb)

for i in range(0, 80):
    # print(i, obj_verb[i])
    obj_verb[i].remove('no_interaction')

unseen_objs = list(set([hico_obj_to_coco[hoi_to_obj[k]] for k in unseen_idx]))
print(unseen_objs)


if dataset == 'gtobj365':
    obj_verb = {}
    for i in range(365 + 1):
        obj_verb[i] = []
    for k in obj365_set_list:
    # for i in range(len(obj365_set_list)):
        # print(obj365_set_list[i][1])
        for item in obj365_set_list[k]:
            obj_verb[k].append(id_vb[item-1])
    # obj_verb = obj365_set_list
print(obj_verb)

no_interactions = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107, 111, 129, 146, 160, 170, 174, 186, 194, 198,
                       208, 214, 224, 232, 235, 239, 243, 247, 252, 257, 264, 273, 283, 290, 295, 305, 313, 325,
                       330, 336, 342, 348, 352, 356, 363, 368, 376, 383, 389, 393, 397, 407, 414, 418, 429, 434,
                       438, 445, 449, 453, 463, 474, 483, 488, 502, 506, 516, 528, 533, 538, 546, 550, 558, 562,
                       567, 576, 584, 588, 595, 600]
no_interactions = [item - 1 for item in no_interactions]

import pickle

f_name = cfg.LOCAL_DATA + "/obj_hoi_map_new/{}_{}_obj_hoi_map_list.pkl".format(dataset, model_name)
import os
if dataset == 'gtobj365_coco' and not os.path.exists(f_name):

    obj_verb_map_list1 = pickle.load(open(cfg.LOCAL_DATA + "/obj_hoi_map_new2/{}_{}_obj_hoi_map_list.pkl".format(
        'gtobj365_coco_1', model_name), 'rb'))
    obj_verb_map_list2 = pickle.load(open(
        cfg.LOCAL_DATA + "/obj_hoi_map_new2/{}_{}_obj_hoi_map_list.pkl".format(
            'gtobj365_coco_2', model_name), 'rb'))

    obj_verb_map_list = obj_verb_map_list1 + obj_verb_map_list2
    assert len(obj_verb_map_list1) + len(obj_verb_map_list2) == len(obj_verb_map_list)
else:

    obj_verb_map_list = pickle.load(open(f_name, 'rb'))


def cal_prec(idx, map_list, num_thres):

    coco_objid = map_list[idx][0]  # 1-80
    verb_stat = [0]* 117
    import numpy as np
    verb_stat = np.matmul(np.asarray(map_list[idx][1]), verb_to_HO_matrix.transpose())

    verb_probs = [verb_stat[k] / num_gt_verbs[k] for k in range(len(verb_stat))]

    if dataset == 'gtobj365':
        # print(coco_objid)
        gt_verbs = set(obj_verb[coco_objid])
    else:
        gt_verbs = set(obj_verb[coco_to_hico_obj[coco_objid]])
    verbs = [id_vb[kk] for kk in range(117) if verb_probs[kk] > 0.9]
    if 'no_interaction' in verbs:
        verbs.remove('no_interaction')
    if len(verbs) == 0:
        return 0, 0
    prec = len(gt_verbs.intersection(set(verbs))) / len(set(verbs))
    recall = len(gt_verbs.intersection(set(verbs))) / len(set(gt_verbs))
    # print(prec, recall, gt_verbs, verbs)

    return prec, recall

def stat_prec_recall(num_thres):
    prec_list = []
    base_prec_list = []
    recall_list = []
    F1_list = []
    base_F1_list = []
    base_recall_list = []
    for i in range(len(obj_verb_map_list)):
        coco_objid = obj_verb_map_list[i][0]
        if dataset == 'gtobj365':
            # print(coco_objid)
            if len(obj_verb[coco_objid]) == 0:
                continue
        else:
            if len(obj_verb[coco_to_hico_obj[coco_objid]]) == 0:
                continue

        base_prec, base_recall = 1, 1
        # import ipdb;ipdb.set_trace()
        # base_prec, base_recall = cal_prec(i, obj_verb_map_list_base, num_thres)
        if model_name.__contains__('zs11'):
            if obj_verb_map_list[i][0] not in unseen_objs:
                continue
        prec, recall = cal_prec(i, obj_verb_map_list, num_thres)
        # import ipdb;
        # ipdb.set_trace()
        base_prec_list.append(base_prec)
        prec_list.append(prec)
        recall_list.append(recall)
        base_recall_list.append(base_recall)

        if prec + recall == 0:
            F1_list.append(0)
        else:
            F1_list.append(2*prec*recall / (prec+recall))
    return sum(base_prec_list) / len(base_prec_list), sum(prec_list) / len(prec_list), sum(base_recall_list) / len(base_recall_list), sum(recall_list)/ len(recall_list), 0, sum(F1_list)/len(F1_list)

b_f1_list = []
f1_list = []
base_prec, prec, base_recall, recall, base_F1_, F1_ = stat_prec_recall(100);
print('stat: {} {} {}'.format(prec, recall, F1_))
# f = open('hico_func_result.txt', 'a')
# f.write('{}\t{}\t{}\t{}\t{}\n'.format(dataset, model_name, "%.2f " % ( recall  * 100), "%.2f " % (prec  * 100), "%.2f " % (F1_ * 100)))
# f.close()

# [17, 21, 23, 30, 31, 35, 37, 39,  54, 73, 75, 80,]

# bus, cat, chair, donut, elephant, giraffe, handbag, hot_dog, remote, traffic_light, truck, zebra
# 'glove', 'microphone', 'american football', 'strawberry', 'flashlight', 'tape', 'baozi', 'durian', 'boots', 'ship', 'flower', 'basketball',

print()
print()
exit()

