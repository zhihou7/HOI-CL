#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from scripts import _init_paths
from ult.config import cfg

dataset = 'gtobj365_coco'
# dataset = 'gtobj365'
# dataset = 'gtval2017'

import sys
num_thres = 10
model_name = 'VCL_R_union_multi_semi_ml5_l05_non_t5_def2_aug5_3_new_VCOCO_test_both_CL_21'

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=259638, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_R_union_batch_semi_l2_def1_epoch2_epic2_cosine5_s1_7_gc_gall_embloss_vloss2_var_gan_dax_xrew45_randso2_aug5_3_x5new_coco_res101_2', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0., type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    parser.add_argument('--dataset', dest='dataset',
                        help='Human threshold',
                        default='cocosub', type=str)
    parser.add_argument('--confidence', type=float, default=0.)
    parser.add_argument('--num', dest='num', default=100,
                        type=int)
    parser.add_argument('--pred_type', dest='pred_type', default=1,
                        type=int)
    parser.add_argument('--num_verbs', dest='num_verbs', default=100,
                        type=int)
    parser.add_argument('--incre_classes', dest='incre_classes',
                        help='Human threshold',
                        default=None, type=str)
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset
model_name = args.model
confidence = args.confidence
num_verbs = args.num_verbs
incre_classes = args.incre_classes
obj_confidence = confidence

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

set_list_old = set_list
if not model_name.__contains__('CL_21'):
    set_list_29 = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (4, 1), (4, 19), (4, 28), (4, 46), (4, 47),
                    (4, 48), (4, 49), (4, 51), (4, 52), (4, 54), (4, 55), (4, 56), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7),
                    (5, 8), (5, 9), (5, 18), (5, 21), (6, 68), (7, 33), (8, 64), (9, 47), (9, 48), (9, 49), (9, 50),
                    (9, 51), (9, 52), (9, 53), (9, 54), (9, 55), (9, 56), (10, 2), (10, 4), (10, 14), (10, 18), (10, 21),
                    (10, 25), (10, 27), (10, 29), (10, 57), (10, 58), (10, 60), (10, 61), (10, 62), (10, 64), (11, 31),
                    (11, 32), (11, 37), (11, 38), (12, 14), (12, 57), (12, 58), (12, 60), (12, 61), (13, 40), (13, 41),
                    (13, 42), (13, 46), (14, 1), (14, 25), (14, 26), (14, 27), (14, 29), (14, 30), (14, 31), (14, 32),
                    (14, 33), (14, 34), (14, 35), (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 47),
                    (14, 50), (14, 68), (14, 74), (14, 75), (14, 78), (15, 30), (15, 33), (9, 43), (9, 44), (9, 45),
                    (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 11), (18, 14), (18, 15),
                    (18, 16), (18, 17), (18, 18), (18, 19), (18, 20), (18, 21), (18, 24), (18, 25), (18, 26), (18, 27),
                    (18, 28), (18, 29), (18, 30), (18, 31), (18, 32), (18, 33), (18, 34), (18, 35), (18, 36), (18, 37),
                    (18, 38), (18, 39), (18, 40), (18, 41), (18, 42), (18, 43), (18, 44), (18, 45), (18, 46), (18, 47),
                    (18, 48), (18, 49), (18, 51), (18, 53), (18, 54), (18, 55), (18, 56), (18, 57), (18, 61), (18, 62),
                    (18, 63), (18, 64), (18, 65), (18, 66), (18, 67), (18, 68), (18, 73), (18, 74), (18, 75), (18, 77),
                    (19, 35), (19, 39), (20, 33), (21, 31), (21, 32), (23, 1), (23, 11), (23, 19), (23, 20), (23, 24),
                    (23, 28), (23, 34), (23, 49), (23, 53), (23, 56), (23, 61), (23, 63), (23, 64), (23, 67), (23, 68),
                    (23, 73), (24, 74), (25, 1), (25, 2), (25, 4), (25, 8), (25, 9), (25, 14), (25, 15), (25, 16), (25, 17),
                    (25, 18), (25, 19), (25, 21), (25, 25), (25, 26), (25, 27), (25, 28), (25, 29), (25, 30), (25, 31),
                    (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37), (25, 38), (25, 39), (25, 40), (25, 41),
                    (25, 42), (25, 43), (25, 44), (25, 45), (25, 46), (25, 47), (25, 48), (25, 49), (25, 50), (25, 51),
                    (25, 52), (25, 53), (25, 54), (25, 55), (25, 56), (25, 57), (25, 64), (25, 65), (25, 66), (25, 67),
                    (25, 68), (25, 73), (25, 74), (25, 77), (25, 78), (25, 79), (25, 80), (26, 32), (26, 37), (28, 30),
                    (28, 33)]

    l_map = {0: 0, 1: 1, 2: 2, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13,
                      16: 7, 18: 14, 19: 15, 20: 15, 21: 16, 24: 17, 25: 18, 26: 19, 28: 20}

    set_list = []
    for item in set_list_29:
        if item[0] in l_map:
            set_list.append((l_map[item[0]], item[1]))
        else:
            set_list.append((-1, item[1]))


# ('walk', 3)
# [('surf_instr', 0), ('ski_instr', 1), ('cut_instr', 2), ('walk', 3), ('cut_obj', 4), ('ride_instr', 5),
#  ('talk_on_phone_instr', 6), ('kick_obj', 7), ('work_on_computer_instr', 8), ('eat_obj', 9), ('sit_instr', 10),
#  ('jump_instr', 11), ('lay_instr', 12), ('drink_instr', 13), ('carry_obj', 14), ('throw_obj', 15),
#  ('eat_instr', 16), ('smile', 17), ('look_obj', 18), ('hit_instr', 19), ('hit_obj', 20), ('snowboard_instr', 21),
#  ('run', 22), ('point_instr', 23), ('read_obj', 24), ('hold_obj', 25), ('skateboard_instr', 26), ('stand', 27),
#  ('catch_obj', 28)]
# {0: 'surf_instr', 1: 'ski_instr', 2: 'cut_instr', 3: 'walk', 4: 'cut_obj', 5: 'ride_instr', 6: 'talk_on_phone_instr',
# 7: 'kick_obj', 8: 'work_on_computer_instr', 9: 'eat_obj', 10: 'sit_instr', 11: 'jump_instr', 12: 'lay_instr',
# 13: 'drink_instr', 14: 'carry_obj', 15: 'throw_obj', 16: 'eat_instr', 17: 'smile', 18: 'look_obj', 19: 'hit_instr',
# 20: 'hit_obj', 21: 'snowboard_instr', 22: 'run', 23: 'point_instr', 24: 'read_obj', 25: 'hold_obj', 26: 'skateboard_instr',
# 27: 'stand', 28: 'catch_obj'}
# vcoco = {0: 'surf_instr', 1: 'ski_instr', 2: 'cut_instr', 3: 'ride_instr', 4: 'talk_on_phone_instr',
#          5: 'kick_obj', 6: 'work_on_computer_instr', 7: 'eat_obj', 8: 'sit_instr', 9: 'jump_instr',
#          10: 'lay_instr', 11: 'drink_instr', 12: 'carry_obj', 13: 'throw_obj', 14: 'look_obj',
#          15: 'hit_instr', 16: 'snowboard_instr', 17: 'read_obj', 18: 'hold_obj',
#          19: 'skateboard_instr', 20: 'catch_obj'}



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
# if dataset == 'gtobj365':
#     set_list = obj365_set_list
#     num_objs = 365

hoi_verb = {}
hoi_obj = {}
obj_verb = {}
for i in range(80 + 1):
    obj_verb[i] = []
for i in range(len(set_list)):
    # print(i, len(set_list), set_list[i])
    hoi_verb[i] = set_list[i][0]
    hoi_obj[i] = set_list[i][1]
    if set_list[i][0]> 0:
        obj_verb[set_list[i][1]].append(vcoco[set_list[i][0]])

for i in range(len(set_list)):
    if hoi_verb[i] >= 0:
        num_gt_verbs[hoi_verb[i]] += num_gt_hois[set_list_old.index(set_list[i])]

if model_name.__contains__('t2'):
    hoi_verb = {0: 0, 1: 1, 2: 2, 3:-1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13,
                      16: 7, 17:-1, 18: 14, 19: 15, 20: 15, 21: 16, 22:-1, 23:-1, 24: 17, 25: 18, 26: 19, 27:-1, 28: 20}
# print(obj_verb);exit()
# print(num_gt_verbs)

if dataset == 'gtobj365':
    obj_verb = {}
    for i in range(365 + 1):
        obj_verb[i] = []
    for i in range(len(obj365_set_list)):
        # print(obj365_set_list[i][1])
        obj_verb[obj365_set_list[i][1]].append(vcoco[obj365_set_list[i][0]])

# print(obj_verb);exit()
import pickle
# obj_verb_map_list_base = pickle.load(open("/opt/data/private/VCL_R_union_multi_base_l05_t5_aug5_3_new_VCOCO_test_CL_21_obj_hoi_map_list.pkl", 'rb'))
# obj_verb_map_list = pickle.load(open("/opt/data/private/VCL_R_union_multi_semi_ml5_l05_t5_def2_aug5_3_new_VCOCO_test_both_CL_21_obj_hoi_map_list.pkl", 'rb'))



# obj_verb_map_list_base = pickle.load(open(cfg.LOCAL_DATA + "/obj_hoi_map/{}_VCL_R_union_multi_base_l05_t5_aug5_3_new_VCOCO_test_CL_21_obj_hoi_map_list.pkl".format(dataset), 'rb'))
# obj_verb_map_list = pickle.load(open("/opt/data/private/{}_VCL_R_union_multi_semi_ml5_l05_t5_def2_aug5_3_new_VCOCO_test_both_CL_21_obj_hoi_map_list.pkl".format(dataset), 'rb'))
# obj_verb_map_list = pickle.load(open("/opt/data/private/{}_VCL_R_union_multi_ml5_l05_t5_def2_aug5_3_new_VCOCO_test_CL_21_obj_hoi_map_list.pkl".format(dataset), 'rb'))
if dataset == 'gtobj365_coco':
    import os

    f_name = cfg.LOCAL_DATA + "/obj_hoi_map_new{}/{}_{}_obj_hoi_map_list.pkl".format(args.pred_type, dataset,
                                                                                     model_name)
    if os.path.exists(f_name):
        obj_verb_map_list = pickle.load(open(f_name, 'rb'))
        print(len(obj_verb_map_list))
        pass
    else:
        obj_verb_map_list1 = pickle.load(open(cfg.LOCAL_DATA + "/obj_hoi_map_new/{}_{}_obj_hoi_map_list.pkl".format(
            'gtobj365_coco_1', model_name), 'rb'))
        obj_verb_map_list2 = pickle.load(open(
            cfg.LOCAL_DATA + "/obj_hoi_map_new{}/{}_{}_obj_hoi_map_list.pkl".format(args.pred_type,
                'gtobj365_coco_2', model_name), 'rb'))

        obj_verb_map_list = obj_verb_map_list1 + obj_verb_map_list2
        assert len(obj_verb_map_list1) + len(obj_verb_map_list2) == len(obj_verb_map_list)
else:
    f_name = cfg.LOCAL_DATA + "/obj_hoi_map_new{}/{}_{}_obj_hoi_map_list.pkl".format(args.pred_type, dataset, model_name)
    print(f_name)
    obj_verb_map_list = pickle.load(open(f_name, 'rb'))




if len(obj_verb_map_list[0]) > 2:
    obj_verb_map_list = [item for item in obj_verb_map_list if item[2] >= obj_confidence]
# print('vcl')

max_num_thres = 0
for item in obj_verb_map_list:
    if max(item[1]) > max_num_thres:
        max_num_thres = max(item[1])

# print('max_num_thres', max_num_thres)

# obj_verb_map_list = pickle.load(open("/opt/data/private/{}_VCL_R_union_multi_semi_ml5_l05_t5_aug5_3_new_VCOCO_test_CL_21_obj_hoi_map_list.pkl".format(dataset), 'rb'))

# length = min(len(obj_verb_map_list_base), len(obj_verb_map_list))
# print('length of list:', length)
# obj_verb_map_list_base = obj_verb_map_list_base[:length]
# obj_verb_map_list = obj_verb_map_list[:length]



def cal_prec(idx, map_list, num_thres):
    # verbs = [vcoco[kk] for kk in set([hoi_verb[k] for k in range(len(map_list[idx][1])) if map_list[idx][1][k] > num_thres])]
    verb_stat = [0]* 21
    for k in range(len(map_list[idx][1])):
        if hoi_verb[k] >= 0:
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
    y_score = verb_probs
    y_gt = [1 if vcoco[kk] in gt_verbs else 0 for kk in range(len(verb_probs))]
    if len(set(verbs)) == 0:
        prec = 0
    else:
        prec = len(gt_verbs.intersection(set(verbs))) / len(set(verbs))
    recall = len(gt_verbs.intersection(set(verbs))) / len(set(gt_verbs))
    # print(prec, recall, gt_verbs, verbs)
    # print(gt_verbs, set(verbs))
    return prec, recall, y_score, y_gt

def stat_prec_recall(num_thres):
    prec_list = []
    base_prec_list = []
    recall_list = []
    F1_list = []
    base_recall_list = []
    y_score_list = []
    y_gt_list = []
    for idx in range(len(obj_verb_map_list)):
        if len(obj_verb[obj_verb_map_list[idx][0]]) == 0:
            continue
        # base_prec, base_recall = cal_prec(idx, obj_verb_map_list_base, num_thres)
        prec, recall, y_score, y_gt = cal_prec(idx, obj_verb_map_list, num_thres)
        # print(idx, '====', base_prec, base_recall, prec, recall)
        # base_prec_list.append(base_prec)
        y_score_list.append(y_score)
        y_gt_list.append(y_gt)
        if prec + recall == 0:
            F1_list.append(0)
        else:
            F1_list.append(2*prec*recall / (prec+recall))
        prec_list.append(prec)
        recall_list.append(recall)
        # base_recall_list.append(base_recall)
    return 0, sum(prec_list) / len(prec_list), 0, sum(recall_list)/ len(recall_list), 0, sum(F1_list)/len(F1_list), y_score_list, y_gt_list

b_f1_list = []
f1_list = []
base_prec, prec, base_recall, recall, base_F1_, F1_, y_score_list, y_gt_list = stat_prec_recall(100);
F1 = 2 * prec * recall / (prec + recall)
# b_F1 = 2 * base_prec * base_recall / (base_prec + base_recall)
# print('stat: {} {} {} {} {} {}'.format(base_prec, prec, base_recall, recall, b_F1, F1))
print('{} {} {}'.format("%.2f " % ( recall  * 100), "%.2f " % (prec  * 100), "%.2f " % (F1_ * 100)))


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import numpy as np
# For each class
precision_dict = dict()
recall_dict = dict()
average_precision = dict()
Y_test = np.asarray(y_gt_list)
y_score = np.asarray(y_score_list)
for i in range(21):
    if i == 57: # no interaction
        continue
    precision_dict[i], recall_dict[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


f = open('./vcoco_func_result.txt', 'a')
print('{} {} {} {}'.format("%.2f " % ( recall  * 100), "%.2f " % (prec  * 100), "%.2f " % (F1 * 100), "%.2f " % (F1_ * 100)))
f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(dataset, model_name, "%.2f " % ( recall  * 100), "%.2f " % (prec  * 100),
                                              "%.2f " % (F1 * 100), "%.2f " % (F1_ * 100), "%.2f " % (average_precision["micro"] * 100) ))
f.close()

f = open('./vcoco_func_result1.txt', 'a')
f.write('{}\t{}\t&\t{}\t&\t{}\t&\t{}\t&\t{}\n'.format(dataset, model_name, "%.2f " % ( recall  * 100), "%.2f " % (prec  * 100),
                                              "%.2f " % (F1_ * 100), "%.2f " % (average_precision["micro"] * 100) ))
f.close()

print()
print()
exit()
for i in range(1, max_num_thres):
    base_prec, prec, base_recall, recall = stat_prec_recall(i)
    # print(i, '====', base_prec, base_recall, prec, recall)
    if prec + recall == 0:
        continue
    F1 = 2 * prec * recall / (prec + recall)
    b_F1 = 2 * base_prec * base_recall / (base_prec + base_recall)
    f1_list.append(F1)
    b_f1_list.append(b_F1)
    print('stat: {} {} {} {} {} {} {} {} {}'.format(i, base_prec, prec, base_recall, recall, b_F1, F1, sum(b_f1_list)/ len(b_f1_list), sum(f1_list)/ len(f1_list)))
print(sum(b_f1_list)/ len(b_f1_list), sum(f1_list)/ len(f1_list))
# gtval2017 0.45561094697299553 0.823904977220483
# gtobj365  0.24141987380362212 0.49960644709275503
# gtobj365_coco 0.37350365292066495 0.80240146989106
exit()
n_classes = 81
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

# precision = dict()
# recall = dict()
# average_precision = dict()
#
# for i in range(1, n_classes):
#     base_res = print_affordance(i, num_thres)
#     res = print_affordance1(i, num_thres)
#     if len(obj_verb[i]) == 0:
#         print(obj[i-1], 0, 0)
#         continue
#     label_verb = [0] * 21
#     for ll in obj_verb[i]:
#         label_verb[ll] = 1
#
#     b_prec_list = []
#     for jj in range(11):
#         b_prec_list.append([print_affordance(i, jj * 1000)] )
#
#     precision[i], recall[i], _ = precision_recall_curve([label_verb] * len(b_prec_list),
#                                                         b_prec_list)
#     average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
#

# person 0.21052631578947367 0.3333333333333333
# bicycle 0.3333333333333333 0.6666666666666666
# car 0.1111111111111111 0.2222222222222222
# motorcycle 0.3333333333333333 0.6666666666666666
# airplane 0.1 0.3333333333333333
# bus 0.15384615384615385 0.6666666666666666
# train 0.2 0.6666666666666666
# truck 0.21428571428571427 0.75
# boat 0.125 0.2857142857142857
# traffic light 0 0
# fire hydrant 0.125 0.3333333333333333
# stop sign 0 0
# parking meter 0 0
# bench 0.2857142857142857 0.4
# bird 0.13333333333333333 0.2222222222222222
# cat 0.2857142857142857 0.6666666666666666
# dog 0.2 0.5
# horse 0.4 0.6666666666666666
# sheep 0.2 0.5
# cow 0.07142857142857142 0.16666666666666666
# elephant 0.3333333333333333 0.8
# bear 0 0
# zebra 0 0
# giraffe 0.1 0.25
# backpack 0.2857142857142857 0.4444444444444444
# umbrella 0.2 0.375
# handbag 0.3076923076923077 0.4444444444444444
# tie 0.14285714285714285 0.75
# suitcase 0.26666666666666666 0.4444444444444444
# frisbee 0.45454545454545453 0.8333333333333334
# skis 0.5 0.75
# snowboard 0.8571428571428571 0.8571428571428571
# sports ball 0.5384615384615384 0.7777777777777778
# kite 0.25 0.3
# baseball bat 0.3 0.5
# baseball glove 0.2 0.3333333333333333
# skateboard 0.4444444444444444 0.5555555555555556
# surfboard 0.35714285714285715 0.4166666666666667
# tennis racket 0.36363636363636365 0.4444444444444444
# bottle 0.23529411764705882 0.4444444444444444
# wine glass 0.25 0.6666666666666666
# cup 0.23529411764705882 0.4
# fork 0.36363636363636365 0.6666666666666666
# knife 0.26666666666666666 0.36363636363636365
# spoon 0.25 0.42857142857142855
# bowl 0.2 0.4444444444444444
# banana 0.3333333333333333 1.0
# apple 0.3076923076923077 0.8
# sandwich 0.4444444444444444 0.6666666666666666
# orange 0.18181818181818182 0.6
# broccoli 0.36363636363636365 1.0
# carrot 0.23076923076923078 0.5
# hot dog 0.3 0.6
# pizza 0.36363636363636365 0.6666666666666666
# donut 0.3076923076923077 0.5
# cake 0.3333333333333333 0.4444444444444444
# chair 0.21052631578947367 0.3076923076923077
# couch 0.16666666666666666 0.4
# potted plant 0 0
# bed 0.25 0.2857142857142857
# dining table 0.1875 0.23076923076923078
# toilet 0.16666666666666666 0.5
# tv 0.1 0.16666666666666666
# laptop 0.36363636363636365 0.5714285714285714
# mouse 0.2 1.0
# remote 0.11764705882352941 0.3333333333333333
# keyboard 0.18181818181818182 0.5
# cell phone 0.3333333333333333 0.6666666666666666
# microwave 0 0
# oven 0 0
# toaster 0 0
# sink 0 0
# refrigerator 0.2 1.0
# book 0.25 0.3333333333333333
# clock 0.14285714285714285 0.6666666666666666
# vase 0 0
# scissors 0.5 1.0
# teddy bear 0.16666666666666666 0.4
# hair drier 1.0 0.5
# toothbrush 0.1111111111111111 0.3333333333333333