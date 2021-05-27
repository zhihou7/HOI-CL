# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# --------------------------------------------------------

"""
Change the HICO-DET detection results to the right format.
"""

import pickle
import numpy as np
import scipy.io as sio
import os
# HICO = None
from ult.tools import get_convert_matrix


def save_HICO(HICO, HICO_dir, classid, begin, finish, fuse_type='spho'):
    all_boxes = []
    for i in range(finish - begin + 1):
        total = []
        score = []
        for key, value in HICO.items():
            for element in value:
                if element[2] == classid:
                    temp = []
                    temp.append(element[0].tolist())  # Human box
                    temp.append(element[1].tolist())  # Object box
                    temp.append(int(key))  # image id
                    temp.append(int(i))  # action id (0-599)
                    # if fuse_type == 'spv':
                    #     preds = element[11]
                    # else:
                    preds = obtain_fuse_preds(element, fuse_type)
                    # preds = obtain_fuse_preds(element, fuse_type)
                    # cls_prob_sp * (cls_prob_O + cls_prob_H) + cls_prob_verbs
                    # preds = pSp * (pO + pH + pVerbs)
                    # preds = pSp * (pO + pH)
                    # preds = pSp
                    # preds = pO + pH
                    # preds = pSp * pVerbs
                    # preds = pVerbs
                    # print(preds, element[4], element[5])
                    temp.append(preds[begin - 1 + i] * element[4] * element[5])
                    total.append(temp)
                    score.append(preds[begin - 1 + i] * element[4] * element[5])

        idx = np.argsort(score, axis=0)[::-1]
        for i_idx in range(min(len(idx), 19999)):
            all_boxes.append(total[idx[i_idx]])
    savefile = HICO_dir + 'detections_' + str(classid).zfill(2) + '.mat'
    # print('length:', classid, len(all_boxes))
    sio.savemat(savefile, {'all_boxes': all_boxes})
    return all_boxes

verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix()
hoi_2_obj = {}
for i in range(600):
    for j in range(80):
        if obj_to_HO_matrix[j][i] > 0:
            hoi_2_obj[i] = j


def obtain_fuse_preds(element, fuse_type):
    preds = element[3]
    if fuse_type != 'preds':
        pH = element[6]
        pO = element[7]
        pSp = element[8]
        pHoi = element[9]
    if fuse_type == 'preds':
        preds = preds
    elif fuse_type == 'spho':
        preds = pSp * (pO + pH)
    elif fuse_type == 'ho':
        preds = pO + pH
    elif fuse_type == 'spv':
        preds = pSp * pHoi
    elif fuse_type == 'sp':
        preds = pSp
    elif fuse_type == 'v':
        preds = pHoi

    else:
        raise Exception('fuse_type error, you must select those types{spho, spv, sp, sphov}')
    return preds


def save_HICO3(HICO, HICO_dir, classid, begin, finish, fuse_type='spho'):
    # "spho" is from iCAN which includes three branch: sp, v, o
    global hoi_2_obj
    global obj_to_HO_matrix
    all_boxes = []
    ones = np.ones(600)
    for i in range(finish - begin + 1):
        total = []
        score = []
        for key, value in HICO.items():
            for element in value:
                if fuse_type == 'spv':
                    preds = element[11]
                else:
                    preds = obtain_fuse_preds(element, fuse_type)
                # st1 = time.time()
                obj_scores = element[12]  # here is the different
                objid = element[13]
                # objid = label_trans_map[objid] + 1
                objid += 1
                element[5] = obj_scores

                if objid == classid:
                    temp = []
                    temp.append(element[0].tolist())  # Human box
                    temp.append(element[1].tolist())  # Object box
                    temp.append(int(key))  # image id
                    temp.append(int(i))  # action id (0-599)
                    temp.append(preds[begin - 1 + i] * element[4] * element[5])
                    total.append(temp)
                    score.append(preds[begin - 1 + i] * element[4] * element[5])

        idx = np.argsort(score, axis=0)[::-1]
        for i_idx in range(min(len(idx), 19999)):
            all_boxes.append(total[idx[i_idx]])
    savefile = HICO_dir + 'detections_' + str(classid).zfill(2) + '.mat'
    # print('length:', classid, len(all_boxes))
    sio.savemat(savefile, {'all_boxes': all_boxes})
    return all_boxes


def Generate_HICO_detection3(HICO, HICO_dir, fuse_type, gpool, func_type = 0):

    if not os.path.exists(HICO_dir):
        os.makedirs(HICO_dir)

    # Remove previous results
    filelist = [ f for f in os.listdir(HICO_dir)]
    for f in filelist:
        os.remove(os.path.join(HICO_dir, f))


    params = [[1 ,161, 170], # 1 person
              [2 ,11,  24],# 2 bicycle
              [3, 66, 76 ],  # 3 car
    [ 4, 147, 160],  # 4 motorcycle
    [ 5, 1, 10],  # 5 airplane
    [ 6, 55, 65],  # 6 bus
    [ 7, 187, 194],  # 7 train
    [ 8, 568, 576],  # 8 truck
    [ 9, 32, 46],  # 9 boat
    [ 10, 563, 567],  # 10 traffic light
    [ 11, 326, 330],  # 11 fire_hydrant
    [ 12, 503, 506],  # 12 stop_sign
    [ 13, 415, 418],  # 13 parking_meter
    [ 14, 244, 247],  # 14 bench
    [ 15, 25, 31],  # 15 bird
    [ 16, 77, 86],  # 16 cat
    [ 17, 112, 129],  # 17 dog
    [ 18, 130, 146],  # 18 horse
    [ 19, 175, 186],  # 19 sheep
    [ 20, 97, 107],  # 20 cow
    [ 21, 314, 325],  # 21 elephant
    [ 22, 236, 239],  # 22 bear
    [ 23, 596, 600],  # 23 zebra
    [ 24, 343, 348],  # 24 giraffe
    [ 25, 209, 214],  # 25 backpack
    [ 26, 577, 584],  # 26 umbrella
    [ 27, 353, 356],  # 27 handbag
    [ 28, 539, 546],  # 28 tie
    [ 29, 507, 516],  # 29 suitcase
    [ 30, 337, 342],  # 30 Frisbee
    [ 31, 464, 474],  # 31 skis
    [ 32, 475, 483],  # 32 snowboard
    [ 33, 489, 502],  # 33 sports_ball
    [ 34, 369, 376],  # 34 kite
    [ 35, 225, 232],  # 35 baseball_bat
    [ 36, 233, 235],  # 36 baseball_glove
    [ 37, 454, 463],  # 37 skateboard
    [ 38, 517, 528],  # 38 surfboard
    [ 39, 534, 538],  # 39 tennis_racket
    [ 40, 47, 54],  # 40 bottle
    [ 41, 589, 595],  # 41 wine_glass
    [ 42, 296, 305],  # 42 cup
    [ 43, 331, 336],  # 43 fork
    [ 44, 377, 383],  # 44 knife
    [ 45, 484, 488],  # 45 spoon
    [ 46, 253, 257],  # 46 bowl
    [ 47, 215, 224],  # 47 banana
    [ 48, 199, 208],  # 48 apple
    [ 49, 439, 445],  # 49 sandwich
    [ 50, 398, 407],  # 50 orange
    [ 51, 258, 264],  # 51 broccoli
    [ 52, 274, 283],  # 52 carrot
    [ 53, 357, 363],  # 53 hot_dog
    [ 54, 419, 429],  # 54 pizza
    [ 55, 306, 313],  # 55 donut
    [ 56, 265, 273],  # 56 cake
    [ 57, 87, 92],  # 57 chair
    [ 58, 93, 96],  # 58 couch
    [ 59, 171, 174],  # 59 potted_plant
    [ 60, 240, 243],  # 60 bed
    [ 61, 108, 111],  # 61 dining_table
    [ 62, 551, 558],  # 62 toilet
    [ 63, 195, 198],  # 63 TV
    [ 64, 384, 389],  # 64 laptop
    [ 65, 394, 397],  # 65 mouse
    [ 66, 435, 438],  # 66 remote
    [ 67, 364, 368],  # 67 keyboard
    [ 68, 284, 290],  # 68 cell_phone
    [ 69, 390, 393],  # 69 microwave
    [ 70, 408, 414],  # 70 oven
    [ 71, 547, 550],  # 71 toaster
    [ 72, 450, 453],  # 72 sink
    [ 73, 430, 434],  # 73 refrigerator
    [ 74, 248, 252],  # 74 book
    [ 75, 291, 295],  # 75 clock
    [ 76, 585, 588],  # 76 vase
    [ 77, 446, 449],  # 77 scissors
    [ 78, 529, 533],  # 78 teddy_bear
    [ 79, 349, 352],  # 79 hair_drier
    [ 80, 559, 562],  # 80 toothbrush

              ]

    import datetime

    # from multiprocessing import Pool
    #
    # process_num = 16 if fuse_type == 'spv' else 2
    # # global pool
    # # if pool is None:
    # pool = Pool(processes=process_num)
    # def func(item):
    #
    #     save_HICO(HICO, HICO_dir, item[0], item[1], item[2])
    #
    from itertools import repeat

    # gpool.starmap(save_HICO1, zip(repeat(output_file), repeat(HICO_dir), params, repeat(fuse_type)))

    from sys import version_info
    print('Load HICO sucessfully', datetime.datetime.now())
    all_boxes = []
    for p in params:
        # print(p)
        res = save_HICO3(HICO, HICO_dir, p[0], p[1], p[2], fuse_type)
        # all_boxes.extend(res)
    # savefile = HICO_dir + 'detections.mat'
    # sio.savemat(savefile, {'all_boxes': all_boxes})
        # print('end', p)
    print("Finish save HICO", datetime.datetime.now())



def Generate_HICO_detection(output_file, HICO_dir, fuse_type, gpool):

    if not os.path.exists(HICO_dir):
        os.makedirs(HICO_dir)

    # Remove previous results
    filelist = [ f for f in os.listdir(HICO_dir)]
    for f in filelist:
        os.remove(os.path.join(HICO_dir, f))


    params = [[1 ,161, 170], # 1 person
              [2 ,11,  24],# 2 bicycle
              [3, 66, 76 ],  # 3 car
    [ 4, 147, 160],  # 4 motorcycle
    [ 5, 1, 10],  # 5 airplane
    [ 6, 55, 65],  # 6 bus
    [ 7, 187, 194],  # 7 train
    [ 8, 568, 576],  # 8 truck
    [ 9, 32, 46],  # 9 boat
    [ 10, 563, 567],  # 10 traffic light
    [ 11, 326, 330],  # 11 fire_hydrant
    [ 12, 503, 506],  # 12 stop_sign
    [ 13, 415, 418],  # 13 parking_meter
    [ 14, 244, 247],  # 14 bench
    [ 15, 25, 31],  # 15 bird
    [ 16, 77, 86],  # 16 cat
    [ 17, 112, 129],  # 17 dog
    [ 18, 130, 146],  # 18 horse
    [ 19, 175, 186],  # 19 sheep
    [ 20, 97, 107],  # 20 cow
    [ 21, 314, 325],  # 21 elephant
    [ 22, 236, 239],  # 22 bear
    [ 23, 596, 600],  # 23 zebra
    [ 24, 343, 348],  # 24 giraffe
    [ 25, 209, 214],  # 25 backpack
    [ 26, 577, 584],  # 26 umbrella
    [ 27, 353, 356],  # 27 handbag
    [ 28, 539, 546],  # 28 tie
    [ 29, 507, 516],  # 29 suitcase
    [ 30, 337, 342],  # 30 Frisbee
    [ 31, 464, 474],  # 31 skis
    [ 32, 475, 483],  # 32 snowboard
    [ 33, 489, 502],  # 33 sports_ball
    [ 34, 369, 376],  # 34 kite
    [ 35, 225, 232],  # 35 baseball_bat
    [ 36, 233, 235],  # 36 baseball_glove
    [ 37, 454, 463],  # 37 skateboard
    [ 38, 517, 528],  # 38 surfboard
    [ 39, 534, 538],  # 39 tennis_racket
    [ 40, 47, 54],  # 40 bottle
    [ 41, 589, 595],  # 41 wine_glass
    [ 42, 296, 305],  # 42 cup
    [ 43, 331, 336],  # 43 fork
    [ 44, 377, 383],  # 44 knife
    [ 45, 484, 488],  # 45 spoon
    [ 46, 253, 257],  # 46 bowl
    [ 47, 215, 224],  # 47 banana
    [ 48, 199, 208],  # 48 apple
    [ 49, 439, 445],  # 49 sandwich
    [ 50, 398, 407],  # 50 orange
    [ 51, 258, 264],  # 51 broccoli
    [ 52, 274, 283],  # 52 carrot
    [ 53, 357, 363],  # 53 hot_dog
    [ 54, 419, 429],  # 54 pizza
    [ 55, 306, 313],  # 55 donut
    [ 56, 265, 273],  # 56 cake
    [ 57, 87, 92],  # 57 chair
    [ 58, 93, 96],  # 58 couch
    [ 59, 171, 174],  # 59 potted_plant
    [ 60, 240, 243],  # 60 bed
    [ 61, 108, 111],  # 61 dining_table
    [ 62, 551, 558],  # 62 toilet
    [ 63, 195, 198],  # 63 TV
    [ 64, 384, 389],  # 64 laptop
    [ 65, 394, 397],  # 65 mouse
    [ 66, 435, 438],  # 66 remote
    [ 67, 364, 368],  # 67 keyboard
    [ 68, 284, 290],  # 68 cell_phone
    [ 69, 390, 393],  # 69 microwave
    [ 70, 408, 414],  # 70 oven
    [ 71, 547, 550],  # 71 toaster
    [ 72, 450, 453],  # 72 sink
    [ 73, 430, 434],  # 73 refrigerator
    [ 74, 248, 252],  # 74 book
    [ 75, 291, 295],  # 75 clock
    [ 76, 585, 588],  # 76 vase
    [ 77, 446, 449],  # 77 scissors
    [ 78, 529, 533],  # 78 teddy_bear
    [ 79, 349, 352],  # 79 hair_drier
    [ 80, 559, 562],  # 80 toothbrush

              ]

    import datetime

    # from multiprocessing import Pool
    #
    # process_num = 16 if fuse_type == 'spv' else 2
    # # global pool
    # # if pool is None:
    # pool = Pool(processes=process_num)
    # def func(item):
    #
    #     save_HICO(HICO, HICO_dir, item[0], item[1], item[2])
    #

    # gpool.starmap(save_HICO1, zip(repeat(output_file), repeat(HICO_dir), params, repeat(fuse_type)))

    from sys import version_info
    if version_info.major == 3:
        HICO = pickle.load(open(output_file, "rb"), encoding='latin1')
    else:
        HICO = pickle.load(open(output_file, "rb"))
    print('Load HICO sucessfully', datetime.datetime.now())
    for p in params:
        # print(p)
        save_HICO(HICO, HICO_dir, p[0], p[1], p[2], fuse_type)
        # print('end', p)
    # pool.close()
    # pool.join()
    # pool.terminate()
    # del pool
    # import gc
    # gc.collect()
    # pool.map(save_HICO, params)
    print("Finish save HICO", datetime.datetime.now())
