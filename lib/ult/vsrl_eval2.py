# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta 
# 
# This file is part of the VCOCO dataset hooks and is available 
# under the terms of the Simplified BSD License provided in 
# LICENSE. Please retain this notice and LICENSE if you use 
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import numpy as np
from pycocotools.coco import COCO
import os, json
import copy
import pickle
import ipdb

from ult.config import cfg


class VCOCOeval(object):

  def __init__(self, vsrl_annot_file, coco_annot_file, 
      split_file):
    """Input:
    vslr_annot_file: path to the vcoco annotations
    coco_annot_file: path to the coco annotations
    split_file: image ids for split
    """
    self.set_list = [(0, 38), (1, 31), (1, 32), (2, 1), (2, 19), (2, 28), (2, 43), (2, 44), (2, 46), (2, 47), (2, 48), (2, 49),
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








    self.label_nums = np.asarray([485, 434, 3, 6, 6, 3, 3, 207, 1, 3, 4, 7, 1, 7, 32, 2, 160, 37, 67, 9, 126, 1, 24, 6, 31,
                       108, 73, 292, 134, 398, 86, 28, 39, 21, 3, 60, 4, 7, 1, 61, 110, 80, 56, 56, 119, 107, 96,
                       59, 2, 1, 4, 430, 136, 55, 1, 5, 1, 20, 165, 278, 26, 24, 1, 29, 228, 1, 15, 55, 54, 1, 2,
                       57, 52, 93, 72, 3, 7, 12, 6, 6, 1, 11, 105, 4, 2, 1, 1, 7, 1, 17, 1, 1, 2, 170, 91, 445, 6,
                       1, 2, 5, 1, 12, 4, 1, 1, 1, 14, 18, 7, 7, 5, 8, 4, 7, 4, 1, 3, 9, 390, 45, 156, 521, 15, 4,
                       5, 338, 254, 3, 5, 11, 15, 12, 43, 12, 12, 2, 2, 14, 1, 11, 37, 18, 134, 1, 7, 1, 29, 291,
                       1, 3, 4, 62, 4, 75, 1, 22, 228, 109, 233, 1, 366, 86, 50, 46, 68, 1, 1, 1, 1, 8, 14, 45, 2,
                       5, 45, 70, 89, 9, 99, 186, 50, 56, 54, 9, 120, 66, 56, 160, 269, 32, 65, 83, 67, 197, 43, 13,
                       26, 5, 46, 3, 6, 1, 60, 67, 56, 20, 2, 78, 11, 58, 1, 350, 1, 83, 41, 18, 2, 9, 1, 466, 224, 32])
    nonrare = np.argwhere(self.label_nums >= 10)  # non rare
    rare = np.argwhere(self.label_nums < 10)

    self.l_map = {0: 0, 1: 1, 2: 2, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13,
                  16: 7, 18: 14, 19: 15, 20: 15, 21: 16, 24: 17, 25: 18, 26: 19, 28: 20}

    # t = {"cut_instr": 2, "snowboard_instr": 21, "cut_obj": 4, "surf_instr": 0, "skateboard_instr": 26, "kick_obj": 7,
    #  "eat_obj": 9, "carry_obj": 14, "throw_obj": 15, "eat_instr": 16, "smile": 17, "look_obj": 18, "hit_instr": 19,
    #  "hit_obj": 20, "ski_instr": 1, "run": 22, "sit_instr": 10, "read_obj": 24, "ride_instr": 5, "walk": 3,
    #  "point_instr": 23, "jump_instr": 11, "work_on_computer_instr": 8, "hold_obj": 25, "drink_instr": 13,
    #  "lay_instr": 12, "talk_on_phone_instr": 6, "stand": 27, "catch_obj": 28}

    # [('surf_instr', 0), ('ski_instr', 1), ('cut_instr', 2), ('walk', 3), ('cut_obj', 4), ('ride_instr', 5),
    #  ('talk_on_phone_instr', 6), ('kick_obj', 7), ('work_on_computer_instr', 8), ('eat_obj', 9), ('sit_instr', 10),
    #  ('jump_instr', 11), ('lay_instr', 12), ('drink_instr', 13), ('carry_obj', 14), ('throw_obj', 15),
    #  ('eat_instr', 16), ('smile', 17), ('look_obj', 18), ('hit_instr', 19), ('hit_obj', 20), ('snowboard_instr', 21),
    #  ('run', 22), ('point_instr', 23), ('read_obj', 24), ('hold_obj', 25), ('skateboard_instr', 26), ('stand', 27),
    #  ('catch_obj', 28)]


    # map_24_to_2 = {}
    # 26
    self.COCO = COCO(coco_annot_file)
    self.VCOCO = _load_vcoco(vsrl_annot_file)
    self.image_ids = np.loadtxt(open(split_file, 'r'))
    # simple check  

    assert np.all(np.equal(np.sort(np.unique(self.VCOCO[0]['image_id'])), np.sort(self.image_ids)))

    self._init_coco()
    self._init_vcoco()
    self.vcocodb = self._get_vcocodb()

  def _init_vcoco(self):
    actions = [x['action_name'] for x in self.VCOCO]
    roles = [x['role_name'] for x in self.VCOCO]
    print(roles, actions)
    self.actions = actions
    self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
    self.num_actions = 222
    self.roles = roles


  def _init_coco(self):
    category_ids = self.COCO.getCatIds()
    categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
    self.category_to_id_map = dict(zip(categories, category_ids))
    self.classes = ['__background__'] + categories
    self.num_classes = len(self.classes)
    self.json_category_id_to_contiguous_id = {
        v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
    self.contiguous_category_id_to_json_id = {
        v: k for k, v in self.json_category_id_to_contiguous_id.items()}


  def _get_vcocodb(self):
    vcocodb = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))
    res_labels = []
    counts = 0
    s_labels = []
    for entry in vcocodb:
      self._prep_vcocodb_entry(entry)
      labels, seen_labels, count = self._add_gt_annotations(entry)
      res_labels.extend(labels)
      s_labels.extend(seen_labels)
      counts += count
    print('data statistics', counts, len(res_labels), len(set(res_labels)), sorted(list(set(res_labels))))
    print('data statistics', len(set(s_labels)), sorted(list(set(s_labels))))

    # data
    # statistics
    # 13780
    # 30
    # 18[
    #   (2, 37), (2, 50), (2, 53), (2, 64), (2, 74), (10, 62), (12, 2), (12, 3), (12, 17), (12, 19), (12, 55), (12, 64), (
    #   14, 9), (14, 10), (14, 59), (18, 20), (18, 60), (18, 61)]
    # data
    # statistics
    # 196[(0, 38), (1, 31), (2, 1), (2, 43), (2, 44), (2, 46), (2, 47), (2, 48), (2, 49), (2, 52), (2, 54), (2, 55), (
    # 2, 56), (2, 77), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 18), (3, 21), (4, 68), (5, 33), (
    #     6, 64), (7, 43), (7, 44), (7, 45), (7, 47), (7, 48), (7, 49), (7, 50), (7, 51), (7, 52), (7, 53), (7, 54), (
    #     7, 55), (7, 56), (8, 2), (8, 4), (8, 14), (8, 18), (8, 21), (8, 29), (8, 57), (8, 58), (8, 60), (8, 61), (
    #     8, 62), (9, 31), (9, 32), (9, 37), (9, 38), (10, 14), (10, 57), (10, 58), (10, 60), (11, 40), (11, 41), (
    #     11, 42), (11, 46), (12, 1), (12, 25), (12, 26), (12, 27), (12, 29), (12, 30), (12, 31), (12, 32), (12, 34), (
    #     12, 35), (12, 37), (12, 38), (12, 47), (12, 68), (12, 78), (13, 30), (13, 33), (14, 1), (14, 2), (14, 3), (
    #     14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (
    #     14, 20), (14, 21), (14, 25), (14, 26), (14, 27), (14, 28), (14, 29), (14, 30), (14, 31), (14, 32), (14, 33), (
    #     14, 34), (14, 35), (14, 36), (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 43), (14, 44), (
    #     14, 45), (14, 46), (14, 47), (14, 48), (14, 49), (14, 53), (14, 54), (14, 55), (14, 56), (14, 57), (14, 61), (
    #     14, 63), (14, 64), (14, 65), (14, 66), (14, 67), (14, 68), (14, 74), (14, 75), (14, 77), (15, 33), (15, 35), (
    #     15, 39), (16, 32), (17, 74), (18, 1), (18, 2), (18, 4), (18, 8), (18, 9), (18, 14), (18, 16), (18, 17), (
    #     18, 18), (18, 19), (18, 21), (18, 25), (18, 26), (18, 27), (18, 28), (18, 29), (18, 30), (18, 31), (18, 32), (
    #     18, 33), (18, 34), (18, 35), (18, 36), (18, 37), (18, 38), (18, 39), (18, 40), (18, 41), (18, 42), (18, 43), (
    #     18, 44), (18, 45), (18, 46), (18, 47), (18, 48), (18, 49), (18, 51), (18, 52), (18, 53), (18, 54), (18, 55), (
    #     18, 56), (18, 57), (18, 64), (18, 65), (18, 66), (18, 67), (18, 68), (18, 74), (18, 77), (18, 78), (18, 79), (
    #     18, 80), (19, 37), (20, 30), (20, 33)]

    # print
    if 0:
      nums = np.zeros((self.num_actions), dtype=np.int32)
      for entry in vcocodb:
        for aid in range(self.num_actions):
          nums[aid] += np.sum(np.logical_and(entry['gt_actions'][:, aid]==1, entry['gt_classes']==1))
      for aid in range(self.num_actions):
        print('Action %s = %d'%(self.actions[aid], nums[aid]))

    return vcocodb


  def _prep_vcocodb_entry(self, entry):
    entry['boxes'] = np.empty((0, 4), dtype=np.float32)
    entry['is_crowd'] = np.empty((0), dtype=np.bool)
    entry['gt_classes'] = np.empty((0), dtype=np.int32)
    entry['gt_actions'] = np.empty((0, self.num_actions), dtype=np.int32)
    entry['gt_role_id'] = np.empty((0, self.num_actions), dtype=np.int32)


  def _add_gt_annotations(self, entry):
    ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
    objs = self.COCO.loadAnns(ann_ids)
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    valid_ann_ids = []
    width = entry['width']
    height = entry['height']
    for i, obj in enumerate(objs):
      if 'ignore' in obj and obj['ignore'] == 1:
          continue
      # Convert form x1, y1, w, h to x1, y1, x2, y2
      x1 = obj['bbox'][0]
      y1 = obj['bbox'][1]
      x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
      y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
      x1, y1, x2, y2 = clip_xyxy_to_image(
          x1, y1, x2, y2, height, width)
      # Require non-zero seg area and more than 1x1 box size
      if obj['area'] > 0 and x2 > x1 and y2 > y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
        valid_ann_ids.append(ann_ids[i])
    num_valid_objs = len(valid_objs)
    assert num_valid_objs == len(valid_ann_ids)

    boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
    is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
    gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
    gt_actions = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_actions'].dtype)
    gt_role_id = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_role_id'].dtype)

    unseen_labels = []
    seen_labels = []
    counts = 0
    for ix, obj in enumerate(valid_objs):
      cls = self.json_category_id_to_contiguous_id[obj['category_id']]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      is_crowd[ix] = obj['iscrowd']
      
      tmp_action, tmp_role_id = \
        self._get_vsrl_data(valid_ann_ids[ix],
            valid_ann_ids, valid_objs, 26)
      # reconstruct 222 from 26

      label_map = json.load(open(cfg.LOCAL_DATA + "/Data/action_index.json"))

      role_id = -np.ones((self.num_actions), dtype=np.int32)
      gt_actions[ix, :] = np.zeros((self.num_actions), dtype=np.int32)
      for aid in np.argwhere(tmp_action == 1):  # loop 26 actions
        # import ipdb
        # ipdb.set_trace()
        for j, rid in enumerate(self.roles[aid[0]]):
          if rid == 'agent':
            continue
          else:
            # tmp_role_id[aid[0]]

            if np.all(tmp_role_id[aid[0]] == -1):
              continue
            for obj_idx in tmp_role_id[aid[0]]:
              if obj_idx == -1:
                continue
              else:
                action_name = self.actions[aid[0]] + '_' + rid
                if action_name not in label_map:
                  continue
                if label_map[action_name] not in self.l_map:
                  continue
                verb_id = self.l_map[label_map[action_name]]

                obj_cls = self.json_category_id_to_contiguous_id[valid_objs[obj_idx]['category_id']]
                if (verb_id, obj_cls) not in self.set_list:
                  unseen_labels.append((verb_id, obj_cls))
                  continue

                counts += 1
                action_id = self.set_list.index((verb_id, obj_cls))
                seen_labels.append((verb_id, obj_cls))
                gt_actions[ix, action_id] = 1
                role_id[action_id] = obj_idx

      gt_role_id[ix, :] = role_id

    entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
    entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
    entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
    entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
    entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)
    return unseen_labels, seen_labels, counts

  def _get_vsrl_data(self, ann_id, ann_ids, objs, num_actions):
    """ Get VSRL data for ann_id."""
    action_id = -np.ones((num_actions), dtype=np.int32)
    role_id = -np.ones((num_actions, 2), dtype=np.int32)
    # check if ann_id in vcoco annotations
    in_vcoco = np.where(self.VCOCO[0]['ann_id'] == ann_id)[0]
    if in_vcoco.size > 0:
      action_id[:] = 0
      role_id[:] = -1
    else:
      return action_id, role_id
    for i, x in enumerate(self.VCOCO):
      assert x['action_name'] == self.actions[i]
      has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
      if has_label.size > 0:
        action_id[i] = 1
        assert has_label.size == 1
        rids = x['role_object_id'][has_label]
        assert rids[0, 0] == ann_id
        for j in range(1, rids.shape[1]):
          if rids[0, j] == 0:
            # no role
            continue
          aid = np.where(ann_ids == rids[0, j])[0]
          assert aid.size > 0
          role_id[i, j - 1] = aid
    return action_id, role_id


  def _collect_detections_for_image(self, dets, image_id):

    agents = np.empty((0, 4 + self.num_actions), dtype=np.float32) # 4 + 26 = 30
    roles = np.empty((0, 5 * self.num_actions, 2), dtype=np.float32) # (5 * 26), 2
    for det in dets: # loop all detection instance
      # print(det.keys())
      if det['image_id'] == image_id:# might be several
        this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
        this_role  = np.zeros((1, 5 * self.num_actions, 2), dtype=np.float32)
        this_agent[0, :4] = det['person_box']
        for aid in range(self.num_actions): # loop 26 actions
          for j, rid in enumerate(self.roles[aid]):
            if rid == 'agent':
                #if aid == 10:
                #  this_agent[0, 4 + aid] = det['talk_' + rid]
                #if aid == 16:
                #  this_agent[0, 4 + aid] = det['work_' + rid]
                #if (aid != 10) and (aid != 16):

                this_agent[0, 4 + aid] = det[self.actions[aid] + '_' + rid]
            else:
                this_role[0, 5 * aid: 5 * aid + 5, j-1] = det[self.actions[aid] + '_' + rid]
        agents = np.concatenate((agents, this_agent), axis=0)
        roles  = np.concatenate((roles, this_role), axis=0)
    return agents, roles

  def _collect_detections_for_image1(self, dets, image_id):

    agents = np.empty((0, 4 + self.num_actions), dtype=np.float32) # 4 + 26 = 30
    roles = np.empty((0, 5 * self.num_actions), dtype=np.float32) # (5 * 26), 2
    for det in dets: # loop all detection instance
      # print(det.keys())
      if det['image_id'] == image_id:# might be several
        this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
        this_role  = np.zeros((1, 5 * self.num_actions), dtype=np.float32)
        this_agent[0, :4] = det['person_box']
        for aid in range(self.num_actions): # loop 222 actions
          # for j, rid in enumerate(self.roles[self.set_list[aid]]):
          #   if rid == 'agent':
          #       #if aid == 10:
          #       #  this_agent[0, 4 + aid] = det['talk_' + rid]
          #       #if aid == 16:
          #       #  this_agent[0, 4 + aid] = det['work_' + rid]
          #       #if (aid != 10) and (aid != 16):
          #
          #       this_agent[0, 4 + aid] = 0
          #   else:
          this_role[0, 5 * aid: 5 * aid + 5] = det[aid]
        agents = np.concatenate((agents, this_agent), axis=0)
        roles  = np.concatenate((roles, this_role), axis=0)
    return agents, roles


  def _do_eval(self, detections_file, ovr_thresh=0.5):

    # self._do_agent_eval(vcocodb, detections_file, ovr_thresh=ovr_thresh)
    self._do_role_eval(self.vcocodb, detections_file, ovr_thresh=ovr_thresh, eval_type='scenario_1')
    # self._do_role_eval(vcocodb, detections_file, ovr_thresh=ovr_thresh, eval_type='scenario_2')


  def _do_role_eval(self, vcocodb, detections_file, ovr_thresh=0.5, eval_type='scenario_1'):

    with open(detections_file, 'rb') as f:
      dets = pickle.load(f)
    
    tp = [[] for a in range(self.num_actions)]
    fp = [[] for a in range(self.num_actions)]
    sc = [[] for a in range(self.num_actions)]

    npos = np.zeros((self.num_actions), dtype=np.float32)
   
    
    for i in range(len(vcocodb)):
      image_id = vcocodb[i]['id']
      gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
      # person boxes
      gt_boxes = vcocodb[i]['boxes'][gt_inds]
      gt_actions = vcocodb[i]['gt_actions'][gt_inds]
      # some peorson instances don't have annotated actions
      # we ignore those instances
      ignore = np.any(gt_actions == -1, axis=1)
      assert np.all(gt_actions[np.where(ignore==True)[0]]==-1)

      for aid in range(self.num_actions):
        npos[aid] += np.sum(gt_actions[:, aid] == 1)

      pred_agents, pred_roles = self._collect_detections_for_image1(dets, image_id)

      for aid in range(self.num_actions):
        # if len(self.roles[aid])<2:
        #   if action has no role, then no role AP computed
          # continue

        for rid in range(1):

          # keep track of detected instances for each action for each role
          covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

          # get gt roles for action and role
          gt_role_inds = vcocodb[i]['gt_role_id'][gt_inds, aid]
          gt_roles = -np.ones_like(gt_boxes)
          for j in range(gt_boxes.shape[0]):
            if gt_role_inds[j] > -1:
              gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

          agent_boxes = pred_agents[:, :4]
          role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4]
          agent_scores = pred_roles[:, 5 * aid + 4]

          valid = np.where(np.isnan(agent_scores) == False)[0]
          #valid = np.where(agent_scores != 0)[0]


          agent_scores = agent_scores[valid]
          agent_boxes = agent_boxes[valid, :]
          role_boxes = role_boxes[valid, :]

          idx = agent_scores.argsort()[::-1]

          for j in idx:
            pred_box = agent_boxes[j, :]
            overlaps = get_overlap(gt_boxes, pred_box)

            # matching happens based on the person 
            jmax = overlaps.argmax()
            ovmax = overlaps.max()

            # if matched with an instance with no annotations
            # continue
            if ignore[jmax]:
              continue

            # overlap between predicted role and gt role
            if np.all(gt_roles[jmax, :] == -1): # if no gt role
              if eval_type == 'scenario_1':
                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                  # if no role is predicted, mark it as correct role overlap
                  ov_role = 1.0
                else:
                  # if a role is predicted, mark it as false 
                  ov_role = 0.0
              elif eval_type == 'scenario_2':
                # if no gt role, role prediction is always correct, irrespective of the actual predition
                ov_role = 1.0   
              else:
                raise ValueError('Unknown eval type')    
            else:
              ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

            is_true_action = (gt_actions[jmax, aid] == 1)
            sc[aid].append(agent_scores[j])
            # print(ovmax, ov_role, gt_roles[jmax])
            # import ipdb
            # ipdb.set_trace()
            if is_true_action and (ovmax>=ovr_thresh) and (ov_role>=ovr_thresh):
              if covered[jmax]:
                fp[aid].append(1)
                tp[aid].append(0)
              else:
                fp[aid].append(0)
                tp[aid].append(1)
                covered[jmax] = True
            else:
              fp[aid].append(1)
              tp[aid].append(0)

    # compute ap for each action
    role_ap = np.zeros((self.num_actions), dtype=np.float32)
    role_ap[:] = np.nan
    for aid in range(self.num_actions):
      # if len(self.roles[aid])<2:
      #   continue
      a_fp = np.array(fp[aid], dtype=np.float32)
      a_tp = np.array(tp[aid], dtype=np.float32)
      a_sc = np.array(sc[aid], dtype=np.float32)
      # sort in descending score order
      idx = a_sc.argsort()[::-1]
      a_fp = a_fp[idx]
      a_tp = a_tp[idx]
      a_sc = a_sc[idx]

      a_fp = np.cumsum(a_fp)
      a_tp = np.cumsum(a_tp)
      if npos[aid] == 0:
        rec = np.zeros(a_tp.shape, np.float32)
      else:
        rec = a_tp / float(npos[aid])
      #check
      assert(np.amax(rec) <= 1), rec
      prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
      role_ap[aid] = voc_ap(rec, prec)

    print('---------Reporting Role AP (%)------------------')
    for aid in range(self.num_actions):
      # if len(self.roles[aid])<2: continue
      # for rid in range(len(self.roles[aid])-1):
      print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(aid, role_ap[aid]*100.0, int(npos[aid])))
    nonrare = np.argwhere(self.label_nums >= 10) # non rare
    rare = np.argwhere(self.label_nums < 10)
    print('Average Role [%s] AP = %.2f'%(eval_type, np.nanmean(role_ap) * 100.00))
    print('Average Role [%s] nonrare = %.2f' % (eval_type, np.nanmean(role_ap[nonrare]) * 100.00))
    print('Average Role [%s] rare = %.2f' % (eval_type, np.nanmean(role_ap[rare]) * 100.00))
    print('---------------------------------------------') 
    # print('Average Role [%s] AP = %.2f, omitting the action "point"'%(eval_type, (np.nanmean(role_ap) * 25 - role_ap[-3][0]) / 24 * 100.00))
    print('---------------------------------------------')

    model_name = detections_file[len(cfg.LOCAL_DATA + 'Results/'):]
    iter_str = model_name.split('_')[0]
    iter_str = iter_str.replace('/', '')
    model_name = model_name[len(iter_str)+2:]
    model_name = model_name.replace('.pkl', '')
    f = open(cfg.LOCAL_DATA + '/coco_csv/{}_{}.csv'.format(model_name, eval_type), 'a')
    f.write('%.2f %.2f %.2f %.2f\n'%(np.nanmean(role_ap) * 100.00, np.nanmean(role_ap) * 100.00, np.nanmean(role_ap[rare]) * 100.00, np.nanmean(role_ap[nonrare]) * 100.00))
    f.flush()
    f.close()

  def _do_agent_eval(self, vcocodb, detections_file, ovr_thresh=0.5):

    with open(detections_file, 'rb') as f:
      dets = pickle.load(f)

    tp = [[] for a in range(self.num_actions)]
    fp = [[] for a in range(self.num_actions)]
    sc = [[] for a in range(self.num_actions)]

    npos = np.zeros((self.num_actions), dtype=np.float32)
    
    for i in range(len(vcocodb)):
      image_id = vcocodb[i]['id']# img ID, not the full name (e.g. id= 165, 'file_name' = COCO_train2014_000000000165.jpg )
      gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]# index of the person's box among all object boxes
      # person boxes
      gt_boxes = vcocodb[i]['boxes'][gt_inds] # all person's boxes in this image
      gt_actions = vcocodb[i]['gt_actions'][gt_inds] # index of Nx26 binary matrix indicating the actions
      # some peorson instances don't have annotated actions
      # we ignore those instances
      ignore = np.any(gt_actions == -1, axis=1)

      for aid in range(self.num_actions):
        npos[aid] += np.sum(gt_actions[:, aid] == 1)# how many actions are involved in this image(for all the human)

      pred_agents, _ = self._collect_detections_for_image(dets, image_id)
      # For each image, we have a pred_agents. For example, there are 2 people detected, then pred_agents is a 2x(4+26) matrix. Each row stands for a human, 0-3 human box, 4-25 the score for each action.
        
      for aid in range(self.num_actions):

        # keep track of detected instances for each action
        covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)# gt_boxes.shape[0] is the number of people in this image

        agent_scores = pred_agents[:, 4 + aid]# score of this action for all people in this image
        agent_boxes = pred_agents[:, :4] # predicted buman box for all people in this image
        # remove NaNs
        # If only use agent, there should be no NAN cause there is no object information provided. Just give a agent score.
        valid = np.where(np.isnan(agent_scores) == False)[0]
        agent_scores = agent_scores[valid] 
        agent_boxes = agent_boxes[valid, :]

        # sort in descending order
        idx = agent_scores.argsort()[::-1]# For this action, sort score of all people. A action cam be done by many people.
    
        for j in idx: # Each predicted person
          pred_box = agent_boxes[j, :]# It's predicted human box
          overlaps = get_overlap(gt_boxes, pred_box)# overlap between this predict human and all human gt_boxes

          jmax = overlaps.argmax()# Find the idx of gt human box that matches this predicted human
          ovmax = overlaps.max()

          # if matched with an instance with no annotations
          # continue
          if ignore[jmax]:
            continue
           
          is_true_action = (gt_actions[jmax, aid] == 1)# Is this person actually doing this action according to gt?

          sc[aid].append(agent_scores[j]) # The predicted score of this person doing this action. In descending order.
          if is_true_action and (ovmax>=ovr_thresh): # bounding box IOU is larger than 0.5 and this this person is doing this action.
            if covered[jmax]:
              fp[aid].append(1)
              tp[aid].append(0)
            else:# first time see this gt human
              fp[aid].append(0)
              tp[aid].append(1)
              covered[jmax] = True
          else:
            fp[aid].append(1)
            tp[aid].append(0)

    # compute ap for each action
    agent_ap = np.zeros((self.num_actions), dtype=np.float32)
    for aid in range(self.num_actions):

      a_fp = np.array(fp[aid], dtype=np.float32)
      a_tp = np.array(tp[aid], dtype=np.float32)
      a_sc = np.array(sc[aid], dtype=np.float32)
      # sort in descending score order
      idx = a_sc.argsort()[::-1]# For each action, sort the score of all predicted people in all images
      a_fp = a_fp[idx]
      a_tp = a_tp[idx]
      a_sc = a_sc[idx]

      a_fp = np.cumsum(a_fp)
      a_tp = np.cumsum(a_tp)
      rec = a_tp / float(npos[aid])
      #check
      
      assert(np.amax(rec) <= 1)
      prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
      agent_ap[aid] = voc_ap(rec, prec)

    print('---------Reporting Agent AP (%)------------------')
    for aid in range(self.num_actions):
      print('{: >20}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid], agent_ap[aid]*100.0, int(npos[aid])))
    print('Average Agent AP = %.2f'%(np.nansum(agent_ap) * 100.00/self.num_actions))
    print('---------------------------------------------')

def _load_vcoco(vcoco_file):
  print('loading vcoco annotations...')
  with open(vcoco_file, 'r') as f:
    vsrl_data = json.load(f)
  for i in range(len(vsrl_data)):
    vsrl_data[i]['role_object_id'] = \
    np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
    for j in ['ann_id', 'label', 'image_id']:
        vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
  return vsrl_data


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
  x1 = np.minimum(width - 1., np.maximum(0., x1))
  y1 = np.minimum(height - 1., np.maximum(0., y1))
  x2 = np.minimum(width - 1., np.maximum(0., x2))
  y2 = np.minimum(height - 1., np.maximum(0., y2))
  return x1, y1, x2, y2


def get_overlap(boxes, ref_box):
  ixmin = np.maximum(boxes[:, 0], ref_box[0])
  iymin = np.maximum(boxes[:, 1], ref_box[1])
  ixmax = np.minimum(boxes[:, 2], ref_box[2])
  iymax = np.minimum(boxes[:, 3], ref_box[3])
  iw = np.maximum(ixmax - ixmin + 1., 0.)
  ih = np.maximum(iymax - iymin + 1., 0.)
  inters = iw * ih

  # union
  uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
         (boxes[:, 2] - boxes[:, 0] + 1.) *
         (boxes[:, 3] - boxes[:, 1] + 1.) - inters)

  overlaps = inters / uni
  return overlaps


def voc_ap(rec, prec):
  """ ap = voc_ap(rec, prec)
  Compute VOC AP given precision and recall.
  [as defined in PASCAL VOC]
  """
  # correct AP calculation
  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

