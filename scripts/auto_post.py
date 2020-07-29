#!/usr/bin/env python
# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou
# ---------
import sys
if len(sys.argv) > 1:
    command = sys.argv[1]
else:
    command = "python ../../scripts/postprocess_test1.py --model '.*rew.*' --fuse_type spv"

arrs = command.split('--')
filter_item = 'fuse_type\ spv'
for item in arrs:
    if item.startswith('fuse'):
        filter_item = item.strip()
import os
print(filter_item)
print(command)
while True:
    print('start new')
    res = os.system(command)
    os.system("ps -ux | grep "+filter_item+" | awk '{print $2}' | xargs kill")
    # res = os.system('python ../../scripts/postprocess_test.py --model .*rew.* --fuse_type sp')
    # res = os.system('python ../../scripts/postprocess_test.py --model .*rew.* --fuse_type v')
    # import time
    # time.sleep(60)