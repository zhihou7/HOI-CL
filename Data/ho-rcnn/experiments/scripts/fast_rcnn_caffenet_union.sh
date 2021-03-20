#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/fast_rcnn_caffenet_union.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./fast-rcnn/tools/train_net.py --gpu $gpu_id \
  --solver ./models/fast_rcnn_caffenet_union/solver.prototxt \
  --weights ./fast-rcnn/data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb hico_det_train2015 \
  --cfg ./experiments/cfgs/fast_rcnn_union.yml \
  --iters 150000

time ./fast-rcnn/tools/test_net.py --gpu $gpu_id \
  --def ./models/fast_rcnn_caffenet_union/test.prototxt \
  --net ./output/union/hico_det_train2015/fast_rcnn_caffenet_iter_150000.caffemodel \
  --imdb hico_det_test2015 \
  --cfg ./experiments/cfgs/fast_rcnn_union.yml \
  --nbatch 1
