#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/fast_rcnn_caffenet_ho_pconv_ip0_s.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./fast-rcnn/tools/train_net.py --gpu $gpu_id \
  --solver ./models/fast_rcnn_caffenet_ho_pconv_s/solver_ip.prototxt \
  --weights ./data/imagenet_models/CaffeNet.v2.ho.caffemodel \
  --imdb hico_det_train2015 \
  --cfg ./experiments/cfgs/fast_rcnn_ho_ip0_s.yml \
  --iters 150000

time ./fast-rcnn/tools/test_net.py --gpu $gpu_id \
  --def ./models/fast_rcnn_caffenet_ho_pconv_s/test_ip.prototxt \
  --net ./output/ho_0_s/hico_det_train2015/fast_rcnn_caffenet_pconv_ip_iter_150000.caffemodel \
  --imdb hico_det_test2015 \
  --cfg ./experiments/cfgs/fast_rcnn_ho_ip0_s.yml \
  --nbatch 4
