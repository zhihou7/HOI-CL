#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

mkdir -p experiments/logs/test_rcnn_caffenet_ho_pfc_ip1_s/

LOG="experiments/logs/test_rcnn_caffenet_ho_pfc_ip1_s/07_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./fast-rcnn/tools/test_net.py --gpu $gpu_id \
  --def ./models/rcnn_caffenet_ho_pfc_s/test_ip.prototxt \
  --net ./output/ho_1_s/hico_det_train2015/rcnn_caffenet_pfc_ip_iter_150000.caffemodel \
  --imdb hico_det_test2015 \
  --cfg ./experiments/cfgs/rcnn_ho_ip1_s.yml \
  --oid 7
