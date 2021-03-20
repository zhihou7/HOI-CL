#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

mkdir -p experiments/logs/test_${exp_name}/

LOG="experiments/logs/test_${exp_name}/${obj_id_pad}_${obj_name}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./fast-rcnn/tools/test_net.py --gpu $gpu_id \
  --def ./models/${model_name}/${test_name}.prototxt \
  --net ./output/${exp_dir}/hico_det_train2015/${snapshot_prefix}_iter_${iter}.caffemodel \
  --imdb hico_det_${image_set} \
  --cfg ./experiments/cfgs/${cfg_name}.yml \
  --oid ${obj_id}