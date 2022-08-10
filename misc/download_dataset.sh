#!/bin/bash

# Download COCO dataset
echo "Downloading V-COCO"

# Download V-COCO
mkdir Data
git clone --recursive https://github.com/s-gupta/v-coco.git Data/v-coco/
cd Data/v-coco/coco

URL_2014_Train_images=http://images.cocodataset.org/zips/train2014.zip
URL_2014_Val_images=http://images.cocodataset.org/zips/val2014.zip
URL_2014_Test_images=http://images.cocodataset.org/zips/test2014.zip
URL_2014_Trainval_annotation=http://images.cocodataset.org/annotations/annotations_trainval2014.zip

wget -N $URL_2014_Train_images
wget -N $URL_2014_Val_images
wget -N $URL_2014_Test_images
wget -N $URL_2014_Trainval_annotation

mkdir images

unzip train2014.zip -d images/
unzip val2014.zip -d images/
unzip test2014.zip -d images/
unzip annotations_trainval2014.zip


rm train2014.zip
rm val2014.zip
rm test2014.zip
rm annotations_trainval2014

# Pick out annotations from the COCO annotations to allow faster loading in V-COCO
echo "Picking out annotations from the COCO annotations to allow faster loading in V-COCO"

cd ../
python script_pick_annotations.py coco/annotations

# Build
echo "Building"
cd coco/PythonAPI/ && make install
cd ../../ && make
cd ../../

# Download HICO-DET dataset
echo "Downloading HICO-DET"

URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz

pip install gdown
#python lib/ult/Download_data.py 1hIElxTyJ0HrTww_p1GpHD9KLZNw8OVJH Data/hico_20160224_det.tar.gz
gdown 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk -O Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

#https://drive.google.com/file/d/1hIElxTyJ0HrTww_p1GpHD9KLZNw8OVJH/view?usp=sharing
# Download HICO-DET evaluation code
cd Data/
git clone https://github.com/ywchao/ho-rcnn.git
cd ../
cp misc/Generate_detection.m Data/ho-rcnn/
cp misc/OctGenerate_detection.m Data/ho-rcnn/
cp misc/oct_eval_one.m Data/ho-rcnn/evaluation
cp misc/oct_eval_run.m Data/ho-rcnn/evaluation
cp misc/save_mat.m Data/ho-rcnn/
cp misc/load_mat.m Data/ho-rcnn/
cp misc/num_inst.npy Data/
cp misc/hoi_to_vb.pkl Data/
cp misc/hoi_to_obj.pkl Data/
cp misc/hico_list_vb.txt Data/
cp misc/hico_list_hoi.txt Data/
cp misc/hico_list_obj.txt Data/

mkdir Data/ho-rcnn/data/hico_20160224_det/
# python lib/ult/Download_data.py 1cE10X9rRzzqeSPi-BKgIcDgcPXzlEoXX Data/ho-rcnn/data/hico_20160224_det/anno_bbox.mat
# python lib/ult/Download_data.py 1ds_qW9wv-J3ESHj_r_5tFSOZozGGHu1r Data/ho-rcnn/data/hico_20160224_det/anno.mat
gdown 1cE10X9rRzzqeSPi-BKgIcDgcPXzlEoXX -O Data/ho-rcnn/data/hico_20160224_det/anno_bbox.mat
gdown 1ds_qW9wv-J3ESHj_r_5tFSOZozGGHu1r -O Data/ho-rcnn/data/hico_20160224_det/anno.mat

# Download COCO Pre-trained weights
echo "Downloading COCO Pre-trained weights..."

mkdir Weights/

# python lib/ult/Download_data.py 1IbR4kiWgLF8seaKjOMmwaHs0Bfwl5Dq1 Weights/res50_faster_rcnn_iter_1190000.ckpt.data-00000-of-00001
# python lib/ult/Download_data.py 1-DbfEloN4c2JaCEMnexaWAsSc4MDlZJx Weights/res50_faster_rcnn_iter_1190000.ckpt.index
# python lib/ult/Download_data.py 1vc5d3OwCtMtRgXq3Pj4_twpK4x3kjgT0 Weights/res50_faster_rcnn_iter_1190000.ckpt.meta
gdown 1IbR4kiWgLF8seaKjOMmwaHs0Bfwl5Dq1 -O Weights/res50_faster_rcnn_iter_1190000.ckpt.data-00000-of-00001
gdown 1-DbfEloN4c2JaCEMnexaWAsSc4MDlZJx -O Weights/res50_faster_rcnn_iter_1190000.ckpt.index
gdown 1vc5d3OwCtMtRgXq3Pj4_twpK4x3kjgT0 -O Weights/res50_faster_rcnn_iter_1190000.ckpt.meta
# 
mkdir Results/
# python lib/ult/Download_data.py 0B1_fAEgxdnvJR1N3c1FYRGo1S1U Weights/coco_900-1190k.tgz
gdown 0B1_fAEgxdnvJR1N3c1FYRGo1S1U -O Weights/coco_900-1190k.tgz
cd Weights
tar -xvf coco_900-1190k.tgz
mv coco_2014_train+coco_2014_valminusminival/res101* ./
cd ../
#https://drive.google.com/file/d/0B1_fAEgxdnvJR1N3c1FYRGo1S1U/view?usp=sharing
# https://drive.google.com/drive/folders/0B1_fAEgxdnvJeGg0LWJZZ1N2aDA down load res101




