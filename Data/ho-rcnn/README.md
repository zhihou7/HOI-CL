# HO-RCNN

Code for reproducing the results in the following paper:

**Learning to Detect Human-Object Interactions**  
Yu-Wei Chao, Yunfan Liu, Xieyang Liu, Huayi Zeng, Jia Deng  
IEEE Winter Conference on Applications of Computer Vision (WACV), 2018  

Check out the [project site](http://www.umich.edu/~ywchao/hico/) for more details.

### Citing HO-RCNN

Please cite HO-RCNN if it helps your research:

    @INPROCEEDINGS{chao:wacv2018,
      author = {Yu-Wei Chao and Yunfan Liu and Xieyang Liu and Huayi Zeng and Jia Deng},
      booktitle = {Proceedings of the IEEE Winter Conference on Applications of Computer Vision},
      title = {Learning to Detect Human-Object Interactions},
      year = {2018},
    }

### Acknowledgements

Some of the instructions in this README are modified from the [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) repo created by [Ross Girshick](https://github.com/rbgirshick).

### Clone the Repository

This repo relies on two submodules (`fast-rcnn` and `caffe`), so make sure you clone with `--recursive`:

  ```Shell
  git clone --recursive https://github.com/ywchao/ho-rcnn.git
  ```

### Contents

1. [Evaluation](#evaluation)
2. [Running Detection with a Trained Model](#running-detection-with-a-trained-model)
3. [Training a Model](#training-a-model)
4. [From R-CNN to Fast R-CNN](#from-r-cnn-to-fast-r-cnn)
5. [Installation: Fast R-CNN](#installation-fast-r-cnn)
6. [Installation: MatCaffe](#installation-matcaffe)

## Evaluation

This demo runs the MATLAB evaluation script and replicates our results in the paper.

1. Download HICO-DET (7.5G):

    ```Shell
    ./scripts/fetch_hico_det.sh
    ```

    This will populate the `data` folder with `hico_20160224_det`.

2. Download pre-computed HOI detection on the HICO-DET test set (2.0G):

    ```Shell
    ./scripts/fetch_hoi_detection.sh
    ./scripts/setup_symlinks_detection.sh
    ```

    This will populate the `output` folder with `precomputed_hoi_detection` and set up a set of symlinks.

3. Evaluating on 600 classes is tedious, so we use `parfor` to speed up. Uncomment and set `poolsize` in `config.m` according to your need, or leave it commented out if you want MATLAB to set it automatically.

4. Start MATLAB `matlab` under `ho-rcnn`. You should see the message `added paths for the experiment!` followed by the MATLAB prompt `>>`.

5. Run `eval_run`. This will run the evaluation for the experiment **HO+IP1 (conv)+S**, under the Default and Known Object setting sequentially. The results will be printed and also saved under the folder `evaluation/result`.

6. You can run the evaluation for other experiments (which appear in the paper) by editing `eval_run.m` and rerunning it. For example, comment the line started with `exp_name = 'rcnn_caffenet_ho_pconv_ip1_s';` and uncomment the line started with `exp_name = 'rcnn_caffenet_ho_pconv_ip1';`.

## Running Detection with a Trained Model

This demo runs detection on the HICO-DET test set using a trained HO-RCNN model.

1. [Install Fast R-CNN](#installation-fast-r-cnn).

2. Obtain a trained model.

    **Option 1:** Download pre-computed HO-RCNN models (4.0G).

    ```Shell
    ./scripts/fetch_ho_rcnn_models.sh
    ./scripts/setup_symlinks_models.sh
    ```

    This will populate the `output` folder with `precomputed_ho_rcnn_models` and set up a set of symlinks.

    **Option 2:** [Train a Model](#training-a-model) yourself.

3. Download HICO-DET (7.5G) if you have not done so:

    ```Shell
    ./scripts/fetch_hico_det.sh
    ```

    This will populate the `data` folder with `hico_20160224_det`.

4. Download pre-computed Fast R-CNN object detection on the HICO-DET test set (37G):

    ```Shell
    ./scripts/fetch_fast_rcnn_detection_test.sh
    ```

    This will populate the `cache` folder with `det_base_caffenet/test2015`.

5. Run HOI detection separately for 80 object classes. Take the experiment **HO+IP1 (conv)+S** for example. Run the 80 scripts under `experiments/scripts/test_rcnn_caffenet_ho_pconv_ip1_s`, which corresponds to 80 object classes of interest. Each script will load the detected objects of one class, classify the associated HOI classes, and generate an output file (e.g. `detections_02.mat`) under `output/ho_1_s/hico_det_test2015/rcnn_caffenet_pconv_ip_iter_150000`.

    ```Shell
    ./experiments/scripts/test_rcnn_caffenet_ho_pconv_ip1_s/01_person.sh
    ./experiments/scripts/test_rcnn_caffenet_ho_pconv_ip1_s/02_bicycle.sh
    ...
    ./experiments/scripts/test_rcnn_caffenet_ho_pconv_ip1_s/80_toothbrush.sh
    ```

    **Warning:** Finishing all 80 scripts may take very long. We run these scripts parallelly on multiple GPUs. If you find the required computation infeasible, you might want to consider a faster approach in [From R-CNN to Fast R-CNN](#from-r-cnn-to-fast-r-cnn).

6. After verifying you have all 80 output files under `output/ho_1_s/hico_det_test2015/rcnn_caffenet_iter_150000`, you can [run the evaluation](#evaluation) following the previous section. Make sure to first remove the result files from evaluating pre-computed hoi detection, if any:

    ```Shell
    rm -r evaluation/result
    ```

## Training a Model

This demo trains a HO-RCNN model on the HICO-DET training set.

1. [Install Fast R-CNN](#installation-fast-r-cnn) if you have not done so.

2. Download HICO-DET (7.5G) if you have not done so:

    ```Shell
    ./scripts/fetch_hico_det.sh
    ```

    This will populate the `data` folder with `hico_20160224_det`.


3. Download pre-computed Fast R-CNN object detection on the HICO-DET train set (145G):

    ```Shell
    ./scripts/fetch_fast_rcnn_detection_train.sh
    ```

    This will populate the `cache` folder with `det_base_caffenet/train2015`.

4. Obtain the pre-trained ImageNet model.

    **Option 1:** Download the post-surgery pre-trained ImageNet model. **(recommended)**

    ```Shell
    ./scripts/fetch_post_surgery_imagenet_models.sh
    ```

    This will populate the `data` folder with `imagenet_models`.

    **Option 2:** Download the pre-trained ImageNet model and perform network surgery.

    We use MatCaffe to perform network surgery, so you need to [install MatCaffe](#installation-matcaffe) first before running the commands below.

    ```Shell
    cd fast-rcnn
    ./data/scripts/fetch_imagenet_models.sh
    cd ..
    matlab -r "net_surgery; quit"
    ```

5. Remove the symlinks from previous sections, if any:

    ```Shell
    find output -type l -delete
    ```

6. Start training. Take the experiment **HO+IP1 (conv)+S** for example:

    ```Shell
    ./experiments/scripts/rcnn_caffenet_ho_pconv_ip1_s.sh
    ```

    The trained models will be saved in `output/ho_1_s/hico_det_train2015` and the log file will be saved in `experiments/logs`.

7. Once the training is complete. You can [run detection](#running-detection-with-a-trained-model) and [run evaluation](#evaluation) following the previous sections.

## From R-CNN to Fast R-CNN

- Note that HO-RCNN performs detection in the R-CNN style, i.e. proposals from the same image are independently processed by the network. This is not very efficient especially in the test time. One way to speed things up is to perform detection in the Fast R-CNN style, i.e. applying the convolutional layers on the whole image followed by extracting proposal specific features with RoI pooling. This has already been implemented and included in this repo.

- We observed a 2-3x speedup on a full training run, and a 5-35x speedup on a full test run, where the amount of speedup is primarily determined by the network architecture. However, we also observed a slight decrease in mAP.

- You only need to run one script to run both **training** and **test**. First, complete step 1 to 5 in [Training a Model](#training-a-model) and step 1 to 4 in [Running Detection with a Trained Model](#running-detection-with-a-trained-model). After that, run the script under `experiments/scripts` with prefix `fast_rcnn_` instead of `rcnn_`. Take the experiment **HO+IP1 (conv)+S** for example:

    ```Shell
    ./experiments/scripts/fast_rcnn_caffenet_ho_pconv_ip1_s.sh
    ```

    The trained models will be saved in `output/ho_1_s/hico_det_train2015` but now with prefix `fast_rcnn_` instead of `rcnn_`. The log file will be saved in `experiments/logs` as before. The test output will now be just a single file `detections.mat` under `output/ho_1_s/hico_det_test2015/fast_rcnn_caffenet_pconv_ip_iter_150000`.

- To run **evaluation**, you only need to slightly edit `eval_run.m` and run it. Take the experiment **HO+IP1 (conv)+S** for example: comment the line started with `exp_name = 'rcnn_caffenet_ho_pconv_ip1_s';` and uncomment the line started with `exp_name = 'fast_rcnn_caffenet_ho_pconv_ip1_s';`.

## Installation: Fast R-CNN

The training and test of HO-RCNN is implemented in [our own branch of Fast R-CNN](https://github.com/ywchao/fast-rcnn/tree/ho-rcnn).

To install Fast R-CNN for HO-RCNN, change the directory:

  ```Shell
  cd fast-rcnn
  ```

and go through the below steps in this [README](https://github.com/ywchao/fast-rcnn/tree/ho-rcnn):

- Requirements: software
- Build the Cython modules
- Build Caffe and pycaffe

**Note:**

- We built Caffe with cuDNN v2 (cudnn-6.5-linux-x64-v2), which is the only cuDNN version the given branch supports.
- All our experiments are ran on the GeForce GTX TITAN X GPU.

## Installation: MatCaffe

MatCaffe is only needed if you want to run the network surgery yourself (see: [Training a Model](#training-a-model)).

To install MatCaffe:

  ```Shell
  cd caffe
  # Now follow the Caffe installation instructions here:
  #   http://caffe.berkeleyvision.org/installation.html

  # If you're experienced with Caffe and have all of the requirements installed
  # and your Makefile.config in place, then simply do:
  make -j8 && make matcaffe
  ```
