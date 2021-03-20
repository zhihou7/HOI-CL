
image_set = 'test2015';
iter = 150000;


% h, o, p streams
exp_name = 'rcnn_caffenet_union';  model_name = exp_name;  cfg_name = 'rcnn_union';  gen_test_one;  %#ok

exp_name = 'rcnn_caffenet_ho';  model_name = exp_name;  cfg_name = 'rcnn_ho';  gen_test_one;  %#ok

exp_name = 'rcnn_caffenet_ho_pfc_vec0';  model_name = 'rcnn_caffenet_ho_pfc';  cfg_name = 'rcnn_ho_vec0';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pfc_box0';  model_name = 'rcnn_caffenet_ho_pfc';  cfg_name = 'rcnn_ho_box0';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pfc_vec1';  model_name = 'rcnn_caffenet_ho_pfc';  cfg_name = 'rcnn_ho_vec1';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pfc_box1';  model_name = 'rcnn_caffenet_ho_pfc';  cfg_name = 'rcnn_ho_box1';  gen_test_one;  %#ok

exp_name = 'rcnn_caffenet_ho_pfc_ip0';    model_name = 'rcnn_caffenet_ho_pfc';    cfg_name = 'rcnn_ho_ip0';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pfc_ip1';    model_name = 'rcnn_caffenet_ho_pfc';    cfg_name = 'rcnn_ho_ip1';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pconv_ip0';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip0';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pconv_ip1';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip1';  gen_test_one;  %#ok

% inidividual streams
clear score_blob
score_blob = 'h';  exp_name = 'rcnn_caffenet_ho_pconv_ip0_h';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip0';  gen_test_one;  %#ok
score_blob = 'o';  exp_name = 'rcnn_caffenet_ho_pconv_ip0_o';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip0';  gen_test_one;  %#ok
score_blob = 'p';  exp_name = 'rcnn_caffenet_ho_pconv_ip0_p';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip0';  gen_test_one;  %#ok

score_blob = 'h';  exp_name = 'rcnn_caffenet_ho_pconv_ip1_h';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip1';  gen_test_one;  %#ok
score_blob = 'o';  exp_name = 'rcnn_caffenet_ho_pconv_ip1_o';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip1';  gen_test_one;  %#ok
score_blob = 'p';  exp_name = 'rcnn_caffenet_ho_pconv_ip1_p';  model_name = 'rcnn_caffenet_ho_pconv';  cfg_name = 'rcnn_ho_ip1';  gen_test_one;  %#ok
clear score_blob

% using object detection scores
exp_name = 'rcnn_caffenet_ho_s';  model_name = exp_name;  cfg_name = 'rcnn_ho_s';  gen_test_one;  %#ok

exp_name = 'rcnn_caffenet_ho_pfc_ip0_s';    model_name = 'rcnn_caffenet_ho_pfc_s';    cfg_name = 'rcnn_ho_ip0_s';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pfc_ip1_s';    model_name = 'rcnn_caffenet_ho_pfc_s';    cfg_name = 'rcnn_ho_ip1_s';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pconv_ip0_s';  model_name = 'rcnn_caffenet_ho_pconv_s';  cfg_name = 'rcnn_ho_ip0_s';  gen_test_one;  %#ok
exp_name = 'rcnn_caffenet_ho_pconv_ip1_s';  model_name = 'rcnn_caffenet_ho_pconv_s';  cfg_name = 'rcnn_ho_ip1_s';  gen_test_one;


fprintf('done.\n');
