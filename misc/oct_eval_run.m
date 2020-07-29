
image_set = 'test2015';
iter = 500000;


% --------------------------------------------------------------------
% RCNN
% --------------------------------------------------------------------

exp_name = 'rcnn_caffenet_ho_pconv_ip1_s_VCL_test_v_iter_500000';
exp_dir = 'ho_1_s';  prefix = 'rcnn_caffenet_pconv_ip_VCL_test_v';  format = 'obj';  score_blob = 'n/a';

% --------------------------------------------------------------------
% Fast-RCNN
% --------------------------------------------------------------------


eval_mode = 'def';  oct_eval_one;  %#ok
%eval_mode = 'ko';   eval_one1;
%exp_name = 'rcnn_caffenet_ho_pconv_ip1_s_R1';  exp_dir = 'ho_1_s';  prefix = 'rcnn_caffenet_pconv_ip_R1';  format = 'obj';  score_blob = 'n/a';
