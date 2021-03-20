
model_name = 'rcnn_caffenet_ho';  use_pairwise = false;  gen_model_score_one;  %#ok

model_name = 'rcnn_caffenet_ho_pfc';    use_pairwise = true;  gen_model_score_one;  %#ok
model_name = 'rcnn_caffenet_ho_pconv';  use_pairwise = true;  gen_model_score_one;  %#ok


model_name = 'fast_rcnn_caffenet_ho';  use_pairwise = false;  gen_model_score_one;  %#ok

model_name = 'fast_rcnn_caffenet_ho_pfc';    use_pairwise = true;  gen_model_score_one;  %#ok
model_name = 'fast_rcnn_caffenet_ho_pconv';  use_pairwise = true;  gen_model_score_one;


fprintf('done.\n');
