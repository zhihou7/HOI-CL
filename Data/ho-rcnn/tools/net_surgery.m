% change layer names (from bvlc_reference_caffenet) for ho experiments

config;

addpath([caffe_root 'matlab/']);

gpu_id = 0;

% set deploy and weight file
%   The pretrained model from fast-rcnn is not the same one as the one
%   provided by bvlc.
deploy_file = [caffe_root 'models/bvlc_reference_caffenet/deploy.prototxt'];
weight_file = [frcnn_root 'data/imagenet_models/CaffeNet.v2.caffemodel'];

% initialize vars for original/finetuned network
layer_names    = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
layer_names_h  = cellfun(@(x)[x '_h'], layer_names, 'UniformOutput', false);
layer_names_o  = cellfun(@(x)[x '_o'], layer_names, 'UniformOutput', false);
weights        = cell(numel(layer_names), 1);
weights_h      = cell(numel(layer_names_h), 1);
weights_o      = cell(numel(layer_names_o), 1);

% load anno
anno = load(anno_file);

% get object list
det_file = './cache/det_base_caffenet/train2015/HICO_train2015_00000001.mat';
assert(exist(det_file,'file') ~= 0);
ld = load(det_file);
list_coco_obj = cellfun(@(x)strrep(x,' ','_'),ld.cls,'UniformOutput',false);
list_coco_obj = list_coco_obj(2:end)';

% set mode and device
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

% load original network
net = caffe.Net(deploy_file, weight_file, 'test');
fprintf('loading original network ... \n');
for i = 1:numel(layer_names)
    fprintf('  %s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
        layer_names{i}, ...
        sprintf('%d ', size(net.layers(layer_names{i}).params(1).get_data())), ...
        sprintf('%d ', size(net.layers(layer_names{i}).params(2).get_data())) ...
        );
    weights{i}{1} = net.layers(layer_names{i}).params(1).get_data();
    weights{i}{2} = net.layers(layer_names{i}).params(2).get_data();
end

% reset caffe
caffe.reset_all();

% get deploy and weight file for ho network
deploy_ho_file = './models/net_surgery_caffenet_ho.prototxt';
weight_ho_file = './data/imagenet_models/CaffeNet.v2.ho.caffemodel';

fprintf('\n');
fprintf('generating ho network ... \n');
fprintf('deploy file: %s\n', deploy_ho_file);
fprintf('weight file: %s\n', weight_ho_file);

% set mode and device
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
    
% load new network
net = caffe.Net(deploy_ho_file, weight_file, 'test');
for i = 1:numel(layer_names_o)
    fprintf('  %s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
        layer_names_o{i}, ...
        sprintf('%d ', size(net.layers(layer_names_o{i}).params(1).get_data())), ...
        sprintf('%d ', size(net.layers(layer_names_o{i}).params(2).get_data())) ...
        );
    weights_o{i}{1} = net.layers(layer_names_o{i}).params(1).get_data();
    weights_o{i}{2} = net.layers(layer_names_o{i}).params(2).get_data();
    assert(all(weights_o{i}{1}(:) == 0) == 1);
    assert(all(weights_o{i}{2}(:) == 0) == 1);
end
for i = 1:numel(layer_names_h)
    fprintf('  %s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
        layer_names_h{i}, ...
        sprintf('%d ', size(net.layers(layer_names_h{i}).params(1).get_data())), ...
        sprintf('%d ', size(net.layers(layer_names_h{i}).params(2).get_data())) ...
        );
    weights_h{i}{1} = net.layers(layer_names_h{i}).params(1).get_data();
    weights_h{i}{2} = net.layers(layer_names_h{i}).params(2).get_data();
    assert(all(weights_h{i}{1}(:) == 0) == 1);
    assert(all(weights_h{i}{2}(:) == 0) == 1);
end
fprintf('  %s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
    'cls_score_so', ...
    sprintf('%d ', size(net.layers('cls_score_so').params(1).get_data())), ...
    sprintf('%d ', size(net.layers('cls_score_so').params(2).get_data())) ...
    );
weights_so{1} = net.layers('cls_score_so').params(1).get_data();
weights_so{2} = net.layers('cls_score_so').params(2).get_data();
assert(all(weights_so{1}(:) == 0) == 1);
assert(all(weights_so{2}(:) == 0) == 1);

% transplant parameters
for i = 1:numel(layer_names_o)
    weights_o{i}{1} = reshape(weights{i}{1}, size(weights_o{i}{1}));
    weights_o{i}{2} = reshape(weights{i}{2}, size(weights_o{i}{2}));
    net.layers(layer_names_o{i}).params(1).set_data(weights_o{i}{1});
    net.layers(layer_names_o{i}).params(2).set_data(weights_o{i}{2});
end
for i = 1:numel(layer_names_h)
    weights_h{i}{1} = reshape(weights{i}{1}, size(weights_h{i}{1}));
    weights_h{i}{2} = reshape(weights{i}{2}, size(weights_h{i}{2}));
    net.layers(layer_names_h{i}).params(1).set_data(weights_h{i}{1});
    net.layers(layer_names_h{i}).params(2).set_data(weights_h{i}{2});
end

% manually set parameters
for i = 1:numel(anno.list_action)
    obj_id = cell_find_string(list_coco_obj, anno.list_action(i).nname);
    weights_so{1}(obj_id, i) = 1;
end
weights_so{1} = single(weights_so{1});
weights_so{2} = single(weights_so{2});
net.layers('cls_score_so').params(1).set_data(weights_so{1});
net.layers('cls_score_so').params(2).set_data(weights_so{2});

% save weights
if ~exist(weight_ho_file,'file')
    path = fileparts(weight_ho_file);
    makedir(path);
    net.save(weight_ho_file);
end

% reset caffe
caffe.reset_all();

% assertions (optional)
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
    
net_1 = caffe.Net(deploy_file, weight_file, 'test');
net_2 = caffe.Net(deploy_ho_file, weight_ho_file, 'test');
fprintf('\n');
fprintf('checking ... \n');
for i = 1:numel(layer_names)
    w1 = net_1.layers(layer_names{i}).params(1).get_data;
    wo = net_2.layers(layer_names_o{i}).params(1).get_data;
    wh = net_2.layers(layer_names_h{i}).params(1).get_data;
    b1 = net_1.layers(layer_names{i}).params(2).get_data;
    bo = net_2.layers(layer_names_o{i}).params(2).get_data;
    bh = net_2.layers(layer_names_h{i}).params(2).get_data;
    fprintf('%-8s %-8s %-8s  %.10f %.10f %.10f %.10f\n', ...
        layer_names{i}, layer_names_o{i}, layer_names_h{i}, ...
        sum(abs(w1(:)-wo(:))), sum(abs(b1(:)-bo(:))), ...
        sum(abs(w1(:)-wh(:))), sum(abs(b1(:)-bh(:))));
end

% reset caffe
caffe.reset_all();
