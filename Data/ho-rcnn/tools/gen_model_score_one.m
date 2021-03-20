
config;

postfix = '_s';

proto_dir     = './models/%s%s/';
proto_dir     = sprintf(proto_dir, model_name, postfix);
proto_file_tr = '%strain.prototxt';
proto_file_ts = '%stest.prototxt';
if use_pairwise
    proto_file_ts = '%stest_ip.prototxt';
end
proto_file_tr = sprintf(proto_file_tr, proto_dir);
proto_file_ts = sprintf(proto_file_ts, proto_dir);
makedir(proto_dir);

flag_indent = true;

anno = load(anno_file);

C = {};

% input
if numel(model_name) > 5 && strcmp(model_name(1:5),'rcnn_') == 1
    code = 'rcnn';
end
if numel(model_name) > 10 && strcmp(model_name(1:10),'fast_rcnn_') == 1
    code = 'fast_rcnn';
end
if use_pairwise
    src_file = sprintf('./experiments/templates/train.prototxt.01.input.%s.ho.p.so',code);
else
    src_file = sprintf('./experiments/templates/train.prototxt.01.input.%s.ho.so',code);
end
S = read_file_lines(src_file, flag_indent);
C = [C; S];

% vo classification
src_file = './experiments/templates/train.prototxt.02.vo.%s';
src_file = sprintf(src_file, model_name);
S = read_file_lines(src_file, flag_indent);
C = [C; S];

% o classification
src_file = './experiments/templates/train.prototxt.03.o.classify';
S = read_file_lines(src_file, flag_indent);
S = cellfun(@(x)strrep(x,'${NUM_OUTPUT}',num2str(numel(anno.list_action),'%d')), S, 'UniformOutput', false);
C = [C; S];

% combine classification
src_file = './experiments/templates/train.prototxt.04.output';
S = read_file_lines(src_file, flag_indent);
ind = cell_find_string(S,'${BOTTOM}');
C_BOTTOM = {'  bottom: "cls_score_vo"'};
C_BOTTOM = [C_BOTTOM; {'  bottom: "cls_score_so"'}];
S = [S(1:ind-1); C_BOTTOM; S(ind+1:end)];
C = [C; S];

% empty line at the end
C = [C; {''}];

% write to file
if ~exist(proto_file_tr, 'file')
    write_file_lines(proto_file_tr, C);
end


% find start line ind
for ind = 1:numel(C)
    if C{ind} == '}'
        break
    end 
end
rm_id = (1:ind)';

% find lines to be removed
for i = ind:numel(C)
    if strcmp(C{i},'  param {') == 1
        rm_id = [rm_id; (i:i+3)'];  %#ok
    end
    if strcmp(C{i},'    weight_filler {') == 1
        rm_id = [rm_id; (i:i+3)'];  %#ok
    end
    if strcmp(C{i},'    bias_filler {') == 1
        rm_id = [rm_id; (i:i+3)'];  %#ok
    end
    if numel(C{i}) >= 14 && strcmp(C{i}(1:14), '  name: "drop6') == 1
        rm_id = [rm_id; (i-1:i+7)'];  %#ok
    end
    if numel(C{i}) >= 14 && strcmp(C{i}(1:14), '  name: "drop7') == 1
        rm_id = [rm_id; (i-1:i+7)'];  %#ok
    end
    if numel(C{i}) >= 18 && strcmp(C{i}(1:18), '  name: "loss_cls"') == 1
        rm_id = [rm_id; (i-1:i+6)'];  %#ok
    end
end

% remove lines and add starting and ending block
C(rm_id) = [];
C_start = {'name: "CaffeNet"'};
if numel(model_name) > 5 && strcmp(model_name(1:5),'rcnn_') == 1
    C_start = [C_start; ...
        {'input: "data_h"'}; ...
        {'input_shape {'}; ...
        {'  dim: 1'}; ...
        {'  dim: 3'}; ...
        {'  dim: 227'}; ...
        {'  dim: 227'}; ...
        {'}'}];
    C_start = [C_start; ...
        {'input: "data_o"'}; ...
        {'input_shape {'}; ...
        {'  dim: 1'}; ...
        {'  dim: 3'}; ...
        {'  dim: 227'}; ...
        {'  dim: 227'}; ...
        {'}'}];
end
if numel(model_name) > 10 && strcmp(model_name(1:10),'fast_rcnn_') == 1
    C_start = [C_start; ...
        {'input: "data"'}; ...
        {'input_shape {'}; ...
        {'  dim: 1'}; ...
        {'  dim: 3'}; ...
        {'  dim: 227'}; ...
        {'  dim: 227'}; ...
        {'}'}];
    C_start = [C_start; ...
        {'input: "rois_h"'}; ...
        {'input_shape {'}; ...
        {'  dim: 1 # to be changed on-the-fly to num ROIs'}; ...
        {'  dim: 5 # [batch ind, x1, y1, x2, y2] zero-based indexing'}; ...
        {'}'}];
    C_start = [C_start; ...
        {'input: "rois_o"'}; ...
        {'input_shape {'}; ...
        {'  dim: 1 # to be changed on-the-fly to num ROIs'}; ...
        {'  dim: 5 # [batch ind, x1, y1, x2, y2] zero-based indexing'}; ...
        {'}'}];
end
if use_pairwise
    C_start = [C_start; ...
        {'input: "data_p"'}; ...
        {'input_shape {'}; ...
        {'  dim: 1'}; ...
        {'  dim: 2'}; ...
        {'  dim: 64'}; ...
        {'  dim: 64'}; ...
        {'}'}];
end
C_start = [C_start; ...
    {'input: "score_o"'}; ...
    {'input_shape {'}; ...
    {'  dim: 1'}; ...
    {'  dim: 1'}; ...
    {'}'}];

C_end = [{'layer {'}; ...
    {'  name: "cls_prob"'}; ...
    {'  type: "Sigmoid"'}; ...
    {'  bottom: "cls_score"'}; ...
    {'  top: "cls_prob"'}; ...
    {'}'}];

% empty line at the end
C = [C_start; C(1:end-1); C_end; C(end)];

% write to file
if ~exist(proto_file_ts, 'file')
    write_file_lines(proto_file_ts, C);
end
