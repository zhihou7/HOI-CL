
% get object list
det_file = './cache/det_base_caffenet/train2015/HICO_train2015_00000001.mat';
assert(exist(det_file,'file') ~= 0);
ld = load(det_file);
list_coco_obj = cellfun(@(x)strrep(x,' ','_'),ld.cls,'UniformOutput',false);
list_coco_obj = list_coco_obj(2:end)';

% template
script_temp = './experiments/templates/run_test_net.sh';

% script
script_name = '%02d_%s.sh';
script_base = './experiments/scripts/test_%s/';
script_base = sprintf(script_base, exp_name);
makedir(script_base);

% test_name
test_name = 'test';
if numel(exp_name) > 4 && numel(strfind(exp_name,'_ip')) == 1
    test_name = 'test_ip';
end
if numel(exp_name) > 4 && numel(strfind(exp_name,'_vec')) == 1
    test_name = 'test_vec';
end
if numel(exp_name) > 4 && numel(strfind(exp_name,'_box')) == 1
    test_name = 'test_box';
end

% exp_dir
cfg_file = ['./experiments/cfgs/' cfg_name '.yml'];
C = read_file_lines(cfg_file);
ind_1 = find(cellfun(@(x)(numel(x) > 9) && strcmp(x(1:9),'EXP_DIR: ') == 1, C));
assert(numel(ind_1) == 1);
ind_2 = strfind(C{ind_1},'"');
assert(numel(ind_2) == 2);
exp_dir = C{ind_1}(ind_2(1)+1:ind_2(2)-1);

% snapshot_prefix
solver_file = ['./models/' model_name '/solver.prototxt'];
if numel(exp_name) > 4 && numel(strfind(exp_name,'_ip')) == 1
    solver_file = ['./models/' model_name '/solver_ip.prototxt'];
end
if numel(exp_name) > 4 && numel(strfind(exp_name,'_vec')) == 1
    solver_file = ['./models/' model_name '/solver_vec.prototxt'];
end
if numel(exp_name) > 4 && numel(strfind(exp_name,'_box')) == 1
    solver_file = ['./models/' model_name '/solver_box.prototxt'];
end
C = read_file_lines(solver_file);
ind_1 = find(cellfun(@(x)(numel(x) > 17) && strcmp(x(1:17),'snapshot_prefix: ') == 1, C));
assert(numel(ind_1) == 1);
ind_2 = strfind(C{ind_1},'"');
assert(numel(ind_2) == 2);
snapshot_prefix = C{ind_1}(ind_2(1)+1:ind_2(2)-1);

% script file
for o = 1:numel(list_coco_obj)
    obj_id = o;
    obj_name = list_coco_obj{o};
    
    clear pre_str new_str
    pre_str{1} = '${exp_name}';
    pre_str{2} = '${iter}';
    pre_str{3} = '${image_set}';
    pre_str{4} = '${obj_id}';
    pre_str{5} = '${obj_id_pad}';
    pre_str{6} = '${obj_name}';
    pre_str{7} = '${model_name}';
    pre_str{8} = '${test_name}';
    pre_str{9} = '${cfg_name}';
    pre_str{10} = '${exp_dir}';
    pre_str{11} = '${snapshot_prefix}';
    new_str{1} = exp_name;
    new_str{2} = num2str(iter);
    new_str{3} = image_set;
    new_str{4} = num2str(obj_id);
    new_str{5} = num2str(obj_id,'%02d');
    new_str{6} = obj_name;
    new_str{7} = model_name;
    new_str{8} = test_name;
    new_str{9} = cfg_name;
    new_str{10} = exp_dir;
    new_str{11} = snapshot_prefix;
    
    flag_indent = true;
    script_file = [script_base sprintf(script_name, obj_id, obj_name)];
    src_file = script_temp;
    trg_file = script_file;
    if ~exist(trg_file,'file')
        C = read_file_lines(src_file,flag_indent);
        % set score blob if specified
        if exist('score_blob','var')
            C = cellfun(@(x)strrep(x,'_${image_set}/',['_${image_set}_' score_blob '/']),C,'UniformOutput',false);
            C{end} = [C{end} ' \'];
            C = [C; sprintf('  --set TEST.SCORE_BLOB %s',score_blob)];  %#ok
        end
        for i = 1:numel(pre_str)
            C = cellfun(@(x)strrep(x,pre_str{i},new_str{i}),C,'UniformOutput',false);
        end
        C = [C; {''}];  %#ok
        write_file_lines(trg_file,C);
        edit_file_permission(trg_file,'755');
    end
end
