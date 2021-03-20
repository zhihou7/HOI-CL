function eval_one1(exp_name, prefix, result_file, iter)

image_set = 'test2015';
score_blob = 'n/a';
format = 'obj';
exp_dir = 'ho_1_s';
eval_mode = 'def';
config;

min_overlap = 0.5;

% assertions
assert(ismember(score_blob,{'n/a','h','o','p'}) == 1);

% set detection root
det_root = './output/%s/hico_det_%s/%s_iter_%s/';
det_root = sprintf(det_root, exp_dir, image_set, prefix, iter);
disp(prefix)
disp(iter)
disp(det_root)
if ismember(score_blob, {'h','o','p'})
    det_root = [det_root(1:end-1) '_' score_blob '/'];
end

disp(det_root)
% set res file
res_root = './evaluation/result/%s/';
res_root = sprintf(res_root, exp_name);
res_file = '%s%s_%s_%s.mat';
res_file = sprintf(res_file, res_root, eval_mode, image_set, iter);
if ismember(score_blob, {'h','o','p'})
    res_file = [res_file(1:end-4) '_' score_blob '.mat'];
end
makedir(res_root);

% load annotations
anno = load(anno_file);
bbox = load(bbox_file);

% get gt bbox
switch image_set
    case 'train2015'
        gt_bbox = bbox.bbox_train;
        list_im = anno.list_train;
        anno_im = anno.anno_train;
    case 'test2015'
        gt_bbox = bbox.bbox_test;
        list_im = anno.list_test;
        anno_im = anno.anno_test;
    otherwise
        error('image_set error\n');
end
assert(numel(gt_bbox) == numel(list_im));

% copy variables
list_action = anno.list_action;
num_action = numel(list_action);
num_image = numel(gt_bbox);

% get object list
det_file = './cache/det_base_caffenet/train2015/HICO_train2015_00000001.mat';
if exist(det_file,'file') ~= 0
    ld = load(det_file);
    list_coco_obj = cellfun(@(x)strrep(x,' ','_'),ld.cls,'UniformOutput',false);
    list_coco_obj = list_coco_obj(2:end)';
else
    list_coco_obj = get_list_coco_obj();
end

% get HOI index intervals for object classes
obj_hoi_int = zeros(numel(list_coco_obj), 2);
for i = 1:numel(list_coco_obj)
    hoi_int = cell_find_string({list_action.nname}', list_coco_obj{i});
    assert(~isempty(hoi_int));
    obj_hoi_int(i, 1) = hoi_int(1);
    obj_hoi_int(i, 2) = hoi_int(end);
end

fprintf('start evaluation\n');
fprintf('setting:     %s\n', eval_mode);
fprintf('exp_name:    %s\n', exp_name);
fprintf('score_blob:  %s\n', score_blob)
fprintf('\n')

if exist(res_file, 'file')
    % load result file
    fprintf('results loaded from %s\n', res_file);
    ld = load(res_file);
    AP = ld.AP;
    REC = ld.REC;
    % print ap for each class
    for i = 1:num_action
        nname = list_action(i).nname;
        aname = [list_action(i).vname_ing '_' list_action(i).nname];
        fprintf('  %03d/%03d %-30s', i, num_action, aname);
        fprintf('  ap: %.4f  rec: %.4f\n', AP(i), REC(i));
    end
else
    % convert gt format
    gt_all = cell(num_action, num_image);
    fprintf('converting gt bbox format ... \n')
    for i = 1:num_image
        assert(strcmp(gt_bbox(i).filename, list_im{i}) == 1)
        for j = 1:numel(gt_bbox(i).hoi)
            if ~gt_bbox(i).hoi(j).invis
                hoi_id = gt_bbox(i).hoi(j).id;
                bbox_h = gt_bbox(i).hoi(j).bboxhuman;
                bbox_o = gt_bbox(i).hoi(j).bboxobject;
                conn = gt_bbox(i).hoi(j).connection;
                boxes = zeros(size(conn, 1), 8);
                for k = 1:size(conn, 1)
                    boxes(k, 1) = bbox_h(conn(k, 1)).x1;
                    boxes(k, 2) = bbox_h(conn(k, 1)).y1;
                    boxes(k, 3) = bbox_h(conn(k, 1)).x2;
                    boxes(k, 4) = bbox_h(conn(k, 1)).y2;
                    boxes(k, 5) = bbox_o(conn(k, 2)).x1;
                    boxes(k, 6) = bbox_o(conn(k, 2)).y1;
                    boxes(k, 7) = bbox_o(conn(k, 2)).x2;
                    boxes(k, 8) = bbox_o(conn(k, 2)).y2;
                end
                gt_all{hoi_id, i} = boxes;
            end
        end
    end
    fprintf('done.\n');

    % load detection
    switch format
        case 'obj'
            % dummy variable
            all_boxes = zeros(num_action, 1);
        case 'all'
            % load detection res (all object mode)
            det_file = [det_root 'detections.mat'];
            ld = load(det_file);
            all_boxes = ld.all_boxes;
    end

    % start parpool
    if ~exist('pool_size','var')
        poolobj = parpool();
    else
        poolobj = parpool(pool_size);
    end
    
    % warning off
    warning('off','MATLAB:mir_warning_maybe_uninitialized_temporary');

    % compute ap for each class    
    AP = zeros(num_action, 1);
    REC = zeros(num_action, 1);
    fprintf('start computing ap ... \n');
    parfor i = 1:num_action
        nname = list_action(i).nname;
        aname = [list_action(i).vname_ing '_' list_action(i).nname];
        fprintf('  %03d/%03d %-30s', i, num_action, aname);
        tic;
        % get detection results
        switch format
            case 'obj'
                % get object id and action id within the object category
                obj_id = cell_find_string(list_coco_obj, nname);
                act_id = i - obj_hoi_int(obj_id, 1) + 1;  %#ok
                assert(numel(obj_id) == 1);
                % load detection res (one object mode)
                det_file = [det_root 'detections_' num2str(obj_id,'%02d') '.mat'];
                ld = load(det_file);
                det = ld.all_boxes(act_id, :);
           case 'all'
                det = all_boxes(i, :);
        end
        % convert detection results
        det_id = zeros(0, 1);
        det_bb = zeros(0, 8);
        det_conf = zeros(0, 1);
        for j = 1:numel(det)
            if ~isempty(det{j})
                num_det = size(det{j}, 1);
                det_id = [det_id; j * ones(num_det, 1)];
                det_bb = [det_bb; det{j}(:, 1:8)];
                det_conf = [det_conf; det{j}(:, 9)];
            end
        end
        % convert zero-based to one-based indices
        det_bb = det_bb + 1;
        % get gt bbox
        assert(numel(det) == numel(gt_bbox));
        gt = gt_all(i, :);
        % adjust det & gt bbox by the evaluation mode
        switch eval_mode
            case 'def'
                % do nothing
            case 'ko'
                nid = cell_find_string({list_action.nname}', nname);  %#ok
                iid = find(any(anno_im(nid, :) == 1, 1));             %#ok
                assert(all(cellfun(@(x)isempty(x),gt(setdiff(1:numel(gt), iid)))) == 1);
                keep = ismember(det_id, iid);
                det_id = det_id(keep);
                det_bb = det_bb(keep, :);
                det_conf = det_conf(keep, :);
        end
        % compute ap
        [rec, prec, ap] = VOCevaldet_bboxpair(det_id, det_bb, det_conf, gt, ...
            min_overlap, aname, false);
        AP(i) = ap;
        if ~isempty(rec)
            REC(i) = rec(end);
        end
        fprintf('  ap: %.4f  rec: %.4f', ap, REC(i));
        fprintf('  time: %.3fs\n', toc);
    end
    fprintf('done.\n');
    
    % warning on
    warning('on','MATLAB:mir_warning_maybe_uninitialized_temporary');

    % delete parpool
    delete(poolobj);
    
    % save AP
    % save(res_file, 'AP', 'REC');
end

% get number of instances for each class
num_inst = zeros(num_action, 1);
for i = 1:numel(bbox.bbox_train)
    for j = 1:numel(bbox.bbox_train(i).hoi)
        if ~bbox.bbox_train(i).hoi(j).invis
            hoi_id = bbox.bbox_train(i).hoi(j).id;
            num_inst(hoi_id) = ...
                num_inst(hoi_id) + size(bbox.bbox_train(i).hoi(j).connection,1);
        end
    end
end

s_ind = num_inst < 10;
p_ind = num_inst >= 10;
n_ind = zeros(num_action, 1);
no_interactions = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107, 111, 129, 146, 160, 170, 174, 186, 194, 198, 208, 214, 224, 232, 235, 239, 243, 247, 252, 257, 264, 273, 283, 290, 295, 305, 313, 325, 330, 336, 342, 348, 352, 356, 363, 368, 376, 383, 389, 393, 397, 407, 414, 418, 429, 434, 438, 445, 449, 453, 463, 474, 483, 488, 502, 506, 516, 528, 533, 538, 546, 550, 558, 562, 567, 576, 584, 588, 595, 600];
for i = 1:length(no_interactions)
    n_ind(no_interactions(i)) = 1;
end
i_ind = n_ind == 0;
n_ind = n_ind > 0;

z_ind = zeros(num_action, 1);

zero_inters = [140] + 1;
if strfind(exp_name, 'zsrare')
    rare_inters = [63, 112, 165, 166, 172, 214, 227, 328, 404, 426, 431, 451, 514, 549, 550, 551, 578, 580] + 1;

    s_ind = zeros(num_action, 1);
    for i = 1:length(rare_inters)
        s_ind(rare_inters(i)) = 1;
    end
    s_ind = s_ind > 0;

    zero_inters = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416, 389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596, 345, 189, 205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229, 158, 195, 238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188, 216, 597, 77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104, 55, 50, 198, 168, 391, 192, 595, 136, 581] + 1;
end


if strfind(exp_name, 'zsnrare')
    nonrare_inters = [0, 1, 2, 3, 4, 5, 10, 12, 13, 14, 15, 16, 17, 21, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 43, 46, 47, 49, 51, 52, 57, 58, 59, 65, 68, 69, 71, 72, 74, 78, 79, 81, 82, 85, 86, 87, 88, 92, 95, 96, 97, 98, 101, 102, 103, 105, 108, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 132, 133, 134, 137, 138, 142, 143, 144, 145, 147, 148, 150, 151, 156, 157, 160, 161, 164, 167, 170, 171, 174, 175, 177, 178, 180, 182, 183, 186, 187, 194, 196, 199, 200, 201, 202, 203, 204, 207, 209, 210, 211, 213, 217, 219, 220, 221, 224, 225, 231, 232, 233, 234, 235, 236, 237, 241, 242, 243, 244, 247, 251, 252, 253, 256, 258, 259, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 275, 276, 277, 278, 282, 283, 285, 287, 288, 290, 291, 293, 295, 296, 298, 299, 300, 301, 302, 305, 306, 307, 309, 310, 313, 314, 316, 318, 319, 321, 322, 324, 326, 327, 330, 331, 332, 335, 336, 339, 341, 342, 343, 344, 347, 348, 349, 352, 353, 355, 356, 357, 359, 360, 361, 362, 363, 365, 366, 367, 368, 369, 372, 373, 375, 376, 378, 380, 382, 384, 386, 388, 392, 393, 394, 396, 400, 406, 408, 409, 411, 412, 413, 414, 415, 419, 420, 421, 422, 423, 424, 425, 428, 430, 432, 433, 434, 435, 437, 438, 441, 442, 443, 444, 445, 447, 448, 450, 452, 453, 454, 458, 460, 462, 464, 465, 466, 467, 468, 473, 475, 476, 477, 484, 486, 487, 488, 489, 490, 491, 492, 494, 495, 496, 497, 500, 501, 502, 503, 505, 507, 510, 511, 512, 513, 519, 521, 523, 524, 525, 527, 528, 530, 532, 533, 536, 537, 538, 540, 541, 542, 543, 545, 553, 554, 557, 558, 561, 562, 563, 564, 565, 567, 568, 569, 570, 571, 573, 574, 579, 584, 585, 587, 588, 590, 591, 598] + 1;
    p_ind = zeros(num_action, 1);
    for i = 1:length(nonrare_inters)
        p_ind(nonrare_inters(i)) = 1;
    end
    p_ind = p_ind > 0;
    zero_inters = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61, 457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73, 159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346,456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572,529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329, 246, 173, 506,383, 93, 516, 64] + 1;
end


if strfind(exp_name, 'zsuo')

    rare_inters = [514, 517, 8, 520, 522, 526, 531, 22, 27, 539, 546, 547, 548, 549, 550, 551, 552, 555, 44, 556, 50, 55, 62, 63, 66, 578, 580, 581, 70, 586, 76, 77, 80, 592, 593, 83, 84, 90, 99, 100, 104, 107, 135, 136, 149, 158, 165, 166, 168, 172, 179, 181, 184, 188, 189, 192, 195, 198, 205, 206, 214, 216, 222, 238, 239, 254, 255, 257, 260, 261, 262, 274, 279, 280, 281, 286, 289, 303, 311, 325, 328, 333, 334, 345, 350, 351, 354, 358, 364, 379, 381, 389, 390, 391, 395, 397, 398, 399, 401, 402, 403, 404, 405, 407, 410, 416, 436, 439, 440, 449, 451, 474, 482, 485, 498, 499, 504, 509] + 1;
    s_ind = zeros(num_action, 1);
    for i = 1:length(rare_inters)
        s_ind(rare_inters(i)) = 1;
    end
    s_ind = s_ind > 0;


    nonrare_inters = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 78, 79, 81, 82, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 101, 102, 103, 105, 106, 108, 109, 110, 129, 130, 131, 132, 133, 134, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 164, 167, 169, 170, 171, 173, 174, 175, 176, 177, 178, 180, 182, 183, 185, 186, 187, 190, 191, 193, 194, 196, 197, 199, 200, 201, 202, 203, 204, 207, 208, 209, 210, 211, 212, 213, 215, 217, 218, 219, 220, 221, 223, 232, 233, 234, 235, 236, 237, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 256, 258, 259, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275, 276, 277, 278, 282, 283, 284, 285, 287, 288, 295, 296, 297, 298, 299, 300, 301, 302, 304, 305, 306, 307, 308, 309, 310, 312, 326, 327, 329, 330, 331, 332, 335, 342, 343, 344, 346, 347, 348, 349, 352, 353, 355, 356, 357, 359, 360, 361, 362, 363, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 380, 382, 383, 384, 385, 386, 387, 388, 392, 393, 394, 396, 400, 406, 408, 409, 411, 412, 413, 414, 415, 417, 434, 435, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 450, 452, 475, 476, 477, 478, 479, 480, 481, 483, 484, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 500, 501, 502, 503, 505, 506, 507, 508, 510, 511, 512, 513, 515, 516, 518, 519, 521, 523, 524, 525, 527, 528, 529, 530, 532, 538, 540, 541, 542, 543, 544, 545, 553, 554, 557, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 579, 582, 583, 584, 585, 587, 588, 589, 590, 591, 594] + 1;
    p_ind = zeros(num_action, 1);
    for i = 1:length(nonrare_inters)
        p_ind(nonrare_inters(i)) = 1;
    end
    p_ind = p_ind > 0;
    zero_inters = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293, 294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337, 338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536, 537, 558, 559, 560, 561, 595, 596, 597, 598, 599] + 1;
end


if strfind(exp_name, 'zs_')

    rare_inters = [514, 517, 8, 520, 522, 526, 531, 22, 535, 27, 539, 546, 547, 548, 549, 550, 551, 552, 555, 44, 556, 560, 50, 55, 62, 63, 578, 580, 581, 586, 76, 77, 80, 592, 593, 83, 84, 595, 596, 597, 599, 99, 100, 104, 107, 112, 127, 135, 136, 149, 158, 165, 166, 168, 172, 179, 181, 184, 188, 189, 192, 195, 198, 205, 206, 214, 216, 222, 227, 229, 238, 239, 254, 255, 257, 260, 261, 262, 274, 281, 292, 315, 317, 328, 333, 334, 345, 350, 354, 364, 381, 390, 391, 395, 397, 398, 399, 401, 403, 404, 405, 407, 410, 426, 429, 431, 436, 440, 449, 451, 463, 469, 474, 482] + 1;
    s_ind = zeros(num_action, 1);
    for i = 1:length(rare_inters)
        s_ind(rare_inters(i)) = 1;
    end
    s_ind = s_ind > 0;

    nonrare_inters = [0, 1, 2, 3, 4, 5, 10, 12, 13, 14, 15, 16, 17, 21, 24, 25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 43, 46, 47, 49, 51, 52, 54, 57, 58, 59, 64, 65, 68, 69, 71, 72, 74, 78, 79, 81, 82, 85, 86, 87, 88, 92, 93, 95, 96, 97, 98, 101, 102, 103, 105, 108, 109, 110, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 132, 133, 134, 137, 138, 142, 143, 144, 145, 147, 148, 150, 151, 156, 157, 160, 161, 163, 164, 167, 170, 171, 173, 174, 175, 177, 178, 180, 182, 183, 186, 187, 193, 194, 196, 197, 199, 200, 201, 202, 203, 204, 207, 209, 210, 211, 213, 217, 219, 220, 221, 224, 225, 231, 232, 233, 234, 235, 236, 237, 241, 242, 243, 244, 246, 247, 251, 252, 253, 256, 258, 259, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 275, 276, 277, 278, 282, 283, 285, 287, 288, 290, 291, 293, 295, 296, 298, 299, 300, 301, 302, 305, 306, 307, 308, 309, 310, 313, 314, 316, 318, 319, 321, 322, 324, 326, 327, 329, 330, 331, 332, 335, 336, 339, 340, 341, 342, 343, 344, 347, 348, 349, 352, 353, 355, 356, 357, 359, 360, 361, 362, 363, 365, 366, 367, 368, 369, 372, 373, 375, 376, 378, 380, 382, 383, 384, 386, 387, 388, 392, 393, 394, 396, 400, 406, 408, 409, 411, 412, 413, 414, 415, 417, 419, 420, 421, 422, 423, 424, 425, 428, 430, 432, 433, 434, 435, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 450, 452, 453, 454, 455, 458, 460, 462, 464, 465, 466, 467, 468, 473, 475, 476, 477, 483, 484, 486, 487, 488, 489, 490, 491, 492, 494, 495, 496, 497, 500, 501, 502, 503, 505, 506, 507, 508, 510, 511, 512, 513, 516, 519, 521, 523, 524, 525, 527, 528, 530, 532, 533, 534, 536, 537, 538, 540, 541, 542, 543, 545, 553, 554, 557, 558, 561, 562, 563, 564, 565, 567, 568, 569, 570, 571, 573, 574, 575, 579, 584, 585, 587, 588, 590, 591, 598] + 1;

    p_ind = zeros(num_action, 1);
    for i = 1:length(nonrare_inters)
        p_ind(nonrare_inters(i)) = 1;
    end
    p_ind = p_ind > 0;
    zero_inters = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416, 389, 90, 38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61, 457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73, 159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346, 456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572, 529, 312, 9] + 1;
end


for i = 1:length(zero_inters)
    z_ind(zero_inters(i)) = 1;
end
nz_ind = z_ind == 0;
z_ind = z_ind > 0;


fprintf('\n');
fprintf('setting:     %s\n', eval_mode);
fprintf('exp_name:    %s\n', exp_name);
fprintf('score_blob:  %s\n', score_blob);
fprintf('\n');
fprintf('  mAP / mRec (full):      %.4f / %.4f\n', mean(AP), mean(REC));
fprintf('\n');
fprintf('  mAP / mRec (rare):      %.4f / %.4f\n', mean(AP(s_ind)), mean(REC(s_ind)));
fprintf('  mAP / mRec (non-rare):  %.4f / %.4f\n', mean(AP(p_ind)), mean(REC(p_ind)));
fprintf('  mAP / mRec (no-inte):  %.4f / %.4f\n', mean(AP(n_ind)), mean(REC(n_ind)));
fprintf('  mAP / mRec (inter):  %.4f / %.4f\n', mean(AP(i_ind)), mean(REC(i_ind)));
fprintf('  mAP / mRec (zero-shot):  %.4f / %.4f\n', mean(AP(z_ind)), mean(AP(nz_ind)));

fprintf(' %.4f / %.4f     %.4f / %.4f      %.4f / %.4f    \n', mean(AP), mean(REC), mean(AP(s_ind)), mean(REC(s_ind)), mean(AP(p_ind)), mean(REC(p_ind)));
fprintf('\n');
% store the results.
fid = fopen(result_file, 'a');
fprintf(fid, ' %.4f  %.4f    %.4f  %.4f     %.4f  %.4f     %.4f  %.4f', mean(AP), mean(REC), mean(AP(s_ind)), mean(REC(s_ind)), mean(AP(p_ind)), mean(REC(p_ind)), mean(AP(n_ind)), mean(AP(i_ind)),  mean(AP(z_ind)), mean(AP(nz_ind)));
if strcmp(eval_mode, 'ko')
    fprintf(fid, '\n');
end
fclose(fid);

%save([exp_name eval_mode '.mat'],'AP');
end