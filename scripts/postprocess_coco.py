import os.path as osp
import sys
import _init_paths
from ult.config import cfg

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '.', 'lib')
add_path(lib_path)

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on HICO')
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_ResNet50_HICO', type=str)
    parser.add_argument('--fuse_type', dest='fuse_type', default='spho')
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.8, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.6, type=float)

    args = parser.parse_args()
    return args

def get_vcoco():
    from ult.vsrl_eval import VCOCOeval
    vcocoeval = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json',
                          cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json',
                          cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
    return vcocoeval


if __name__ == '__main__':
    vcocoeval = get_vcoco()
    max_iteration = 300001
    args = parse_args()
    print(args)
    from ult.config import cfg
    # model = 'VCL_mask_vloss_ResNet_HICO'
    model = args.model
    stride = 10000
    import time
    while True:
        filesss = os.listdir(cfg.LOCAL_DATA + '/Results/')
        import re
        r = re.compile(args.model)
        filesss = list(filter(r.match, filesss))
        filesss = sorted(filesss)
        filesss = [item for item in filesss if item.__contains__('VCOCO')]
        filesss = filesss[::-1]
        print(filesss)
        #
        # if args.model.__contains__('*'):
        #     print('line', args.model)
        #     import re
        #
        #     r = re.compile(args.model)
        #     model_arr = list(filter(r.match, list(
        #         set(os.listdir('/opt/data/private/Weights/')))))
        #     model_arr = [item for item in model_arr if item.__contains__('VCOCO')]
        #     model_arr = sorted(model_arr)
        #     model_arr = model_arr[::-1]
        #     print(model_arr)
        # else:
        #     model_arr = args.model.split(',')
        for pklfile in filesss:
            iteration = int(pklfile.split('_')[0])
            fuse_type = args.fuse_type
            model_suffix = pklfile[len(str(iteration)) + 1:]
            model = model_suffix.replace('.pkl', '')
            result_file_name = cfg.LOCAL_DATA +'/coco_csv/{}_{}.csv'.format(model, 'scenario_1')
            print(result_file_name)
            iter_list = list(range(stride, 600001, stride)) + list(range(19999, 600001, stride))
            cur_list = []
            existing = False
            is_processing = False
            if os.path.exists(result_file_name):
                f = open(result_file_name, 'r')
                max_iter = stride
                for item in f.readlines():
                    if len(item.strip()) < 3:
                        continue
                    if len(item.strip().split(' ')) < 2:
                        print(result_file_name)
                        is_processing = True
                        break
                    temp = int(float(item.strip().split(' ')[0]))
                    cur_list.append(temp)
                    max_iter = max(temp, max_iter)
                    if temp == iteration:
                        existing = True
                        break
                f.close()
                # print('%s start from %d' %(model, max_iter + stride))
                # iter_list = range(max_iter + stride, max_iteration, stride)
            if is_processing:
                break
            import os
            if not existing:
                print("not exists", result_file_name)
                output_file = cfg.LOCAL_DATA + '/Results/' + str(iteration) + '_' + model + '.pkl'
                if not os.path.exists(output_file) or os.path.getsize(output_file) < 1000:
                    print("not exists", output_file)
                    continue

                if os.path.exists(cfg.LOCAL_DATA + '/coco_csv/{}_{}.csv'.format(model, 'scenario_1')):
                    f = open(cfg.LOCAL_DATA + '/coco_csv/{}_{}.csv'.format(model, 'scenario_1'), 'r')
                    results = f.readlines()
                    f.close()
                    if len(results) > 0 and results[-1].startswith(str(iteration)):
                        continue
                print("processing", output_file)
                f = open(cfg.LOCAL_DATA + '/coco_csv/{}_{}.csv'.format(model, 'scenario_1'), 'a')
                f.write('%s ' % (str(iteration)))
                f.flush()
                f.close()
                import os
                vcocoeval._do_eval(output_file, ovr_thresh=0.5)
                if os.path.exists(output_file):
                    os.remove(output_file)

                print('continue', iteration + stride)
            else:
                print('exist', pklfile)
                # if os.path.exists(cfg.LOCAL_DATA + '/Results/' + pklfile):
                #     os.remove(cfg.LOCAL_DATA + '/Results/' + pklfile)
        time.sleep(30)