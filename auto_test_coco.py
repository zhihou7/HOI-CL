import os.path as osp
import sys

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



if __name__ == '__main__':
    max_iteration = 300001
    args = parse_args()
    print(args)
    from ult.config import cfg
    model = args.model
    stride = 10000
    import time
    while True:
        if args.model.__contains__('*'):
            print('line', args.model)
            import re

            r = re.compile(args.model)
            model_arr = list(filter(r.match, list(
                set(os.listdir('/opt/data/private/Weights/')))))
            model_arr = [item for item in model_arr if item.__contains__('VCOCO')]
            model_arr = sorted(model_arr)
            print(model_arr)
        else:
            model_arr = args.model.split(',')
        for i, model in enumerate(model_arr):
            result_file_name = '/opt/data/private/coco_csv/{}_{}.csv'.format(model, 'scenario_1')
            iter_list = list(range(stride, 600001, stride)) + list(range(19999, 600001, stride))
            cur_list = []
            if os.path.exists(result_file_name):
                f = open(result_file_name, 'r')
                max_iter = stride
                for item in f.readlines():
                    if len(item.strip().split(' ')) < 2:
                        print(result_file_name)
                        continue
                    temp = int(float(item.strip().split(' ')[0]))
                    cur_list.append(temp)
                    max_iter = max(temp, max_iter)
                f.close()
                # print('%s start from %d' %(model, max_iter + stride))
                # iter_list = range(max_iter + stride, max_iteration, stride)
            for index in iter_list:
                if index in cur_list:
                    continue
                import os

                if not os.path.exists(cfg.LOCAL_DATA0 + '/Weights/' + model + '/HOI_iter_' + str(index) + '.ckpt.index'):
                    # print('not exist', cfg.LOCAL_DATA0 + '/Weights/' + model + '/HOI_iter_' + str(index) + '.ckpt.index')
                    continue
                output_file = cfg.LOCAL_DATA + '/Results/' + str(index) + '_' + model + '.pkl'
                print(os.path.exists(output_file), output_file)
                if os.path.exists(output_file):
                    continue
                iteration = index

                output_file = cfg.LOCAL_DATA + '/Results/' + str(iteration) + '_' + model + '.pkl'
                open(output_file, 'a').close()

                if model.__contains__('CL'):
                    os.system(
                        'python tools/Test_R_ResNet_VCOCO_CL.py --num_iteration %d --model %s' % (iteration, model))
                elif model.__contains__('24'):
                    os.system('python tools/Test_R_ResNet_VCOCO_24.py --num_iteration %d --model %s' % (iteration, model))
                else:
                    os.system('python tools/Test_R_ResNet_VCOCO.py --num_iteration %d --model %s' % (iteration, model))
                # if os.path.getsize(output_file) < 5000:
                #     print("fail", output_file, os.path.getsize(output_file))
                #     os.remove(output_file)
                #     continue
                # os.system(
                #     'cd Data/ho-rcnn/;python ../../tools/postprocess_test.py --num_iteration %d --model %s --fuse_type %s;cd ../../' % (
                #         iteration, model, 'spv'))
                print('continue', index + stride)
        time.sleep(30)