import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '.', 'lib')
add_path(lib_path)

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on HICO')
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='.*aug.*', type=str)
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

    args = parse_args()
    print(args)
    from ult.config import cfg
    # model = 'VCL_mask_vloss_ResNet_HICO'
    model = args.model

    import os

    stride = 5000
    import time
    while True:
        if args.model.__contains__('*'):
            print('line', args.model)
            import re

            r = re.compile(args.model)

            import glob
            tmp = glob.glob(cfg.LOCAL_DATA + '/Weights/*')
            tmp.sort(key=os.path.getmtime, reverse=True)
            tmp = tmp[:40]
            # tmp = tmp[-40:]
            print(tmp)
            # tmp = list(set(os.listdir(DATA_DIR + '/Weights/')))
            tmp = [item.split('/')[-1] for item in tmp]
            model_arr = list(filter(r.match, tmp))
            # model_arr = sorted(model_arr)

            # print(cfg.LOCAL_DATA)
            # model_arr = list(filter(r.match, list(os.listdir(cfg.LOCAL_DATA + '/Weights/'))))
            model_arr = [item for item in model_arr if not item.__contains__('VCOCO')]
        else:
            model_arr = args.model.split(',')
        print(model_arr)
        model_arr = sorted(model_arr)
        model_arr = model_arr[::-1]
        for i, model in enumerate(model_arr):
            if model.__contains__('zs4'):
                stride = 5000
            else:
                stride = 5000
            for index in list(range(stride, 3000001, stride)) + list(range(43273, 3000001, 43273)):
                import os
                # print(cfg.LOCAL_DATA + '/Weights/' + model + '/HOI_iter_' + str(index) + '.ckpt.index')
                if not (os.path.exists(cfg.ROOT_DIR + '/Weights/' + model + '/HOI_iter_' + str(index) + '.ckpt.index')
                        or os.path.exists(cfg.LOCAL_DATA + '/Weights/' + model + '/HOI_iter_' + str(index) + '.ckpt.index')):
                    # print('not exist', index)
                    continue
                model = model.strip()
                # print(os.path.exists(cfg.LOCAL_DATA + '/Results/' + str(index) + '_' + model + '_tin.pkl'), cfg.LOCAL_DATA + '/Results/' + str(index) + '_' + model + '_tin.pkl')
                if not os.path.exists(cfg.LOCAL_DATA + '/Results/' + str(index) + '_' + model + '_tin.pkl') \
                        and not os.path.exists(cfg.LOCAL_DATA + '/Results/' + str(index) + '_' + model + '.pkl'):

                    exists = False
                    for suffix in ['_tin', '']:
                        fuse_type = 'spv'
                        # print(model)
                        if model.__contains__('_R_V'): fuse_type = 'v'
                        result_file_name = cfg.LOCAL_DATA + '/csv/' + model + '_' + str(fuse_type) + suffix + '.csv'

                        if os.path.exists(result_file_name):
                            f = open(result_file_name, 'r')
                            max_iter = stride
                            for item in f.readlines():
                                # print(item, result_file_name)
                                # temp = int(item.strip().split(' ')[0])
                                temp = item.strip().split(' ')[0]
                                temp = float(temp)
                                temp = int(temp)
                                if temp == index:
                                    exists = True
                                    break
                            f.close()
                    if exists:
                        continue

                    print('', index, model)
                    iteration = index

                    output_file = cfg.LOCAL_DATA + '/Results/' + str(iteration) + '_' + model + '_tin.pkl'
                    open(output_file, 'a').close()
                    os.system('python tools/Test_VCL_ResNet_HICO.py --num_iteration %d --model %s' % (iteration, model))
                    if os.path.getsize(output_file) < 50000:
                        print("fail", output_file, os.path.getsize(output_file))
                        os.remove(output_file)
                        continue
                    # os.system(
                    #     'cd Data/ho-rcnn/;python ../../tools/postprocess_test.py --num_iteration %d --model %s --fuse_type %s;cd ../../' % (
                    #         iteration, model, 'spv'))
                # print('continue', index + stride)
        time.sleep(60)