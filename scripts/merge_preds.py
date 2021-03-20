
if __name__ == '__main__':
    import sys
    import pickle
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    print('f1', file1)
    print('f2', file2)
    HICO1 = pickle.load(open(file1, "rb"), encoding='latin1')
    HICO2 = pickle.load(open(file2, "rb"), encoding='latin1')


    new_result = {}
    for key, value in HICO1.items():
        res_item = []
        for i in range(len(value)):
            if len(HICO1[key]) != len(HICO2[key]):
                print(key, i)
            if len(HICO1[key][i]) != len(HICO2[key][i]):
                print(key, i)
            res_item.append([(HICO1[key][i][j] + HICO2[key][i][j]) / 2 for j in range(len(HICO1[key][i]))])
        new_result[key] = res_item
    print(file1, '\n', file2)
    pickle.dump(new_result, open('tmp_tin.pkl', 'wb'))