import os
DEFAULT_FN = '.tmp.txt'

def parse(lines):
    ret = dict()
    for line in lines:
        line = line.strip()
        k, v = line.split(":")
        ret[k] = v
    return ret

def read_data(fn=None):
    fn = fn or DEFAULT_FN
    dic = {}
    try:
        with open(fn, 'r') as f:
            lines = f.readlines()
            dic = parse(lines)
    except:
        print('No this file: {}'.format(fn))
    return dic

def write_data(dic, fn=None, append=False, desc=None):
    fn = fn or DEFAULT_FN
    mode = 'a' if append else 'w'
    with open(fn, mode) as f:
        if desc: f.write(desc+'\n')
        for k, v in dic.items():
            line = str(k) + ':' + str(v) + '\n'
            f.write(line)

if __name__ == '__main__':
    test_data = {'Top1': 8.57, 'Top5': 20.67}
    write_data(test_data)

        

