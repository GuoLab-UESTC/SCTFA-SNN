import numpy as np
import scipy.io as scio
import os


def my_chunk_evs_pol_dvs(data, dt=1000 * 25, T=20, size=[2, 128, 128]):
    t_start = data[0][0]
    ts = range(t_start, t_start + T * dt, dt)
    chunks = np.zeros([len(ts)] + size, dtype='int8')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        while data[idx_end, 0] < t + dt:
            idx_end += 1
        idx_end -= 1
        if idx_end > idx_start:
            ee = data[idx_start:idx_end, 3:]
            ee[ee == -1] = 0
            pol, x, y = ee[:, 2], ee[:, 0], ee[:, 1]
            np.add.at(chunks, (i, pol, x, y), 1)
        idx_start = idx_end + 1
    return chunks


if __name__ == '__main__':
    w = []
    for n in range(10):
        for scale in [4]:
            for j in range(1, 1001):
                print('正在处理   数字', n, '   尺度为', scale, '   序号', j)
                str_scale = str(scale)
                str_j = str(j)
                if len(str_scale) == 1:
                    str_scale = '0' + str_scale
                for mm in range(4 - len(str_j)):
                    str_j = '0' + str_j
                path = './原始数据/matdata' + str(n) + \
                       '/scale' + str(scale) + \
                       '/mnist_' + str(n) + '_scale' + str_scale + '_' + str_j + '.mat'
                data = scio.loadmat(path)
                datasets = my_chunk_evs_pol_dvs(data['data'])
                os.makedirs('./T5/' + str(n) + '/scale' + str(scale), exist_ok=True)
                np.savez('./T5/' + str(n) + '/scale' + str(scale) + '/mnist_' + str(
                    n) + '_scale' + str_scale + '_' + str_j + '.npz', data=datasets, label=n)

