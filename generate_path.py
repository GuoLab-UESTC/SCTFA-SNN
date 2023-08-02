import numpy as np


for scale in [4]:
    path = []
    for n in range(10):
        for j in range(1, 1001):
            str_scale = str(scale)
            str_j = str(j)
            if len(str_scale) == 1:
                str_scale = '0' + str_scale
            for mm in range(4 - len(str_j)):
                str_j = '0' + str_j
            path.append('/' + str(n) + \
                   '/scale' + str(scale) + \
                   '/mnist_' + str(n) + '_scale' + str_scale + '_' + str_j + '.npz' + '    ' + str(n))
    path = np.array(path)
    np.random.shuffle(path)
    np.savetxt('path_' + str(scale) + '.txt', path, fmt="%s")
