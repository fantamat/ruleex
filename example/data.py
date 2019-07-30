import numpy as np
import random



def classification1(y):
    if   (y[0] >= 0.2) and (y[1] < 0.1):
        return 0
    elif (y[0] >= -0.5) and (y[0] < -0.25):
        return 1
    elif (y[0] > 0.4) and (y[1] >= 0.15):
        return 2
    elif y[1] < 0:
        return 3
    else:
        return 4


def split_data(y):
    r1 = random.randint(0, 4)
    r2 = random.randint(0, 15)%11
    if r1 == 0:
        v1 = [y[0] / 2, 0, 0]
    elif r1 == 1:
        v1 = [0, y[0] / 3, y[0] * 2 / 3]
    elif r1 == 2:
        v1 = [0, 0, y[0]]
    elif r1 == 3:
        v1 = [0, y[0], 0]
    elif r1 == 4:
        v1 = [y[0] / 6, 0, y[0] * 2 / 3]
    v2 = [0 for _ in range(11)]
    v2[r2] = y[1]
    return v1 + v2


def gen_sparse_linear(seed=None, class_fun=classification1):
    """
    :return:


    """
    np.random.seed(seed)
    test_ratio = 0.1
    samples = 1000
    impurity = 0.05

    y = np.reshape(np.random.uniform(-1,1,samples*2), (samples, 2))
    l = np.apply_along_axis(class_fun, 1, y)
    # add wrongli classified samples
    for i in range(len(l)):
        if np.random.rand() < impurity:
            l[i] = np.random.random_integers(0,5,1)
    x = np.apply_along_axis(split_data, 1, y)

    s = round(samples*test_ratio)
    return (
        x[s:],
        l[s:],
        x[:s],
        l[:s]
    )




x, y, x_test, y_test = gen_sparse_linear(seed=55)
a = 5
"""
k = x[y==1]
plt.plot(k[:, 0]*2+k[:, 1]+k[:, 2], sum(k[:, i] for i in range(3, 14)), "+")
"""