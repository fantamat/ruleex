import numpy as np
import scipy.stats as stat


def test_t(x, y, level, equal_var=True):
    """
    statistical test of the sets of labels x and y based on t statistics
    Assumes normal distributed sets of samples x and y
    :param x: a list of labels
    :param y: a list of labels
    :param level:
    :param equal_var: True if the variance of the x and y are the same
    :return: True if the test passed, i.e., the sets have the same mean
    """
    if len(x) < 2 or len(y) < 2:
        return True
    s, p_value = stat.ttest_ind(x, y, 0, equal_var=equal_var)
    if np.any(p_value < level):
        return False
    else:
        return True


def test_welch(x, y, level):
    """
    statistical test of the sets of labels x and y based on t statistics
    Assumes the same variance of x, y and their normality
    :param x: a list of labels
    :param y: a list of labels
    :param level:
    :return: True if the test passed, i.e., the sets have the same mean
    """
    return test_t(x, y, level, equal_var=False)


def test_F(x, y, level):
    """
    statistical test of the sets of labels x and y based on F statistics
    :param x: a list of labels
    :param y: a list of labels
    :param level:
    :return: True if the test passed, i.e., the sets have the same mean
    """
    if len(x) < 2 or len(y) < 2:
        return True
    vx = np.var(x, 0, ddof=1)
    vy = np.var(y, 0, ddof=1)
    vx, vy = vx[vx*vy>0], vy[vx*vy>0]
    if len(vx)==0:
        return False
    F = vx/vy
    p_value = stat.f.cdf(F, len(x)-1, len(y)-1)
    p_value = 2*np.min([p_value, 1-p_value], axis=0)
    if np.any(p_value < level):
        return False
    else:
        return True


def test_chi2(y0, y1, level):
    """
    statistical test of the sets of labels x and y based on Chi2 statistics
    :param x: a list of labels
    :param y: a list of labels
    :param level:
    :return: True if the test passed, i.e., the sets have the same mean
    """
    if len(y0) == 0 or len(y1) == 0:
        return True
    l0 = np.argmax(y0, axis=1)
    l1 = np.argmax(y1, axis=1)
    v, c = np.unique(np.append(l0,l1), return_counts=True)
    v0, c0 = np.unique(l0, return_counts=True)
    v1, c1 = np.unique(l1, return_counts=True)
    p = np.zeros(len(y0[0]))
    p0 = p.copy()
    p1 = p.copy()
    p[v] = c / np.sum(c)
    p0[v0] = c0 / np.sum(c0)
    p1[v1] = c1 / np.sum(c1)
    p0[p0==0] = 0.05
    p1[p1 == 0] = 0.05
    p[p==0] = 0.05
    _, p0_value = stat.chisquare(p0, p)
    _, p1_value = stat.chisquare(p1, p)
    if 1-p0_value > level or 1-p1_value > level:
        return False
    else:
        return True

