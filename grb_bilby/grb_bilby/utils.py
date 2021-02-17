import os

import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'paper.mplstyle')
plt.style.use(filename)


def find_path(path):
    if path == 'default':
        return os.path.join(dirname, '../data/GRBData')
    else:
        return path
