import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'paper.mplstyle')
plt.style.use(filename)

def find_path(path):
    if path == 'default':
        data_dir = os.path.join(dirname, '../data/GRBData')
    else:
        data_dir = path
    return data_dir