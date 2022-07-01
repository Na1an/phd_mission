import pandas as pd
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import laspy

# setting matplotlib
andy_theme = {'axes.grid': True,
              'grid.linestyle': '--',
              'legend.framealpha': 1,
              'legend.facecolor': 'white',
              'legend.shadow': True,
              'legend.fontsize': 14,
              'legend.title_fontsize': 16,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'axes.labelsize': 16,
              'axes.titlesize': 20,
              'figure.dpi': 100}

matplotlib.rcParams.update(andy_theme)

# plot histogram
def plot_histogram():
    

    return None

if __name__ == "__main__":
    print("###### start plotting histogram ######")
    las = lasio.read(r'/home/yuchen/Documents/PhD/phd_mission/src/keep_result/keep_15_not_bad_v_point_number_p_intensity.las')
    df = las.df()
    df.head()
