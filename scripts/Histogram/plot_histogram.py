import pandas as pd
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
import argparse as ap
from utility import *

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
def plot_histogram(data_las):
    '''
    Args:
        data_las: a las file.
    Return:
        None. 
    '''
    print(">> type data_las.wood_proba ={}".format(type(data_las.wood_proba)))
    res = np.array([data_las.wood_proba, data_las.leave_proba, data_las.llabel]).T
    print("res.shape={}".format(res.shape))
    df = pd.DataFrame(res, columns=['wood_proba','leave_proba','llabel'])
    #print(df.describe())
    feature = 'leave_proba'
    # KDE
    #sns.kdeplot(data=df, x=feature, hue="llabel", common_norm=False, bw_method=0.15).set_title("Kernel Density Estimation (KDE) plots for predicted pribabilities: {} part".format(feature))

    '''
    # KDE filled
    below0 = norm.cdf(x=0,loc=df[feature],scale=0.15)
    above1 = 1- norm.cdf(x=1,loc=df[feature],scale=0.15)
    df['edgeweight'] = 1/ (1 - below0 - above1)
    cum_plot_wood = sns.kdeplot(data=df, x=feature, hue="llabel", 
                       common_norm=False, bw_method=0.15,
                       clip=(0,1), weights='edgeweight', 
                       multiple="fill", legend=True)

    cum_plot_wood.legend_._set_loc(4)
    cum_plot_wood.set_title("Kernel Density Estimation (KDE) plots (filled) for predicted pribabilities: {} part".format(feature))
    '''

    '''
    viol_tree = sns.violinplot(x='wood_proba', y="leave_proba", hue="llabel",
                          data=df, split=True, cut=0, 
                          bw=0.15, inner=None,
                          scale='count', scale_hue=False)
    #viol_tree.legend_.set_bbox_to_anchor((0.65, 0.95))
    '''
    
    plt.show()
    return None

# plot raw data
def plot_raw_hist_keepithis(data_las, feature):
    # 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 'classification', 'synthetic', 
    # 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time', 'red', 'green', 'blue', 'WL'
    #features = np.array([data_las.llabel, data_las.intensity, data_las.point_source_id, data_las.user_data, data_las.scan_angle_rank, data_las.number_of_returns, data_las.return_number, (data_las.return_number*data_las.intensity)/data_las.number_of_returns]).T
    features = np.array([data_las.WL, data_las.intensity, data_las.point_source_id, data_las.user_data, data_las.scan_angle_rank, data_las.number_of_returns, data_las.return_number, data_las.intensity_transform_pct]).T

    print("features.shape={}".format(features.shape))
    df = pd.DataFrame(features, columns=['WL','intensity','point_source_id', 'user_data', 'scan_angle_rank', 'number_of_returns', 'return_number','intensity_moyen'])
    
    #feature = 'intensity'
    #sns.kdeplot(data=df, x=feature, hue="WL", common_norm=False, bw_method=0.15).set_title("Kernel Density Estimation (KDE) plots for predicted pribabilities: {} part".format(feature))
    # kde plot
    #sns.histplot(data=df, x=feature, hue="WL", kde=True).set_title("Kernel Density Estimation (KDE) plots for predicted pribabilities: {} part".format(feature))
    sns.histplot(data=df, x=feature, hue="return_number", kde=True).set_title("Kernel Density Estimation (KDE) plots for predicted pribabilities: {} part".format(feature))
    plt.show()
    # boxplot
    sns.boxplot(x=df['WL'], y=df['intensity_moyen'], hue=df['return_number']).set_title("boxplots for predicted pribabilities: {} part".format(feature))
    plt.show()
    return None

# plot simulated data
def plot_raw_hist(data_las, feature):
    # 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 'classification', 'synthetic', 
    # 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time', 'red', 'green', 'blue', 'WL'
    #features = np.array([data_las.llabel, data_las.intensity, data_las.point_source_id, data_las.user_data, data_las.scan_angle_rank, data_las.number_of_returns, data_las.return_number, (data_las.return_number*data_las.intensity)/data_las.number_of_returns]).T
    features = np.array([data_las.gps_time, data_las.scan_angle*65535,data_las.number_of_returns, data_las.return_number]).T

    print("features.shape={}".format(features.shape))
    df = pd.DataFrame(features, columns=['gps_time','intensity_moyen','number_of_returns', 'return_number'])
    df['max_return_veg'] = df.groupby('gps_time')['return_number'].transform('max')
    #df = df[np.where(df['max_return_veg']==df['number_of_returns'])]
    df_abc = df[(df['max_return_veg']==df['number_of_returns'])]
    print(df.describe())    
    print(df_abc.describe())
    
    feature = 'intensity_moyen'
    #sns.kdeplot(data=df, x="number_of_returns", common_norm=False, bw_method=0.15).set_title("Kernel Density Estimation (KDE) plots for predicted pribabilities: {} part".format(feature))
    # kde plot
    #sns.histplot(data=df, x=feature, hue="WL", kde=True).set_title("Kernel Density Estimation (KDE) plots for predicted pribabilities: {} part".format(feature))
    #sns.histplot(data=df, x=feature, hue="return_number", kde=True).set_title("Kernel Density Estimation (KDE) plots for predicted pribabilities: {} part".format(feature))
    # boxplot
    sns.boxplot(x=df_abc['number_of_returns'], y=df_abc['intensity_moyen'], hue=df_abc["return_number"]).set_title("boxplots for predicted pribabilities: {} part".format(feature))
    plt.show()
    return None


if __name__ == "__main__":
    print("###### start plotting histogram ######")
    
    parser = ap.ArgumentParser(description="-- Yuchen PhD mission, how to plot a histogram from the result of code! --")
    parser.add_argument("file_path", help="The path of raw data (train data with labels).", type=str)
    parser.add_argument("--feature", help="The feature.", type=str, default="intensity")
    args = parser.parse_args()

    # take arguments
    file_path = args.file_path
    feature = args.feature
    data_las = laspy.read(file_path)
    get_info(data_las)
    #print("> Dimension names of data: ", list(data_las.point_format.dimension_names))

    #plot_histogram(data_las, feature)
    '''
    #plot_raw_hist(data_las[np.where(data_las['number_of_returns']==1)], feature)
    plot_raw_hist(data_las[np.where(data_las['number_of_returns']==1)], feature)
    plot_raw_hist(data_las[np.where(data_las['number_of_returns']==2)], feature)
    plot_raw_hist(data_las[np.where(data_las['number_of_returns']==3)], feature)
    plot_raw_hist(data_las[np.where(data_las['number_of_returns']==4)], feature)
    plot_raw_hist(data_las[np.where(data_las['number_of_returns']==5)], feature)
    '''
    plot_raw_hist(data_las[np.where(data_las.Z>0.5)], feature)
    #data_las = data_las[np.where(data_las.Z>0.5)]

    print("###### program end ######")
