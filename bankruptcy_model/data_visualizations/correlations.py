import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import preprocessing

from bankruptcy_model.utils.data_loading import load_dataset_by_year
from bankruptcy_model.utils.data_prep import change_class_values_to_binary


def data_prep(data):
    data = data.dropna()
    data = data_normalization(data)
    return data


def data_normalization(data):
    normed = preprocessing.scale(data)
    normed = pd.DataFrame(normed)
    return normed


def remove_outliers(df, out_cols, T=1.5, verbose=True):
    new_df = df.copy()
    for c in out_cols[:-1]:
        q1 = new_df[c].quantile(.25)
        q3 = new_df[c].quantile(.75)
        col_iqr = q3 - q1
        col_max = q3 + T * col_iqr
        col_min = q1 - T * col_iqr
        filtered_df = new_df[(new_df[c] <= col_max) & (new_df[c] >= col_min)]
        if verbose:
            n_out = new_df.shape[0] - filtered_df.shape[0]
            print(f" Columns {c} had {n_out} outliers removed")
        new_df = filtered_df
    return new_df


def plot_corrmatrix(year):
    data = load_dataset_by_year(year)
    data = change_class_values_to_binary(data)
    corr_matrix = data.corr()
    plt.figure(figsize=(20, 20))
    sn.heatmap(corr_matrix, cmap='coolwarm')
    plt.show()


def get_subset(data, bankrupt):
    subset = data[data['bankrupt'] == bankrupt]
    return subset


def get_table_with_mean_and_sd(data):
    data = remove_outliers(data, data.iloc[:, :-1], verbose=False)
    data.loc[data['class'].astype(str).str.contains("0"), 'bankrupt'] = 0
    data.loc[data['class'].astype(str).str.contains("1"), 'bankrupt'] = 1
    bankrupt = get_subset(data, 1)
    non_bankrupt = get_subset(data, 0)
    table = pd.DataFrame()
    table['means_bankrupt'] = bankrupt.mean()
    table['sd_bankrupt'] = np.std(bankrupt)
    table['means_non_bankrupt'] = non_bankrupt.mean()
    table['sd_non_bankrupt'] = np.std(non_bankrupt)
    table = table[:-2]
    return table
