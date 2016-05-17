import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from pprint import pprint
from jinja2 import FileSystemLoader, Environment


def entry_point(filename):
    series = read_file(filename)
    models = build_models(series)
    indexes = clusterize(models)

    result = pd.DataFrame(data=indexes, index=series.columns, dtype=object)

    return result

def build_models(series):
    models = []
    fig, ax = plt.subplots()
    for index, row in series.iteritems():
        arma_mod = sm.tsa.ARMA(row, (3,0))
        arma_res = arma_mod.fit(trend='nc', disp=-1, transparams=True)
        ax = row.ix['2004':].plot(ax=ax)
        models = np.append(models, arma_res.params.values)

    ax.legend(loc="best");
    # plt.show()
    models = np.reshape(models, [9, 3])

    return models

def clusterize(models):
    clusters = KMeans(n_clusters=4).fit_predict(models)
    return clusters

def read_file(filename):
    time_series_frame = pd.read_csv(filename, sep=',', index_col=0, parse_dates=True)

    return time_series_frame