import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics


def entry_point(filename):
    series = read_file(filename)
    models = build_models(series)
    indexes, scores = clusterize(models)

    result = pd.DataFrame(data=indexes, index=series.columns, dtype=object)

    return result, models, indexes, scores


def build_models(series):
    models = []
    fig, ax = plt.subplots()
    for index, row in series.iteritems():
        arma_mod = sm.tsa.ARMA(row, (3, 0))
        arma_res = arma_mod.fit(trend='nc', disp=-1, transparams=True, method='css')
        ax = row.ix['2004':].plot(ax=ax)
        models = np.append(models, arma_res.params.values)

    ax.legend(loc="best")
    # plt.show()
    models = np.reshape(models, [9, 3])
    print(models)
    fig.savefig('output/cluster_fig.png')

    return models


def clusterize(models):
    labels_true = [2, 0, 3, 2, 2, 0, 1, 2, 3]
    clusters = KMeans(n_clusters=4).fit_predict(models)
    rand = metrics.adjusted_rand_score(labels_true, clusters)
    mi = metrics.mutual_info_score(labels_true, clusters)
    homo = metrics.homogeneity_score(labels_true, clusters)
    completeness = metrics.completeness_score(labels_true, clusters)

    scores_list = [rand, mi, homo, completeness]
    scores_names = ['Приведенный индекс Ранда', 'Коэф. взаимной информации', 'Гомогенность', 'Полнота кластеров']
    scores = pd.DataFrame(data=scores_list, index=scores_names, dtype=object)
    return clusters, scores


def read_file(filename):
    time_series_frame = pd.read_csv(filename, sep=',', index_col=0, parse_dates=True)
    return time_series_frame
