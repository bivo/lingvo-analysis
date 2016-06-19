import pandas as pd
from sklearn.cluster import KMeans


def entry_point(filename):
    series = read_file(filename)
    models = build_models(series)
    indexes = clusterize(models)

    result = pd.DataFrame(data=indexes, index=series.ix[:, 0].index, dtype=object)

    return result


def build_models(series):
    models = []
    for _, row in series.iterrows():
        models.append(row.values)

    return models


def clusterize(models):
    clusters = KMeans(n_clusters=6).fit_predict(models)
    return clusters


def read_file(filename):
    geo_series_frame = pd.read_csv(filename, sep=',', index_col=0)
    return geo_series_frame
