import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def entry_point(filename, train_data, train_targets):
    series = read_file(filename)
    models = build_models(series)

    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(train_data, train_targets)
    classes = classify(classifier, models)

    result = pd.DataFrame(data=classes, index=series.columns, dtype=object)

    return result, series


def build_models(series):
    models = []
    fig, ax = plt.subplots()
    for index, row in series.iteritems():
        arma_mod = sm.tsa.ARMA(row, (3,0))
        arma_res = arma_mod.fit(trend='nc', disp=-1, transparams=True)
        ax = row.ix['2009':].plot(ax=ax)
        models = np.append(models, arma_res.params.values)

    ax.legend(loc="best")
    models = np.reshape(models, [5, 3])
    fig.savefig('output/non_classified.png')

    return models

def classify(classifier, models):
    prediction = classifier.predict(models)
    log = classifier.predict_log_proba(models)
    proba = classifier.predict_proba(models)

    result = list(zip(prediction, log, proba))
    return prediction

def read_file(filename):
    time_series_frame = pd.read_csv(filename, sep=',', index_col=0, parse_dates=True)
    return time_series_frame
