import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, datasets


def entry_point(filename, train_data, train_targets):
    series = read_file(filename)
    models = build_models(series)

    classifier = get_default_classifier()
    classifier.fit(train_data, train_targets)
    classes = classify(classifier, models)

    result = pd.DataFrame(data=classes, index=series.columns, dtype=object)

    return result, series


def get_default_classifier():
    return RandomForestClassifier(n_estimators=10)


def build_models(series):
    models = []
    fig, ax = plt.subplots()
    for index, row in series.iteritems():
        arma_mod = sm.tsa.ARMA(row, (3, 0))
        arma_res = arma_mod.fit(trend='nc', disp=-1, transparams=True)
        ax = row.ix['2009':].plot(ax=ax)
        models = np.append(models, arma_res.params.values)

    ax.legend(loc="best")
    models = np.reshape(models, [5, 3])
    fig.savefig('output/non_classified.png')

    return models


def classify(classifier, models):
    prediction = classifier.predict(models)
    proba = classifier.predict_proba(models)

    result = list(zip(prediction, proba))
    return result


def test_classifier():
    prepare_seed = 123123
    fig, ax = plt.subplots()

    sizes = []
    vals = []

    for i in range(1, 10):
        size = math.ceil(math.exp(i / 2 + 3))
        sizes.append(size)
        data, labels = prepare_random_data(prepare_seed, size)
        classifier = get_default_classifier()

        cross_val = cross_validation.cross_val_score(classifier, data, labels, cv=10)

        vals.append(cross_val.mean())
        pass

    ax.set_xlabel('Sample size')
    ax.set_ylabel('Mean accuracy')
    ax.plot(sizes, vals)
    fig.savefig('output/cross_validation.png')

    # print(cross_val)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (cross_val.mean(), cross_val.std() * 2))


def prepare_random_data(seed, size):
    data, labels = datasets.make_classification(n_samples=size, n_features=3, n_redundant=0, n_informative=3,
                                                n_classes=4, random_state=seed, n_clusters_per_class=1)
    return data, labels


def read_file(filename):
    time_series_frame = pd.read_csv(filename, sep=',', index_col=0, parse_dates=True)
    return time_series_frame
