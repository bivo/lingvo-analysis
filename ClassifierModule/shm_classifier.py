import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


def entry_point():
    file_name = "data/classification/winequality-red.csv"
    series = pd.read_csv(file_name, sep=',', index_col=11)
    train_data = []
    train_targets = []
    for index, row in series.iterrows():
        train_data.append(row.values)
        train_targets.append(index)
    cross_val, predictions = test_classification(train_data, train_targets)
    series = pd.read_csv(file_name, sep=',')
    series["predict"] = pd.Series(predictions)
    class_result = find_classification_result(train_targets, predictions)
    return cross_val, series, class_result


def test_classification(train_data, train_targets):
    classifier = RandomForestClassifier()
    size = int(len(train_data)/2)
    classifier.fit(train_data[:size], train_targets[:size])
    predictions = classifier.predict(train_data)
    cross_val = cross_validation.cross_val_score(classifier, train_data, train_targets, cv=10)
    return int(round(cross_val.mean()*100)), predictions


def classify(classifier, models):
    prediction = classifier.predict(models)
    proba = classifier.predict_proba(models)

    result = list(zip(prediction, proba))
    return result


def find_classification_result(before, after):
    right = 0
    for i in range(0, len(before)):
        if before[i] == after[i]:
            right += 1
    return int(round(right*100/len(before)))
