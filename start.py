from LingvoAnalysisModule import lingvo
from ClusteringModule import time_series
from ClusteringModule import geo_series
from ClassifierModule import classifier
from ClassifierModule import shm_classifier


def main():
    cross_val, series, class_result = shm_classifier.entry_point()
    clusters, models, indexes, scores = time_series.entry_point('data/time_series.csv')
    classes, non_classified_frame = classifier.entry_point('not_classified.csv', models, indexes)

    classifier.test_classifier()
    geo_clusters = geo_series.entry_point('revenue.csv')
    lingvo.entry_point(clusters, geo_clusters, classes, non_classified_frame, scores, cross_val, series, class_result)
    print("Success!")


if __name__ == "__main__":
    main()
