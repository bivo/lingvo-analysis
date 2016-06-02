import sys
import getopt
from LingvoAnalysisModule import lingvo
from ClusteringModule import time_series
from ClassifierModule import classifier

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
        filename = args[0]
    except (getopt.error, IndexError):
        print("Enter CSV filename as argument!")
        sys.exit(2)

    clusters, models, indexes = time_series.entry_point('data/time_series.csv')
    classes, non_classified_frame = classifier.entry_point('not_classified.csv', models, indexes)
    lingvo.entry_point(filename, clusters, classes, non_classified_frame)
    print("Success!")


if __name__ == "__main__":
    main()
