import sys
import getopt
from LingvoAnalysisModule import lingvo
from ClusteringModule import time_series


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
        filename = args[0]
    except (getopt.error, IndexError):
        print("Enter CSV filename as argument!")
        sys.exit(2)

    clusters = time_series.entry_point('data/time_series.csv')
    lingvo.entry_point(filename, clusters)
    print("Success!")


if __name__ == "__main__":
    main()
