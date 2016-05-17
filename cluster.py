import sys
import getopt
from ClusteringModule import time_series

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
        filename = args[0]
    except (getopt.error, IndexError):
        print("Enter CSV filename as argument!")
        sys.exit(2)
    time_series.entry_point(filename)
    print("Success!")

if __name__ == "__main__":
    main()
