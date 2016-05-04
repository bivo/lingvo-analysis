import sys
import getopt
from AnalysisModule import analyse


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
        filename = args[0]
    except (getopt.error, IndexError):
        print("Enter CSV filename as argument!")
        sys.exit(2)
    analyse.entry_point(filename)


if __name__ == "__main__":
    main()
