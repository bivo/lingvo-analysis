import sys
import getopt
from jinja2 import FileSystemLoader, Environment
from pandas import read_csv


# Some global vars
data_frame = None


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
        filename = args[0]
    except (getopt.error, IndexError):
        print("Enter CSV filename as argument!")
        sys.exit(2)

    read_file(filename)
    # TODO: analyse...

    count = len(data_frame.columns)

    render_result(objects_count=count)


def read_file(filename):
    global data_frame
    data_frame = read_csv(filename, sep=',', index_col=0)

    print("Результат: ")
    print(data_frame)


def render_result(**kwargs):
    loader = FileSystemLoader(".")
    env = Environment(loader=loader)
    template = env.get_template("template.html")
    rendered_template = template.render(**kwargs)

    with open("result.html", "w") as file:
        file.write(rendered_template)


if __name__ == "__main__":
    main()
