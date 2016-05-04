from jinja2 import FileSystemLoader, Environment
from pandas import read_csv

from AnalysisModule.objects import Report, Cluster

# Some global vars
data_frame = None


def entry_point(filename):
    read_file(filename)
    result = start_analysis()
    render_result(**result.__dict__)


def start_analysis():
    report = Report()

    report.objects_count = len(data_frame.columns)
    unique_clusters = set(data_frame.loc["Кластер"])
    report.clusters_count = len(unique_clusters)
    report.cluster_names = unique_clusters
    report.most_popular_cluster = "NaN"
    report.least_popular_cluster = "NaN"

    report.all_clusters = []
    for cluster in unique_clusters:
        cluster_obj = Cluster()
        cluster_obj.cluster_id = str(cluster)
        cluster_obj.cluster_name = "Кластер " + str(cluster)
        cluster_obj.cluster_percentage = "NaN"
        report.all_clusters.append(cluster_obj)

    return report


def read_file(filename):
    global data_frame
    data_frame = read_csv(filename, sep=',', index_col=0)
    print("Результат:")
    print(data_frame)


def render_result(**kwargs):
    loader = FileSystemLoader(".")
    env = Environment(loader=loader)
    template = env.get_template("template.html")
    rendered_template = template.render(**kwargs)
    with open("result.html", "w") as file:
        file.write(rendered_template)
