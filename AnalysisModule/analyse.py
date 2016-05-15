from jinja2 import FileSystemLoader, Environment
from pandas import read_csv
from collections import Counter

from AnalysisModule.objects import Report, Cluster

# Some global vars
data_frame = None


def entry_point(filename):
    read_file(filename)
    result = start_analysis()
    render_result(**result.__dict__)


def start_analysis():
    all_clusters = data_frame.loc["Кластер"]
    all_clusters_list = all_clusters.values.tolist()
    unique_clusters = set(all_clusters)
    cluster_statistics = Counter(all_clusters)

    report = Report()
    report.html_table = data_frame.to_html(classes=["table", "table-striped", "table-bordered", "table-hover", "table-condensed"])
    report.objects_count = len(data_frame.columns)
    report.objects_list = data_frame.columns
    report.clusters_count = len(unique_clusters)
    report.cluster_names = unique_clusters
    report.most_popular_cluster = cluster_statistics.most_common(1)[0][0]
    report.least_popular_cluster = cluster_statistics.most_common()[:-1-1:-1][0][0]

    report.all_clusters = []
    for clusterId in unique_clusters:
        # print("Cluster " + str(clusterId))

        cluster_freq = all_clusters_list.count(clusterId)
        percentage = (cluster_freq / 100) * len(all_clusters_list)

        cluster = Cluster()
        cluster.cluster_id = str(clusterId)
        cluster.cluster_name = "Кластер " + str(clusterId)
        cluster.cluster_percentage = percentage

        report.all_clusters.append(cluster)

    return report


def read_file(filename):
    global data_frame
    data_frame = read_csv(filename, sep=',', index_col=0)
    # print("Результат:")
    # print(data_frame)


def render_result(**kwargs):
    loader = FileSystemLoader(".")
    env = Environment(loader=loader)
    template = env.get_template("template.html")
    rendered_template = template.render(**kwargs)
    with open("result.html", "w") as file:
        file.write(rendered_template)
