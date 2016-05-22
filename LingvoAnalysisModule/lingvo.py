from jinja2 import FileSystemLoader, Environment
from pandas import read_csv
from collections import Counter

from LingvoAnalysisModule.objects import Report, Cluster

# Some global vars
data_frame = None
time_series_clusters = None
css_classes = ["table", "table-striped", "table-bordered", "table-hover", "table-condensed"]


def entry_point(filename, clusters):
    read_file(filename)
    result = start_analysis(clusters)
    render_result(**result.__dict__)


def start_analysis(clusters):
    all_clusters_list = clusters.values.flatten().tolist()
    all_clusters_set = set(all_clusters_list)
    clusters_statistics = Counter(all_clusters_list)

    report = Report()
    report.raw_data_table = data_frame.to_html(classes=css_classes)
    report.cluster_table = clusters.to_html(classes=css_classes, header=False)
    report.objects_count = len(data_frame.columns)
    report.objects_list = data_frame.columns
    report.spaces_names = set(data_frame.loc["Пространство"])
    report.clusters_count = len(all_clusters_set)
    report.most_popular_cluster = clusters_statistics.most_common(1)[0][0]
    report.least_popular_cluster = clusters_statistics.most_common()[:-1-1:-1][0][0]

    report.all_clusters = []
    for clusterId in all_clusters_set:
        cluster_freq = all_clusters_list.count(clusterId)
        percentage = cluster_freq * (len(all_clusters_list) + 2)

        cluster = Cluster()
        cluster.id = str(clusterId)
        cluster.percentage = percentage
        cluster.stat_mid = 5
        cluster.stat_min = 1
        cluster.stat_max = 10

        report.all_clusters.append(cluster)

    return report


def read_file(filename):
    global data_frame
    data_frame = read_csv(filename, sep=',', index_col=0)


def render_result(**kwargs):
    loader = FileSystemLoader(".")
    env = Environment(loader=loader)
    template = env.get_template("template.html")
    rendered_template = template.render(**kwargs)
    with open("output/report.html", "w") as file:
        file.write(rendered_template)
