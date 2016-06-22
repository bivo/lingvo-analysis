from jinja2 import FileSystemLoader, Environment
from pandas import read_csv
from collections import Counter
import scipy

from LingvoAnalysisModule.objects import Report, Cluster

# Some global vars
data_frame = None
time_series_clusters = None
css_classes = ["table", "table-striped", "table-bordered", "table-hover", "table-condensed"]


def entry_point(filename, clusters, classes, non_classified, scores):
    read_file(filename)
    result = start_analysis(clusters, classes, non_classified,scores)
    render_result(**result.__dict__)


def start_analysis(clusters, classes, non_classified,scores):
    all_clusters_list = clusters.values.flatten().tolist()
    all_clusters_set = set(all_clusters_list)
    clusters_statistics = Counter(all_clusters_list)

    report = Report()
    report.raw_data_table = data_frame.to_html(classes=css_classes)
    report.cluster_table = clusters.to_html(classes=css_classes, header=False)
    report.cluster_scores = scores.to_html(classes=css_classes, header=False)
    report.objects_count = len(data_frame.columns)
    report.objects_list = data_frame.columns
    report.spaces_names = set(data_frame.loc["Пространство"])
    report.clusters_count = len(all_clusters_set)
    report.most_popular_cluster = clusters_statistics.most_common(1)[0][0]
    report.least_popular_cluster = clusters_statistics.most_common()[:-1-1:-1][0][0]

    # Non classified
    report.non_classified_objects_list = non_classified.columns
    report.non_classified_objects_count = len(non_classified.columns)
    report.non_classified_classes = classes.values.flatten().tolist()
    report.non_classified_objects = classes.to_html(classes=css_classes, header=False)

    values_stat = {}
    i = 0
    for column, series in data_frame.iteritems():
        series = series.drop("Пространство")
        series = [int(x) for x in series.tolist()]
        tendency = get_tendency(series)
        values_stat[i] = (int(max(series)), int(scipy.mean(series)), int(min(series)), tendency)
        i += 1

    report.all_clusters = []
    for clusterId in all_clusters_set:
        cluster_freq = all_clusters_list.count(clusterId)
        percentage = cluster_freq * (len(all_clusters_list) + 2)

        all_max, all_mid, all_min, tendencies = [], [], [], []
        for index, cl in enumerate(clusters.values):
            if clusters.values[index][0] == clusterId:
                all_max.append(values_stat[index][0])
                all_mid.append(values_stat[index][1])
                all_min.append(values_stat[index][2])
                tendencies.append(values_stat[index][3])

        cluster = Cluster()
        cluster.id = str(clusterId)
        cluster.percentage = percentage
        cluster.stat_max = int(max(all_max))
        cluster.stat_mid = int(scipy.mean(all_mid))
        cluster.stat_min = int(min(all_min))
        tend_stats = Counter(tendencies)
        cluster.tendency_most = tend_stats.most_common(1)[0][0]
        cluster.tendency_most_count = tend_stats.most_common(1)[0][1]
        if (len(tend_stats.most_common()) > 1):
            cluster.least_tendency = True
            cluster.tendency_least = tend_stats.most_common()[:-1-1:-1][0][0]
            cluster.tendency_least_count = tend_stats.most_common()[:-1-1:-1][0][1]

        report.all_clusters.append(cluster)

    return report

def get_tendency(series):
    if abs(series[0] - series[-1]) > 10:
        if series[0] > series[-1]:
            return 'Падение'
        else:
            return 'Рост'
    else:
        return 'Стабильность'

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
