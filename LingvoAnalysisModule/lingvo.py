from jinja2 import FileSystemLoader, Environment
from pandas import read_csv
from collections import Counter
import scipy
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from LingvoAnalysisModule.objects import Report, Cluster, GeoCluster

# Some global vars
data_frame = None
geo_frame = None
time_series_clusters = None
css_classes = ["table", "table-striped", "table-bordered", "table-hover", "table-condensed"]


def entry_point(clusters, geo_clusters, classes, non_classified):
    read_files('clustered_data.csv', 'revenue.csv')
    result = start_analysis(clusters, geo_clusters, classes, non_classified)
    render_result(**result.__dict__)


def start_analysis(clusters, geo_clusters, classes, non_classified):
    all_clusters_list = clusters.values.flatten().tolist()
    all_clusters_set = set(all_clusters_list)

    geo_clusters_list = geo_clusters.values.flatten().tolist()
    geo_clusters_set = set(geo_clusters_list)

    report = Report()
    report.raw_data_table = data_frame.to_html(classes=css_classes)
    report.raw_geo_table = read_geo().to_html(classes=css_classes)
    report.objects_count = len(data_frame.columns)
    report.geo_objects_count = len(geo_frame.ix[:, 0])
    report.objects_list = data_frame.columns
    report.geo_objects_list = geo_frame.ix[:, 0].keys()
    report.spaces_names = set(data_frame.loc["Пространство"])
    report.clusters_count = len(all_clusters_set)
    report.geo_clusters_count = len(geo_clusters_set)
    report.most_popular_cluster = count_clusters_statistics(all_clusters_list)[0]
    report.geo_most_popular_cluster = count_clusters_statistics(geo_clusters_list)[0]
    report.least_popular_cluster = count_clusters_statistics(all_clusters_list)[1]
    report.geo_least_popular_cluster = count_clusters_statistics(geo_clusters_list)[1]

    cluster_elements = {0: [], 1: [], 2: [], 3: []}
    for i in clusters.iterrows():
        cluster_elements.get(i[1][0]).append(i[0])

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
        values_stat[i] = (int(max(series)), int(scipy.mean(series)), int(min(series)))
        i += 1

    report.all_clusters = []
    for cluster_id in all_clusters_set:
        cluster_freq = all_clusters_list.count(cluster_id)
        percentage = cluster_freq * (len(all_clusters_list) + 2)

        all_max, all_mid, all_min = [], [], []
        for index, cl in enumerate(clusters.values):
            if clusters.values[index][0] == cluster_id:
                all_max.append(values_stat[index][0])
                all_mid.append(values_stat[index][1])
                all_min.append(values_stat[index][2])

        cluster = Cluster()
        cluster.id = str(cluster_id)
        cluster.percentage = percentage
        cluster.objects = cluster_elements[cluster_id]
        cluster.stat_max = int(max(all_max))
        cluster.stat_mid = int(scipy.mean(all_mid))
        cluster.stat_min = int(min(all_min))
        cluster.lingvo_result = fuzzy_lingvo_scale(cluster_elements[cluster_id], cluster_id)

        report.all_clusters.append(cluster)

    geo_cluster_elements = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for i in geo_clusters.iterrows():
        geo_cluster_elements.get(i[1][0]).append(i[0])

    report.geo_clusters = []
    for cluster_id in geo_clusters_set:
        cluster_freq = geo_clusters_list.count(cluster_id)
        percentage = round(cluster_freq * 100 / len(geo_clusters_list))

        cluster = GeoCluster()
        cluster.id = str(cluster_id + 1)
        cluster.percentage = percentage
        cluster.objects = geo_cluster_elements[cluster_id]

        cluster.big_data_percent, cluster.lingvo_style, cluster.lingvo_result, cluster.graph, cluster.average_profit, cluster.dominating_segment = \
            geo_fuzzy_lingvo_scale(geo_cluster_elements[cluster_id], cluster_id)

        report.geo_clusters.append(cluster)

    return report


def count_clusters_statistics(all_clusters_list):
    stat = Counter(all_clusters_list).most_common(None)
    most_popular = []
    least_popular = []

    tuple0, tuple1 = stat[:2]
    if tuple0[1] == tuple1[1]:
        most_popular.append(tuple0[0])
        most_popular.append(tuple1[0])
    else:
        most_popular.append(tuple0[0])

    tuple2, tuple3 = stat[-2:]
    if tuple2[1] == tuple3[1]:
        least_popular.append(tuple2[0])
        least_popular.append(tuple3[0])
    else:
        least_popular.append(tuple3[0])

    return most_popular, least_popular


def fuzzy_lingvo_scale(objects, cluster_id):
    mean_growth_times = 0

    is_first_iteration = True
    for column, series in data_frame.iteritems():
        if column not in objects:
            continue

        series = series.drop("Пространство")
        start_value = int(series[0])
        end_value = int(series[-1])

        growth_times = end_value if start_value == 0 else float(end_value / start_value)
        mean_growth_times = growth_times if is_first_iteration else (mean_growth_times + growth_times) / 2
        is_first_iteration = False

    # Generate universe variables
    x_tendency = np.arange(0, 40, 1)

    # Generate fuzzy membership functions
    mfn_high_fall = fuzz.zmf(x_tendency, 0, 0.5)
    mfn_fall = fuzz.trimf(x_tendency, [0.5, 1, 1])
    mfn_flat = fuzz.trimf(x_tendency, [1, 2, 2])
    mfn_growth = fuzz.trimf(x_tendency, [2, 10, 10])
    mfn_high_growth = fuzz.trimf(x_tendency, [10, 40, 40])

    # Visualize these universes and membership functions
    fig, ax = plt.subplots(nrows=1, figsize=(8, 3))

    ax.plot(x_tendency, mfn_high_fall, "r", linewidth=2.5, label="High fall")
    ax.plot(x_tendency, mfn_fall, "r", linewidth=1, label="Fall")
    ax.plot(x_tendency, mfn_flat, "grey", linewidth=1, label="Flat")
    ax.plot(x_tendency, mfn_growth, "g", linewidth=1, label="Growth")
    ax.plot(x_tendency, mfn_high_growth, "g", linewidth=2.5, label="High growth")
    ax.set_title("Fuzzy scale of trend")
    ax.legend()

    # Turn off top/right axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.tight_layout()
    fig.savefig("output/fuzzy_scale.png")

    # Interpretation
    interp_high_fall = fuzz.interp_membership(x_tendency, mfn_high_fall, mean_growth_times)
    interp_fall = fuzz.interp_membership(x_tendency, mfn_fall, mean_growth_times)
    interp_flat = fuzz.interp_membership(x_tendency, mfn_flat, mean_growth_times)
    interp_growth = fuzz.interp_membership(x_tendency, mfn_growth, mean_growth_times)
    interp_high_growth = fuzz.interp_membership(x_tendency, mfn_high_growth, mean_growth_times)

    activation_high_fall = np.fmin(interp_high_fall, mfn_high_fall)
    activation_fall = np.fmin(interp_fall, mfn_fall)
    activation_flat = np.fmin(interp_flat, mfn_flat)
    activation_growth = np.fmin(interp_growth, mfn_growth)
    activation_high_growth = np.fmin(interp_high_growth, mfn_high_growth)

    zeroes = np.zeros_like(x_tendency)

    # Visualize this
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_tendency, zeroes, activation_high_fall, facecolor="r", alpha=0.7)
    ax0.plot(x_tendency, mfn_high_fall, "r", linewidth=2, linestyle="--", )

    ax0.fill_between(x_tendency, zeroes, activation_fall, facecolor="r", alpha=0.7)
    ax0.plot(x_tendency, mfn_fall, "r", linewidth=0.5, linestyle="--", )

    ax0.fill_between(x_tendency, zeroes, activation_flat, facecolor="grey", alpha=0.7)
    ax0.plot(x_tendency, mfn_flat, "grey", linewidth=0.5, linestyle="--")

    ax0.fill_between(x_tendency, zeroes, activation_growth, facecolor="g", alpha=0.7)
    ax0.plot(x_tendency, mfn_growth, "g", linewidth=0.5, linestyle="--")

    ax0.fill_between(x_tendency, zeroes, activation_high_growth, facecolor="g", alpha=0.7)
    ax0.plot(x_tendency, mfn_high_growth, "g", linewidth=2, linestyle="--")

    ax0.set_title("Cluster: " + str(cluster_id) + " | Objects: " + str(objects))

    # Turn off top/right axes
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.get_xaxis().tick_bottom()
    ax0.get_yaxis().tick_left()

    plt.tight_layout()
    img = "fuzzy_" + str(cluster_id) + ".png"
    fig.savefig("output/" + img)

    # Aggregate all output membership functions together
    aggregated = np.fmax(activation_flat,
                         np.fmax(activation_growth,
                                 np.fmax(activation_fall,
                                         np.fmax(activation_high_growth, activation_high_fall))))

    # Calculate defuzzified result
    tendency = fuzz.defuzz(x_tendency, aggregated, 'centroid')
    # tendency_activation = fuzz.interp_membership(x_tendency, aggregated, tendency)  # for plot

    data_for_ui = None, None, None
    if tendency <= 0.5:
        data_for_ui = "danger", "glyphicon-arrow-down", "High fall"
    elif 0 < tendency < 1:
        data_for_ui = "danger", "glyphicon-arrow-down", "Fall"
    elif 1 <= tendency < 2:
        data_for_ui = "default", "glyphicon-minus", "Flat"
    elif 2 <= tendency < 10:
        data_for_ui = "success", "glyphicon-arrow-up", "Growth"
    elif 10 <= tendency <= 40:
        data_for_ui = "success", "glyphicon-arrow-up", "High growth"

    return str("%.2f" % tendency), data_for_ui[0], data_for_ui[1], data_for_ui[2], img


def geo_fuzzy_lingvo_scale(objects, cluster_id):
    percentage = 0
    average_profit = 0
    profit_by_type = {geo_frame.columns[2]: 0, geo_frame.columns[3]: 0, geo_frame.columns[4]: 0}

    for row, series in geo_frame.iterrows():
        if row not in objects:
            continue
        average_profit += int(series[0])
        percentage += int(series[1])
        profit_by_type[geo_frame.columns[2]] += int(series[2])
        profit_by_type[geo_frame.columns[3]] += int(series[3])
        profit_by_type[geo_frame.columns[4]] += int(series[4])

    for key in profit_by_type.keys():
        profit_by_type[key] /= len(objects)

    dominating_type = get_dominating_type(profit_by_type)

    average_profit = round(average_profit / len(objects))
    percentage = round(percentage / len(objects), 2)
    # Generate universe variables
    x_tendency = np.arange(0, 100, 1)

    # Generate fuzzy membership functions
    mfn_insignificant = fuzz.zmf(x_tendency, 0, 5)
    mfn_low = fuzz.trimf(x_tendency, [5, 30, 30])
    mfn_middle = fuzz.trimf(x_tendency, [30, 65, 65])
    mfn_main = fuzz.trimf(x_tendency, [65, 100, 100])

    # Visualize these universes and membership functions
    fig, ax = plt.subplots(nrows=1, figsize=(8, 3))

    ax.plot(x_tendency, mfn_insignificant, "r", linewidth=2.5, label="незначительная")
    ax.plot(x_tendency, mfn_low, "r", linewidth=1, label="малая")
    ax.plot(x_tendency, mfn_middle, "grey", linewidth=1, label="средняя")
    ax.plot(x_tendency, mfn_main, "g", linewidth=1, label="основная")
    ax.set_title("Доля выручки сегмента Big Data")
    ax.legend()

    # Turn off top/right axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.tight_layout()
    fig.savefig("output/geo_fuzzy_scale.png")

    # Interpretation
    interp_insignificant = fuzz.interp_membership(x_tendency, mfn_insignificant, percentage)
    interp_low = fuzz.interp_membership(x_tendency, mfn_low, percentage)
    interp_middle = fuzz.interp_membership(x_tendency, mfn_middle, percentage)
    interp_main = fuzz.interp_membership(x_tendency, mfn_main, percentage)

    activation_insignificant = np.fmin(interp_insignificant, mfn_insignificant)
    activation_low = np.fmin(interp_low, mfn_low)
    activation_middle = np.fmin(interp_middle, mfn_middle)
    activation_main = np.fmin(interp_main, mfn_main)

    zeroes = np.zeros_like(x_tendency)

    # Visualize this
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_tendency, zeroes, activation_insignificant, facecolor="r", alpha=0.7)
    ax0.plot(x_tendency, mfn_insignificant, "r", linewidth=2, linestyle="--", )

    ax0.fill_between(x_tendency, zeroes, activation_low, facecolor="r", alpha=0.7)
    ax0.plot(x_tendency, mfn_low, "r", linewidth=0.5, linestyle="--", )

    ax0.fill_between(x_tendency, zeroes, activation_middle, facecolor="grey", alpha=0.7)
    ax0.plot(x_tendency, mfn_middle, "grey", linewidth=0.5, linestyle="--")

    ax0.fill_between(x_tendency, zeroes, activation_main, facecolor="g", alpha=0.7)
    ax0.plot(x_tendency, mfn_main, "g", linewidth=0.5, linestyle="--")

    ax0.set_title("Cluster: " + str(cluster_id))

    # Turn off top/right axes
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.get_xaxis().tick_bottom()
    ax0.get_yaxis().tick_left()

    img = "geo_fuzzy_" + str(cluster_id) + ".png"
    fig.savefig("output/" + img)

    # Aggregate all output membership functions together
    aggregated = np.fmax(activation_middle,
                         np.fmax(activation_main,
                                 np.fmax(activation_low, activation_insignificant)))

    # Calculate defuzzified result
    tendency = fuzz.defuzz(x_tendency, aggregated, 'centroid')
    # tendency_activation = fuzz.interp_membership(x_tendency, aggregated, tendency)  # for plot

    data_for_ui = None, None, None
    if tendency <= 5:
        data_for_ui = "danger", "glyphicon-arrow-down", "Незначительная"
    elif 5 < tendency < 30:
        data_for_ui = "danger", "glyphicon-arrow-down", "Малая"
    elif 30 <= tendency < 65:
        data_for_ui = "default", "glyphicon-minus", "Средняя"
    elif 65 <= tendency < 100:
        data_for_ui = "success", "glyphicon-arrow-up", "Основная"

    return str("%.2f" % tendency), data_for_ui[0], data_for_ui[2], img, average_profit, dominating_type


def read_files(time_filename, geo_filename):
    global data_frame
    data_frame = read_csv(time_filename, sep=",", index_col=0)

    global geo_frame
    geo_frame = read_csv(geo_filename, sep=",", index_col=0)


def get_dominating_type(types):
    best_key = ""
    best_val = 0
    for tpl in types.items():
        if tpl[1] > best_val:
            best_key = tpl[0]
            best_val = tpl[1]

    result_keys = [best_key]
    for tpl in types.items():
        if tpl[0] != best_key and tpl[1]/best_val > 0.75:
            result_keys.append(tpl[0])

    if len(result_keys) == 3:
        return "Отсутствует"
    else:
        return best_key


def read_geo():
    return read_csv("data/revenue.csv", sep=",", index_col=0)


def render_result(**kwargs):
    loader = FileSystemLoader(".")
    env = Environment(loader=loader)
    template = env.get_template("template.html")
    rendered_template = template.render(**kwargs)
    with open("output/index.html", "w") as file:
        file.write(rendered_template)
