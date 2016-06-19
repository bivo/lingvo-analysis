class Report(object):
    raw_data_table = ""
    raw_geo_table = ""
    objects_count = ""
    geo_objects_count = ""
    objects_list = []
    geo_objects_list = []
    clusters_count = ""
    geo_clusters_count = ""
    spaces_names = []
    most_popular_cluster = ""
    geo_most_popular_cluster = ""
    least_popular_cluster = ""
    geo_least_popular_cluster = ""
    all_clusters = []
    geo_clusters = []


class Cluster(object):
    id = ""
    name = ""
    percentage = ""
    objects = []
    lingvo_result = ""

    stat_mid = ""
    stat_min = ""
    stat_mix = ""


class GeoCluster(object):
    id = ""
    percentage = ""
    objects = []
    big_data_percent = ""
    lingvo_style = ""
    lingvo_result = ""
    graph = ""

    average_profit = ""
    dominating_segment = ""
