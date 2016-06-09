
class Report(object):
    raw_data_table = ""
    raw_geo_table = ""
    cluster_table = ""
    objects_count = ""
    objects_list = []
    clusters_count = ""
    spaces_names = []
    most_popular_cluster = ""
    least_popular_cluster = ""
    all_clusters = []
    trends_table = ""


class Cluster(object):
    id = ""
    name = ""
    percentage = ""

    stat_mid = ""
    stat_min = ""
    stat_mix = ""
