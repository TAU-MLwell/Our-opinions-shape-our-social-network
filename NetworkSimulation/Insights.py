import collections
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import Strings as strs
import Plots
from numpy import linalg as LA

def get_norm(mat, p):
    return LA.norm(mat, p)


def count_not_zero_edges(weight_mat, zero_thresh):
    count = 0
    num_of_nodes = len(weight_mat)
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if weight_mat[i][j] > zero_thresh:
                count += 1
    return count


def calc_total_mono_weight(weights_mat, mono_edges_mat):
    mono_total_weights = np.sum(weights_mat, where = mono_edges_mat)
    return mono_total_weights


def threshold_num_of_nodes_with_mono_view(adj_mat, nodes_opinions, threshold):
    mono_count = 0
    mono_nodes = []
    colorful_count = 0
    colorful_nodes = []
    for i in range(len(adj_mat)):
        curr_node_color = nodes_opinions[i]
        in_edges = adj_mat[:, i].flatten()
        above_thresh_indexes = np.where(in_edges > threshold)[0]
        is_all_above_thresh_mono = all(curr_node_color == nodes_opinions[node] for node in above_thresh_indexes)
        is_all_above_thresh_colorful = all(curr_node_color != nodes_opinions[node] for node in above_thresh_indexes)
        if is_all_above_thresh_mono:
            mono_nodes.append(i)
            mono_count += 1
        if is_all_above_thresh_colorful:
            colorful_nodes.append(i)
            colorful_count += 1

    return mono_count


def calculate_edges_type_perc(mono_count, colorful_count, not_zero_edges):
    mono_per = mono_count / not_zero_edges
    colorful_per = colorful_count / not_zero_edges
    return mono_per, colorful_per


def num_of_nodes_with_mono_view(adj_mat, nodes_opinions):
    count = 0
    mono_view_nodes = []
    for idx, node in enumerate(nodes_opinions):
        node_color = nodes_opinions[idx]
        friends_ind = np.where(adj_mat[:, idx] > 0.0)[0]
        if all(node_color == nodes_opinions[friend] for friend in friends_ind):
            mono_view_nodes.append(idx)
            count += 1
    return count


def nodes_degree_histogram(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

    return degreeCount


def nodes_out_degree_histogram(adj_mat):
    degree_sequence = []
    for i in range(len(adj_mat)):
        degree_sequence.append(sum(k > 0 for k in adj_mat[i]))
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()


def calc_nodes_entropy_sorted(nodes_in_mono_sum):
    nodes_entropy = [0] * len(nodes_in_mono_sum)
    for i in range(len(nodes_in_mono_sum)):
        node_mono_wiehgt = nodes_in_mono_sum[i]
        nodes_entropy[i] = entropy([node_mono_wiehgt, 1 - node_mono_wiehgt], base=2)
    return nodes_entropy


def nodes_mono_in_wights_sorted(weights_mat, nodes_opinions):
    num_of_nodes = len(nodes_opinions)
    nodes_mono_in_weights = [0] * num_of_nodes
    nodes_colorful_weights = [0] * num_of_nodes
    for i in range(num_of_nodes):
        node_color = nodes_opinions[i]
        mono_weight = 0
        colorful_weight = 0
        for j in range(num_of_nodes):
            if i != j and node_color == nodes_opinions[j]:
                mono_weight += weights_mat[j][i]
            elif i != j and node_color != nodes_opinions[j]:
                colorful_weight += weights_mat[j][i]
        nodes_mono_in_weights[i] = mono_weight
        nodes_colorful_weights[i] = colorful_weight
    return nodes_mono_in_weights, nodes_colorful_weights


def nodes_out_sum(adj_mat):
    out_sums = adj_mat.sum(axis=1)
    sorted_out_sums = -np.sort(-out_sums)
    return sorted_out_sums


def out_sum_log_log(sorted_out_sums):
    log_out_sums = np.log(sorted_out_sums)
    return log_out_sums


def get_edges_count_and_colors(weight_mat, nodes_opinions, zero_thresh):
    count_mono_edges = 0
    count_colorful_edges = 0
    mono_edges_thresh_mat = np.zeros_like(weight_mat)
    colorful_edges_thresh_mat = np.zeros_like(weight_mat)
    for i in range(len(weight_mat)):
        for j in  range(len(weight_mat)):
            if  weight_mat[i][j] > zero_thresh:
                if nodes_opinions[i] == nodes_opinions[j]:
                    count_mono_edges += 1
                    mono_edges_thresh_mat[i][j] = 1
                else:
                    count_colorful_edges += 1
                    colorful_edges_thresh_mat[i][j] = 1
    return count_mono_edges, count_colorful_edges, mono_edges_thresh_mat, colorful_edges_thresh_mat


def nodes_degree_sorted(G):
    return sorted([d for n, d in G.degree()], reverse=True)


def nodes_degree_with_zero_thresh(weight_mat, zero_thresh):
    num_of_nodes = len(weight_mat)
    nodes_degrees = []
    for i in range(num_of_nodes):
        node_out_weights = weight_mat[i]
        nodes_degrees.append(len(np.where(node_out_weights > zero_thresh)[0]))
    return sorted(nodes_degrees, reverse=True)

def local_disagreement(node, exp_opinions, weights_mat):
    num_of_nodes = len(weights_mat)
    node_exp_opinion = exp_opinions[node]
    node_local_disagreement = 0
    for i in range(num_of_nodes):
        if i == node:
            continue
        node_local_disagreement += weights_mat[i][node]\
                                   * np.square(node_exp_opinion - exp_opinions[i])
    return node_local_disagreement

def global_disagreement(exp_opinions, weights_mat):
    num_of_nodes = len(weights_mat)
    global_disagr = 0
    for i in range(num_of_nodes):
        global_disagr += local_disagreement(i, exp_opinions, weights_mat)
    return global_disagr

def nodes_feed_mean_and_distance_from_feed_mean(nodes_cont_opinions, weights_mat):
    num_of_nodes = len(weights_mat)
    distance_from_feed_mean = []
    nodes_feed_mean = []
    for node in range(num_of_nodes):
        curr_nodes_distance_from_feed_mean = 0
        node_cont_opinion = nodes_cont_opinions[node]
        for i in range(num_of_nodes):
            curr_nodes_distance_from_feed_mean += weights_mat[i][node] * nodes_cont_opinions[i]
        distance_from_feed_mean.append(node_cont_opinion - curr_nodes_distance_from_feed_mean)
        nodes_feed_mean.append(curr_nodes_distance_from_feed_mean)
    return nodes_feed_mean, distance_from_feed_mean

def get_measurements(weights_mat, adj_mat, nodes_opinions, insights_dict):
    num_of_nodes = len(nodes_opinions)
    zero_thresh = 1e-4
    norm = 2
    not_zero_edges_count = 0
    thresh_edges_count = 0
    total_mono_count = 0
    thresh_total_mono_count = 0

    thresh_total_weight = 0
    total_mono_weight = 0
    thresh_total_mono_weight = 0

    nodes_total_out_sum = weights_mat.sum(axis=1)
    thresh_nodes_total_out_sum = np.zeros(num_of_nodes)
    thresh_nodes_total_in_sum = np.zeros(num_of_nodes)
    nodes_out_mono_sum = np.zeros(num_of_nodes)
    thresh_nodes_out_mono_sum = np.zeros(num_of_nodes)
    nodes_in_mono_sum = np.zeros(num_of_nodes)
    thresh_nodes_in_mono_sum = np.zeros(num_of_nodes)

    nodes_with_mono_view_count = 0
    thresh_nodes_with_mono_view_count = 0

    nodes_local_disagreement = np.zeros(num_of_nodes)

    for i in range(num_of_nodes):
        nodes_local_disagreement[i] = local_disagreement(i, nodes_opinions, weights_mat)

        curr_node_color = nodes_opinions[i]
        curr_node_in_edges = adj_mat[:, i].flatten()
        in_edges_indexes = np.where(curr_node_in_edges > 0)[0]
        thresh_in_edges_indexes = np.where(curr_node_in_edges > zero_thresh)[0]
        is_all_view_mono = all(curr_node_color == nodes_opinions[node] for node in in_edges_indexes)
        if is_all_view_mono:
            nodes_with_mono_view_count += 1
            thresh_nodes_with_mono_view_count += 1
        elif all(curr_node_color == nodes_opinions[node] for node in thresh_in_edges_indexes):
            thresh_nodes_with_mono_view_count += 1

        for j in range(num_of_nodes):
            edge_weight = weights_mat[i][j]
            if edge_weight > 0:
                not_zero_edges_count += 1

                if edge_weight > zero_thresh:
                    thresh_total_weight += edge_weight
                    thresh_edges_count += 1
                    thresh_nodes_total_out_sum[i] += edge_weight
                    thresh_nodes_total_in_sum[j] += edge_weight
                    if nodes_opinions[i] == nodes_opinions[j]:
                        thresh_total_mono_count += 1
                        thresh_total_mono_weight += edge_weight
                        thresh_nodes_out_mono_sum[i] += edge_weight
                        thresh_nodes_in_mono_sum[j] += edge_weight

                if nodes_opinions[i] == nodes_opinions[j]:
                    total_mono_count += 1
                    total_mono_weight += edge_weight
                    nodes_out_mono_sum[i] += edge_weight
                    nodes_in_mono_sum[j] += edge_weight


    total_colorful_count = not_zero_edges_count - total_mono_count
    thresh_total_colorful_count = thresh_edges_count - thresh_total_mono_count
    total_colorful_weight = num_of_nodes - total_mono_weight
    thresh_total_colorful_weight = thresh_total_weight - thresh_total_mono_weight
    nodes_out_colorful_sum = nodes_total_out_sum - nodes_out_mono_sum
    thresh_nodes_out_colorful_sum = np.full(num_of_nodes, thresh_total_weight) - thresh_nodes_out_mono_sum
    nodes_in_colorful_sum = np.full(num_of_nodes, 1) - nodes_in_mono_sum
    thresh_nodes_in_colorful_sum = thresh_nodes_total_in_sum - thresh_nodes_in_mono_sum
    nodes_with_colorful_view_count = num_of_nodes - nodes_with_mono_view_count
    thresh_nodes_with_colorful_view_count = num_of_nodes - thresh_nodes_with_mono_view_count
    total_mono_per = total_mono_count / not_zero_edges_count
    thresh_total_mono_per = thresh_total_mono_count / thresh_edges_count
    total_colorful_per = total_colorful_count / not_zero_edges_count
    thresh_total_colorful_per = thresh_total_colorful_count / thresh_edges_count
    nodes_entropy_sorted = calc_nodes_entropy_sorted(nodes_in_mono_sum)

    mat_norm = get_norm(weights_mat, norm)

    insights_dict[strs.str_not_zero_edges_count].append(not_zero_edges_count)
    insights_dict[strs.str_thresh_edges_count].append(thresh_edges_count)
    insights_dict[strs.str_thresh_total_weight].append(thresh_total_weight)
    insights_dict[strs.str_total_mono_count].append(total_mono_count)
    insights_dict[strs.str_thresh_total_mono_count].append(thresh_total_mono_count)
    insights_dict[strs.str_total_colorful_count].append(total_colorful_count)
    insights_dict[strs.str_thresh_total_colorful_count].append(thresh_total_colorful_count)
    insights_dict[strs.str_total_mono_per].append(total_mono_per)
    insights_dict[strs.str_thresh_total_mono_per].append(thresh_total_mono_per)
    insights_dict[strs.str_total_colorful_per].append(total_colorful_per)
    insights_dict[strs.str_thresh_total_colorful_per].append(thresh_total_colorful_per)
    insights_dict[strs.str_total_mono_weight].append(total_mono_weight)
    insights_dict[strs.str_thresh_total_mono_weight].append(thresh_total_mono_weight)
    insights_dict[strs.str_total_colorful_weight].append(total_colorful_weight)
    insights_dict[strs.str_thresh_total_colorful_weight].append(thresh_total_colorful_weight)
    insights_dict[strs.str_nodes_total_out_sum].append(nodes_total_out_sum)
    insights_dict[strs.str_thresh_nodes_total_out_sum].append(thresh_nodes_total_out_sum)
    insights_dict[strs.str_thresh_nodes_total_in_sum].append(thresh_nodes_total_in_sum)
    insights_dict[strs.str_nodes_out_mono_sum].append(nodes_out_mono_sum)
    insights_dict[strs.str_thresh_nodes_out_mono_sum].append(thresh_nodes_out_mono_sum)
    insights_dict[strs.str_nodes_out_colorful_sum].append(nodes_out_colorful_sum)
    insights_dict[strs.str_thresh_nodes_out_colorful_sum].append(thresh_nodes_out_colorful_sum)
    insights_dict[strs.str_nodes_in_mono_sum].append(nodes_in_mono_sum)
    insights_dict[strs.str_thresh_nodes_in_mono_sum].append(thresh_nodes_in_mono_sum)
    insights_dict[strs.str_nodes_in_colorful_sum].append(nodes_in_colorful_sum)
    insights_dict[strs.str_thresh_nodes_in_colorful_sum].append(thresh_nodes_in_colorful_sum)
    insights_dict[strs.str_nodes_with_mono_view_count].append(nodes_with_mono_view_count)
    insights_dict[strs.str_thresh_nodes_with_mono_view_count].append(thresh_nodes_with_mono_view_count)
    insights_dict[strs.str_nodes_with_colorful_view_count].append(nodes_with_colorful_view_count)
    insights_dict[strs.str_thresh_nodes_with_colorful_view_count].append(thresh_nodes_with_colorful_view_count)
    insights_dict[strs.str_nodes_entropy_sorted].append(nodes_entropy_sorted)
    insights_dict[strs.str_nodes_total_out_sum_sorted].append(-np.sort(-nodes_total_out_sum))
    insights_dict[strs.str_mat_norm].append(mat_norm)

    insights_dict[strs.str_edge_weight].append(weights_mat.flatten())
    insights_dict[strs.str_local_disagreement].append(nodes_local_disagreement)

    global_disag = global_disagreement(nodes_opinions, weights_mat)
    insights_dict[strs.str_global_disagreement].append(global_disag)

    nodes_feed_mean, nodes_distance_from_feed_mean = nodes_feed_mean_and_distance_from_feed_mean(nodes_opinions, weights_mat)
    insights_dict[strs.str_nodes_distance_feed_mean].append(np.asarray(nodes_distance_from_feed_mean))
    insights_dict[strs.str_nodes_feed_mean].append(np.asarray(nodes_feed_mean))

    return insights_dict


def initiate_insights_dict():
    insights_dict = {}
    insights_dict[strs.str_not_zero_edges_count] = []
    insights_dict[strs.str_thresh_edges_count] = []
    insights_dict[strs.str_thresh_total_weight] = []
    insights_dict[strs.str_total_mono_count] = []
    insights_dict[strs.str_thresh_total_mono_count] = []
    insights_dict[strs.str_total_colorful_count] = []
    insights_dict[strs.str_thresh_total_colorful_count] = []
    insights_dict[strs.str_total_mono_per] = []
    insights_dict[strs.str_thresh_total_mono_per] = []
    insights_dict[strs.str_total_colorful_per] = []
    insights_dict[strs.str_thresh_total_colorful_per] = []
    insights_dict[strs.str_total_mono_weight] = []
    insights_dict[strs.str_thresh_total_mono_weight] = []
    insights_dict[strs.str_total_colorful_weight] = []
    insights_dict[strs.str_thresh_total_colorful_weight] = []
    insights_dict[strs.str_nodes_total_out_sum] = []
    insights_dict[strs.str_thresh_nodes_total_out_sum] = []
    insights_dict[strs.str_thresh_nodes_total_in_sum] = []
    insights_dict[strs.str_nodes_out_mono_sum] = []
    insights_dict[strs.str_thresh_nodes_out_mono_sum] = []
    insights_dict[strs.str_nodes_out_colorful_sum] = []
    insights_dict[strs.str_thresh_nodes_out_colorful_sum] = []
    insights_dict[strs.str_nodes_in_mono_sum] = []
    insights_dict[strs.str_thresh_nodes_in_mono_sum] = []
    insights_dict[strs.str_nodes_in_colorful_sum] = []
    insights_dict[strs.str_thresh_nodes_in_colorful_sum] = []
    insights_dict[strs.str_nodes_with_mono_view_count] = []
    insights_dict[strs.str_thresh_nodes_with_mono_view_count] = []
    insights_dict[strs.str_nodes_with_colorful_view_count] = []
    insights_dict[strs.str_thresh_nodes_with_colorful_view_count] = []
    insights_dict[strs.str_nodes_total_out_sum_sorted] = []
    insights_dict[strs.str_nodes_entropy_sorted] = []
    insights_dict[strs.str_mat_norm] = []
    insights_dict[strs.str_local_disagreement] = []
    insights_dict[strs.str_global_disagreement] = []
    insights_dict[strs.str_edge_weight] = []
    insights_dict[strs.str_eig_val] = []
    insights_dict[strs.str_eig_vec] = []
    insights_dict[strs.str_nodes_distance_feed_mean] = []
    insights_dict[strs.str_nodes_feed_mean] = []

    return insights_dict


def blocks_insights(weights_blocks_mat, iteration):
    num_of_nodes = len(weights_blocks_mat)
    half_num_of_nodes = int(num_of_nodes / 2)
    all_mat_sum = weights_blocks_mat.sum()
    RR = weights_blocks_mat[: half_num_of_nodes, : half_num_of_nodes]
    BB = (weights_blocks_mat[half_num_of_nodes : num_of_nodes, half_num_of_nodes: num_of_nodes])
    BR = weights_blocks_mat[half_num_of_nodes: num_of_nodes, : half_num_of_nodes]
    RB = weights_blocks_mat[: half_num_of_nodes, half_num_of_nodes: num_of_nodes]
    RR_normalized = RR.sum() / all_mat_sum
    BB_normalized = BB.sum() / all_mat_sum
    BR_normalized = BR.sum() / all_mat_sum
    RB_normalized = RB.sum() / all_mat_sum
    print('RR: ', RR_normalized)
    print('BB: ', BB_normalized)
    print('BR: ', BR_normalized)
    print('RB: ', RB_normalized)

    RR_nodes_out_sum = nodes_out_sum(RR)
    BB_nodes_out_sum = nodes_out_sum(BB)
    RR_nodes_out_sum_sorted = -np.sort(-RR_nodes_out_sum)
    BB_nodes_out_sum_sorted = -np.sort(-BB_nodes_out_sum)

    nodes_arr = range(half_num_of_nodes)
    Plots.plot_all(nodes_arr, [RR_nodes_out_sum_sorted, BB_nodes_out_sum_sorted], 'same opinion nodes sorted by out-weight',
                   'out sum',
                   'out-edges weights sum iteration ' + str(iteration), ['RR', 'BB'], 'Iteration')

    Plots.plot_all_symlog_loglog(nodes_arr, [RR_nodes_out_sum_sorted, BB_nodes_out_sum_sorted],
                                 'same opinion log_matplot nodes sorted by out-weight',
                                 'log out sum', 'log log matplot symlog iteration' + str(iteration), ['RR', 'BB'], 'Iteration')


