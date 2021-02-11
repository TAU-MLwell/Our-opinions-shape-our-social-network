from random import sample
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import Plots


def get_nodes_log_enumeration(num_of_nodes):
    enumeration = range(num_of_nodes)
    log_enumeration = np.log(enumeration)
    return log_enumeration


def count_mutual_friends(adj_mat):
    mutual_count_mat = np.zeros_like(adj_mat)
    for row in range(0, len(adj_mat) - 1):  # iterate lower triangle
        for col in range(row + 1, len(adj_mat)):
            vec1 = adj_mat[row]
            vec2 = adj_mat[col]
            mutual_vec = np.logical_and(vec1, vec2)
            count_mutual = np.sum(mutual_vec)
            mutual_count_mat[row][col] = count_mutual

    mutual_count_mac_sym = mutual_count_mat + mutual_count_mat.transpose()
    return mutual_count_mac_sym


def draw_graph(G, opinions_map, title):
    pos = nx.spring_layout(G, k=0.9, iterations=20)
    plt.title(title)
    nx.draw(G, pos, node_color=opinions_map , with_labels=True, font_color='white')
    plt.show()


def get_mono_edges_mat(nodes_opinions):
    num_of_nodes = nodes_opinions.size
    mono_edges_mat = np.zeros(shape=(num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if nodes_opinions[i] == nodes_opinions[j]:
                mono_edges_mat[i][j] = 1

    return mono_edges_mat.astype(bool)


def get_two_in_layers_around_node(weight_mat, node, zero_thresh):
    node_layers_adj_mat = np.zeros(shape=(len(weight_mat), len(weight_mat)))
    first_in_layer = weight_mat[:, node]
    first_in_layer[first_in_layer < zero_thresh] = 0
    node_layers_adj_mat[:, node] = first_in_layer
    for f_node in range(len(first_in_layer)):
        if first_in_layer[f_node] > 0:
            f_node_in_layer = weight_mat[:, f_node]
            f_node_in_layer[f_node_in_layer < zero_thresh] = 0
            node_layers_adj_mat[:, f_node] = f_node_in_layer
    node_layers_adj_mat[node] = 0
    return node_layers_adj_mat


def get_two_out_layers_around_node(weight_mat, node, zero_thresh):
    node_layers_adj_mat = np.zeros(shape=(len(weight_mat), len(weight_mat)))
    first_out_layer = weight_mat[node]
    first_out_layer[first_out_layer < zero_thresh] = 0
    node_layers_adj_mat[node] = first_out_layer
    for f_node in range(len(first_out_layer)):
        if first_out_layer[f_node] > 0:
            f_node_out_layer = weight_mat[f_node]
            f_node_out_layer[f_node_out_layer < zero_thresh] = 0
            node_layers_adj_mat[f_node] = f_node_out_layer
    node_layers_adj_mat[:, node] = 0
    return node_layers_adj_mat


def draw_directed_graph(G, nodes_opinions):
    pos = nx.layout.spring_layout(G)

    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

def directed_graph_from_weights_mat(weights_mat, zero_thresh):
    weights_mat_copy = np.copy(weights_mat)
    weights_mat_copy[weights_mat_copy > zero_thresh] = 1
    weights_mat_copy[weights_mat_copy <= zero_thresh] = 0
    G = nx.from_numpy_array(weights_mat_copy, parallel_edges=True, create_using=nx.DiGraph)
    return G


def graph_from_weights_mat(weights_mat, zero_thresh):
    weights_mat_copy = np.copy(weights_mat)
    weights_mat_copy[weights_mat_copy > zero_thresh] = 1
    weights_mat_copy[weights_mat_copy <= zero_thresh] = 0
    G = nx.from_numpy_array(weights_mat_copy)
    return G


def set_nodes_opinions_map(nodes_opinions, opinion1, opinion2):
    color_map = []
    for idx, color in enumerate(nodes_opinions):
        if color == 0:
            color_map.append(opinion1)
        else:
            color_map.append(opinion2)
    return color_map


def arrange_weights_mat_by_two_opinions(weights_mat, nodes_opinions):
    num_of_nodes = len(weights_mat)
    indexes_of_one = np.where(nodes_opinions == 1)[0]
    indexes_of_zero = np.where(nodes_opinions == 0)[0]
    indexes_by_opinion = np.concatenate((indexes_of_zero, indexes_of_one), axis=None)
    block_weights_mat = np.zeros(shape=(num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            block_weights_mat[i][j] = weights_mat[indexes_by_opinion[i]][indexes_by_opinion[j]]
    return block_weights_mat

def arrange_weights_mat_by_multi_opinion(weights_mat, nodes_opinions):
    num_of_nodes = len(weights_mat)
    indexes_of_three = np.where(nodes_opinions == 3)[0]
    indexes_of_two = np.where(nodes_opinions == 2)[0]
    indexes_of_one = np.where(nodes_opinions == 1)[0]
    indexes_of_zero = np.where(nodes_opinions == 0)[0]
    indexes_by_opinion = np.concatenate((indexes_of_zero, indexes_of_one, indexes_of_two, indexes_of_three), axis=None)
    block_weights_mat = np.zeros(shape=(num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            block_weights_mat[i][j] = weights_mat[indexes_by_opinion[i]][indexes_by_opinion[j]]
    return block_weights_mat

def arrange_weights_mat_by_cont_opinions(weights_mat, nodes_opinions):
    num_of_nodes = len(weights_mat)
    sorted_indexes_by_opinion = np.argsort(nodes_opinions)
    block_weights_mat = np.zeros(shape=(num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            block_weights_mat[i][j] = weights_mat[sorted_indexes_by_opinion[i]][sorted_indexes_by_opinion[j]]
    return block_weights_mat

def arrange_weights_multi_topics(weights_mat, first_topic_opinions, second_topic_opinions):
    num_of_nodes = len(weights_mat)

    first_zero = set(np.where(first_topic_opinions == 0)[0])
    first_one = set(np.where(first_topic_opinions == 1)[0])
    first_two = set(np.where(first_topic_opinions == 2)[0])

    second_zero = set(np.where(second_topic_opinions == 0)[0])
    second_one = set(np.where(second_topic_opinions == 1)[0])
    second_two = set(np.where(second_topic_opinions == 2)[0])

    zerozero = list(first_zero.intersection(second_zero))
    zeroone = list(first_zero.intersection(second_one))
    zerotwo = list(first_zero.intersection(second_two))

    onezero = list(first_one.intersection(second_zero))
    oneone = list(first_one.intersection(second_one))
    onetwo = list(first_one.intersection(second_two))

    twozero = list(first_two.intersection(second_zero))
    twoone = list(first_two.intersection(second_one))
    twotwo = list(first_two.intersection(second_two))


    indexes_by_opinion = np.concatenate((zerozero, zeroone, zerotwo, onezero, oneone, onetwo, twozero, twoone, twotwo), axis=None)
    block_weights_mat = np.zeros(shape=(num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            block_weights_mat[i][j] = weights_mat[indexes_by_opinion[i]][indexes_by_opinion[j]]
    return block_weights_mat


def init_random_nodes_d_opinions(num_of_nodes, d):
    opinions_arr = np.zeros(shape=(num_of_nodes, ))
    n = num_of_nodes
    group_size = int(num_of_nodes/d)
    set_indexes = set(list(range(n)))
    for i in range(d):
        curr_opinion_indexes = sample(set_indexes, group_size)
        np.put(opinions_arr, curr_opinion_indexes, i)
        set_indexes = set_indexes - set(curr_opinion_indexes)
    return opinions_arr


def compute_nodes_two_opinions_scores_matrix(nodes_opinions_array, q):
    opinion_scores_matrix = np.zeros(shape=(len(nodes_opinions_array), len(nodes_opinions_array)))
    for i in range(len(nodes_opinions_array)):
        for j in range(len(nodes_opinions_array)):
            if nodes_opinions_array[i] == nodes_opinions_array[j]:
                opinion_scores_matrix[i][j] = 1 + q
            else:
                opinion_scores_matrix[i][j] = 1
    return opinion_scores_matrix

def init_edge_weights_from_adj_mat(adj_mat):
    num_friends = len(adj_mat)
    weights_mat = np.zeros(shape = (num_friends, num_friends))
    for node in range(num_friends):
        friends_indexes = list((np.where(adj_mat[node] == 1)[0]))
        weights = list(np.random.dirichlet(np.ones(num_friends), size=1))
        node_vec = np.zeros(shape=(num_friends,))
        np.put(node_vec, friends_indexes, weights)
        weights_mat[:, node] = node_vec
    return weights_mat

def compute_nodes_multiple_opinions_scores(nodes_opinions_array, q_1, q_2):
    opinions_scores_matrix = np.zeros(shape=(len(nodes_opinions_array), len(nodes_opinions_array)))
    for i in range(len(nodes_opinions_array)):
        for j in range(len(nodes_opinions_array)):
            if nodes_opinions_array[i] == nodes_opinions_array[j]:
                opinions_scores_matrix[i][j] = 1 + q_1
            elif np.abs(nodes_opinions_array[i] - nodes_opinions_array[j]) == 1:
                opinions_scores_matrix[i][j] = 1 + q_2
            else:
                opinions_scores_matrix[i][j] = 1
    return opinions_scores_matrix

def init_continuous_nodes_opinions(num_of_nodes):
    opinions_arr = np.zeros(shape=(num_of_nodes, ))
    for i in range(num_of_nodes):
        opinions_arr[i] = random.uniform(-1, 1)
    return opinions_arr

def compute_nodes_continuous_diff_op_scores(nodes_opinions_array):
    opinions_scores_matrix = np.zeros(shape=(len(nodes_opinions_array), len(nodes_opinions_array)))
    for i in range(len(nodes_opinions_array)):
        for j in range(len(nodes_opinions_array)):
            opinions_scores_matrix[i][j] = 2 - (np.abs(nodes_opinions_array[i] - nodes_opinions_array[j]))
    return opinions_scores_matrix

def compute_nodes_multiple_topics_scores(first_topic_opinions, second_topic_opinions, q_1, q_2):
    num_of_nodes = len(first_topic_opinions)
    opinions_scores_matrix = np.zeros(shape=(num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if (first_topic_opinions[i] == first_topic_opinions[j])\
                    and (second_topic_opinions[i] == second_topic_opinions[j]):
                opinions_scores_matrix[i][j] = 1 + q_1
            elif first_topic_opinions[i] == first_topic_opinions[j] or second_topic_opinions[i] == second_topic_opinions[j]:
                opinions_scores_matrix[i][j] = 1 + q_2
            else:
                opinions_scores_matrix[i][j] = 1
    return opinions_scores_matrix

def sort_lists_by_listargsort(local_dis_list_of_lists, nodes_cont_opinions):
    nodes_soretd_indexes = np.argsort(nodes_cont_opinions)
    sorted = []
    for lst in local_dis_list_of_lists:
        sorted.append(lst[nodes_soretd_indexes])
    return sorted

def print_blocks(weights_mat, nodes_opinions, model_type):
    if model_type == 1:
        weights_blocks_map = arrange_weights_mat_by_two_opinions(weights_mat, nodes_opinions)
    elif model_type == 2:
        weights_blocks_map = arrange_weights_mat_by_cont_opinions(weights_mat, nodes_opinions)
    elif model_type == 3:
        weights_blocks_map = arrange_weights_mat_by_multi_opinion(weights_mat, nodes_opinions)
    else:
        weights_blocks_map = arrange_weights_multi_topics(weights_mat, nodes_opinions)

    log_weights_pixel_map = np.log(weights_blocks_map)
    Plots.plot_mat(log_weights_pixel_map, 'nodes sorted by opinion', 'nodes sorted by opinion', 'Log blocks weights iteration: ' + str(i))
    return log_weights_pixel_map