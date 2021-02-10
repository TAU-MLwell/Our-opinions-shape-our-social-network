import networkx as nx
import numpy as np
import Insights
import Plots
import Strings as strs
import Utils
import copy

def normalization(adj_mat):
    adj_mat = adj_mat / adj_mat.sum(axis=0, keepdims=1)
    return adj_mat


def update_step(weights_mat, square_initial, nodes_opinions_scores_mat, initial_wights):
    square_lambda = 0.1
    decay_lambda = 1
    init_lambda = 0.6
    updated_weights_mat = weights_mat + square_lambda*np.matmul(weights_mat, square_initial) + \
                  decay_lambda * np.multiply(weights_mat, nodes_opinions_scores_mat) + init_lambda*initial_wights
    np.fill_diagonal(updated_weights_mat, 0)
    return updated_weights_mat


def simulation(num_of_nodes, num_of_iterations, take_measures_every, model_type):
    num_of_nodes = num_of_nodes
    num_of_iterations = num_of_iterations
    q_1 = 0.6
    q_2 = 0.1
    alpha = 0.1
    lamda = 0.4
    nodes_arr = np.arange(1, num_of_nodes + 1, 1)
    barabasi_albert_graph = nx.barabasi_albert_graph(num_of_nodes, 3)
    adj_mat = nx.adjacency_matrix(barabasi_albert_graph).toarray()
    weights_mat = Utils.init_edge_weights_from_adj_mat(adj_mat)
    initial_wights_mat_copy = copy.deepcopy(weights_mat)
    nodes_opinions = []
    opinions_scores_mat = np.zeros(shape=(num_of_nodes, num_of_nodes))
    if model_type == 1:
        nodes_opinions = Utils.init_random_nodes_d_opinions(num_of_nodes, 2)
        opinions_scores_mat = Utils.compute_nodes_two_opinions_scores_matrix(nodes_opinions, q_1)
    elif model_type == 2:
        nodes_opinions = Utils.init_continuous_nodes_opinions(num_of_nodes)
        opinions_scores_mat = Utils.compute_nodes_continuous_diff_op_scores(nodes_opinions)
    elif model_type == 3:
        nodes_opinions = Utils.init_random_nodes_d_opinions(num_of_nodes, 4)
        opinions_scores_mat = Utils.compute_nodes_multiple_opinions_scores(nodes_opinions, q_1, 1_2)
    else:
        first_topic_nodes_opinions = Utils.init_random_nodes_d_opinions(num_of_nodes, 3)
        second_topic_nodes_opinions = Utils.init_random_nodes_d_opinions(num_of_nodes, 3)
        opinions_scores_mat = Utils.compute_nodes_multiple_topics_scores(first_topic_nodes_opinions, second_topic_nodes_opinions, q_1, q_2)

    iterations_arr = []
    insights_dict = Insights.initiate_insights_dict()
    for i in range(1, num_of_iterations + 1):
        print('Iteration: ' + str(i))

        if (i == take_measures_every or i == 1):
            Utils.print_blocks(weights_mat, nodes_opinions)
            iterations_arr.append(i)
            Insights.get_measurements(weights_mat, adj_mat, nodes_opinions, insights_dict)

        length_two_paths_mat = np.matmul(weights_mat, weights_mat)
        length_two_paths_mat_zero_diag = np.fill_diagonal(length_two_paths_mat, 0)
        weights_mat = np.multiply(weights_mat, opinions_scores_mat) + \
                      lamda * length_two_paths_mat_zero_diag + alpha * initial_wights_mat_copy
        weights_mat = normalization(weights_mat)


    total_mono_weight = insights_dict[strs.str_total_mono_weight]
    total_mono_weight = (np.asarray(total_mono_weight) / num_of_nodes) * 100
    Plots.plot_graph(iterations_arr, total_mono_weight, 'Iteration',
                        '% of total edges weight from monochromatic edges', '')

    Plots.plot_all(nodes_arr, insights_dict[strs.str_nodes_in_mono_sum], 'Nodes',
                      'Total weight from in monochromatic edges', '', iterations_arr, 'Iteration')


    Plots.plot_all_symlog_loglog(nodes_arr, insights_dict[strs.str_nodes_total_out_sum_sorted], 'Nodes sorted by out-weight',
                                 'Sum of out-edges weights', '', iterations_arr, 'Iteration')

    nodes_distance_from_mean_list_of_lists = insights_dict[strs.str_nodes_distance_feed_mean]
    sorted_nodes_lists = Utils.sort_lists_by_listargsort(nodes_distance_from_mean_list_of_lists, nodes_opinions)
    Plots.plot_all(nodes_arr, sorted_nodes_lists, 'Nodes sorted by opinion', 'Distance from feed mean',
                      '', iterations_arr, 'Iteration')

    nodes_feeds_mean_list_of_lists = insights_dict[strs.str_nodes_feed_mean]
    sorted_nodes_lists = Utils.sort_lists_by_listargsort(nodes_feeds_mean_list_of_lists, nodes_opinions)
    Plots.plot_all(nodes_arr, sorted_nodes_lists, 'Nodes sorted by opinion', 'Feed mean',
                      '', iterations_arr, 'Iteration')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_polarization], 'iteration',
                        'Polarization per iteration', 'polarization')

    nodes_local_dis_list_of_lists = insights_dict[strs.str_local_disagreement]
    sorted_loal_dis = Utils.sort_lists_by_listargsort(nodes_local_dis_list_of_lists, nodes_opinions)
    Plots.plot_all(nodes_arr, sorted_loal_dis, 'nodes sorted by opinion', '',
                      'local disagreement sorted', iterations_arr, 'Iteration')

    return

simulation(num_of_nodes=100, num_of_iterations=100, take_measures_every=10, model_type = 1)
