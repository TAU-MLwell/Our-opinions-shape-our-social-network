import networkx as nx
import numpy as np
import Insights
import Plots
import Strings as strs
import Utils
import copy

def edges_boosting(weights_mat):
    length_two_paths_mat = np.matmul(weights_mat, weights_mat)
    np.fill_diagonal(length_two_paths_mat, 0)
    lambda_1 = 0.1
    weights_mat = weights_mat + lambda_1*length_two_paths_mat
    return weights_mat


def edges_decay(weights_mat, opinions_score_mat, initial_wights):
    lambda_2 = 1
    lambda_3 = 0.1
    weights_mat = lambda_2*np.multiply(weights_mat, opinions_score_mat)
    weights_mat = weights_mat + lambda_3*initial_wights
    return weights_mat


def normalization(adj_mat):
    adj_mat = adj_mat / adj_mat.sum(axis=0, keepdims=1)
    return adj_mat


def simulation(num_of_nodes, num_of_iterations, take_measures_every, zero_thresh, model_type=strs.str_model_type_basic, num_of_opinions=2):
    num_of_nodes = num_of_nodes
    num_of_iterations = num_of_iterations
    take_measures_every = take_measures_every
    norm = 2
    nodes_arr = np.arange(1, num_of_nodes + 1, 1)
    edges_arr = np.arange(1, (num_of_nodes*num_of_nodes) + 1, 1)
    all_iterations_arr = np.arange(1, num_of_iterations + 1, 1)
    barabasi_albert_graph = nx.barabasi_albert_graph(num_of_nodes, 3)
    adj_mat = nx.adjacency_matrix(barabasi_albert_graph).toarray()
    weights_mat = Utils.init_edge_weights_from_adj_mat(adj_mat)
    initial_wights_mat_copy = copy.deepcopy(weights_mat)
    if model_type == strs.str_model_type_basic:
        nodes_opinons = Utils.init_random_nodes_d_colors(num_of_nodes, 2)
    elif model_type == strs.str_model_type_multiple_opinions:
        nodes_opinons = Utils.init_random_nodes_d_colors(num_of_nodes, num_of_opinions)
        opinions_scores_mat = Utils.compute_nodes_multiple_opinions_scores(nodes_opinons, 0.5, 0.2)
    elif model_type == strs.str_model_type_continuous_opinions:
        nodes_opinons = Utils.init_continuous_nodes_opinions(num_of_nodes)
        opinions_scores_mat = Utils.compute_nodes_continuous_diff_op_scores(nodes_opinons)
    else:
        first_topic_nodes_opinions = Utils.init_random_nodes_d_colors(num_of_nodes, 2)
        second_topic_nodes_opinions = Utils.init_random_nodes_d_colors(num_of_nodes, 2)
        opinions_scores_mat = Utils.compute_nodes_multiple_topics_scores(first_topic_nodes_opinions,
                                                                       second_topic_nodes_opinions)
    iterations_arr = []
    insights_dict = Insights.initiate_insights_dict()
    mat_dis = []
    for i in range(1, num_of_iterations + 1):
        print('Iteration: ' + str(i))
        weights_mat_copy = copy.deepcopy(weights_mat)
        if (i % take_measures_every == 0 or i == 1):
            # weights_blocks_map = Utils.arrange_weights_two_opinions(weights_mat, first_topic_nodes_opinions, second_topic_nodes_opinions)
            # log_weights_pixel_map = np.log(weights_blocks_map)
            # Plots.plot_mat(log_weights_pixel_map, 'nodes sorted by color', 'nodes sorted by color', 'Log blocks weights iteration: ' + str(i))

            iterations_arr.append(i)
            Insights.get_measurements(weights_mat, adj_mat, nodes_opinons, zero_thresh, insights_dict, norm)

        weights_mat = edges_boosting(weights_mat)
        weights_mat = edges_decay(weights_mat, opinions_scores_mat, initial_wights_mat_copy)
        weights_mat = normalization(weights_mat)
        dis = Insights.get_norm(weights_mat_copy - weights_mat, norm)
        mat_dis.append(dis)


    total_mono_weight = insights_dict[strs.str_total_mono_weight]
    total_mono_weight = (np.asarray(total_mono_weight) / num_of_nodes) * 100
    Plots.plot_graph(iterations_arr, total_mono_weight, 'Iteration',
                        '% of total edges weight from monochromatic edges', '')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_not_zero_edges_count], 'iteration', 'num of not zero edges', 'number of edges')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_total_mono_count], 'iteration', 'monoch edges num', 'number of mono edges')

    Plots.plot_graph(iterations_arr,insights_dict[strs.str_total_colorful_count], 'iteration', 'colorful edges num',  'number of colorful edges')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_total_mono_per], 'iteration', 'percentage of monochromatic edges',
                        'monoch edges per')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_total_colorful_per], 'iteration', 'percentage of colorful edges',
                        'colorful edges per')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_total_mono_weight], 'iteration', 'total mono weights',
                        'total mono weights')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_nodes_with_mono_view_count], 'iteration',
                     'num of nodes with monochromatic view', 'monochromatic view')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_thresh_nodes_with_mono_view_count],  'iteration',
                     'num of nodes with thresh monochromatic view', 'thresh monochromatic view')

    Plots.plot_all(nodes_arr, insights_dict[strs.str_nodes_in_mono_sum], 'Nodes',
                      'Total weight from in monochromatic edges', '', iterations_arr, 'Iteration')

    Plots.plot_all(nodes_arr, insights_dict[strs.str_nodes_entropy_sorted], 'nodes sorted by entropy', 'entropy', 'entropy',
                   iterations_arr, 'Iteration')

    Plots.plot_all(nodes_arr, insights_dict[strs.str_nodes_total_out_sum_sorted], 'nodes sorted by out-weight', 'out sum',
                      'out-edges weights sum', iterations_arr, 'Iteration')

    Plots.plot_all_symlog_loglog(nodes_arr, insights_dict[strs.str_nodes_total_out_sum_sorted], 'Nodes sorted by out-weight',
                                 'Sum of out-edges weights', ' ', iterations_arr, 'Iteration')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_mat_norm],  'iteration',
                     'weights mat ' + str(norm) + ' norm', str(norm) + ' mat norm')
    Plots.plot_graph(all_iterations_arr, mat_dis,  'iteration',
                     str(norm) + ' distance', 'weights mat ' + str(norm) + ' distances')

    Plots.plot_graph(iterations_arr, insights_dict[strs.str_mat_norm],  'iteration',
                     'weights mat ' + str(norm) + ' norm', str(norm) + ' mat norm')

    nodes_distances_list_of_lists = insights_dict[strs.str_nodes_mix]
    sorted_nodes_mix = Utils.sort_lists_by_listargsort(nodes_distances_list_of_lists, nodes_opinons)

    Plots.plot_all(nodes_arr, nodes_distances_list_of_lists, 'nodes', 'Distance from feed mean',
                      '', iterations_arr, 'Iteration')

    Plots.plot_all(nodes_arr, sorted_nodes_mix, 'Nodes sorted by opinion', 'Distance from feed mean',
                      '', iterations_arr, 'Iteration')

    nodes_local_dis_list_of_lists = insights_dict[strs.str_local_disagreement]
    sorted_loal_dis = Utils.sort_lists_by_listargsort(nodes_local_dis_list_of_lists, nodes_opinons)
    Plots.plot_all(nodes_arr, nodes_local_dis_list_of_lists, 'nodes', 'disagreement',
                      'local disagreement', iterations_arr, 'Iteration')

    Plots.plot_all(nodes_arr, sorted_loal_dis, 'nodes sorted by opinion', 'disagreement',
                      'local disagreement sorted', iterations_arr, 'Iteration')


    Plots.plot_all(edges_arr, insights_dict[strs.str_edge_weight], 'edges', 'edges weights',
                      'edges weights', iterations_arr, 'Iteration')


    return

"""
choose a model type 
basic: str_model_type_basic 
multiple_opinions: str_model_type_multiple_opinions
multiple_topics: str_model_type_multiple_topics
continuous_opinions: str_model_type_continuous_opinions
"""
simulation(num_of_nodes=1000, num_of_iterations=1000, take_measures_every=100,
           zero_thresh=1e-4, model_type=strs.str_model_type_basic, num_of_opinions=2)
