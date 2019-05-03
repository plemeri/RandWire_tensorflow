import networkx as nx
import os
import matplotlib.pyplot as plt

def graph_generator(model, graph_param, save_path, file_name):
    graph_param[0] = int(graph_param[0])
    if model == 'ws':
        graph_param[1] = int(graph_param[1])
        graph = nx.random_graphs.connected_watts_strogatz_graph(*graph_param)
    elif model == 'er':
        graph = nx.random_graphs.erdos_renyi_graph(*graph_param)
    elif model == 'ba':
        graph_param[1] = int(graph_param[1])
        graph = nx.random_graphs.barabasi_albert_graph(*graph_param)

    if os.path.isfile(save_path + '/' + file_name + '.yaml') is True:
        print('graph loaded')
        dgraph = nx.read_yaml(save_path + '/' + file_name + '.yaml')

    else:
        dgraph = nx.DiGraph()
        dgraph.add_nodes_from(graph.nodes)
        dgraph.add_edges_from(graph.edges)

    in_node = []
    out_node = []
    for indeg, outdeg in zip(dgraph.in_degree, dgraph.out_degree):
        if indeg[1] == 0:
            in_node.append(indeg[0])
        elif outdeg[1] == 0:
            out_node.append(outdeg[0])
    # print(in_node, out_node)
    sorted = list(nx.topological_sort(dgraph))
    # nx.draw(dgraph)
    # plt.draw()
    # plt.show()

    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)

    if os.path.isfile(save_path + '/' + file_name + '.yaml') is False:
        print('graph_saved')
        nx.write_yaml(dgraph, save_path + '/' + file_name + '.yaml')

    return dgraph, sorted, in_node, out_node

