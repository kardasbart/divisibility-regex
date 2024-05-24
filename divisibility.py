import networkx as nx
import argparse
import matplotlib.pyplot as plt
import numpy as np
import itertools as it


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('base', type=int, help='numeric system base')
    parser.add_argument('div', type=int, help='generate regex for divisibility by \'div\'')
    args = parser.parse_args()
    return args

def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        G, pos, connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): f"{attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        font_color="black",
        bbox={"alpha": 1, "color":"white"},
        ax=ax,
    )

    plt.show()

def create_edge_regex(ein, vloop, eout):
    vin = ein[2]["label"]
    vout = eout[2]["label"]
    # if loop == None:
    #     label = f"({vin}{vout})"
    # else:
    #     vloop = loop[2]["label"]
    label = f"({vin}{vloop}*{vout})"
    return (ein[0], eout[1]), {"label" : label}

def substitute_node(g, nodeid):
    graph = g.copy()
    print("# edges = ", len(graph.edges.data()))
    loops = [e for e in graph.edges.data() if e[0] == e[1] == nodeid]
    edge_in = [e for e in graph.edges.data() if e[1] == nodeid and e[0] != e[1]]
    edge_out = [e for e in graph.edges.data() if e[0] == nodeid and e[0] != e[1]]
    # print(f"loops: {loops}\nin: {edge_in}\nout: {edge_out}")
    graph.remove_node(nodeid)
    loop = "|".join([e[2]["label"] for e in loops])
    loop = f"({loop})"
    if len(loops) == 0:
        loops.append(None)
    for ein in edge_in:
        for eout in edge_out:
            edge, data = create_edge_regex(ein,loop,eout)
            graph.add_edge(*edge, **data)
    return graph


def main():
    args = parse_args()
    graph = nx.MultiDiGraph()
    for n in range(args.div):
        graph.add_node(n)

    for n in range(args.div):
        for e in range(args.base):
            dst = (n * args.base + e) % args.div
            graph.add_edge(n,dst,label=str(e))

    # print(graph.edges.data())
    draw_labeled_multigraph(graph, "label")
    # graph = substitute_node(graph, 3)
    # draw_labeled_multigraph(graph, "label")
    # graph = substitute_node(graph, 4)
    # draw_labeled_multigraph(graph, "label")
    for n in reversed(range(1,args.div)):
        graph = substitute_node(graph, n)
    remain_regexes = [e[2]["label"] for e in graph.edges.data()]
    inner_regex = "|".join(remain_regexes)
    print("Regex length = ", len(f"({inner_regex})*"))
    print("Regex = ", f"({inner_regex})*")

if __name__ == "__main__":
    main()