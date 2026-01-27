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

def compose_default_regex(begin, inner, end):
    return f"({begin}{inner}*{end})"


def create_edge_regex(ein, eout, label):
    return (ein[0], eout[1]), {"label" : label}

def retrive_label(edge):
    return edge[2]["label"]

def retrive_labels(edges):
    return [retrive_label(e) for e in edges]

def edges2regex(edges):
    result = ""
    if len(edges):
        is_digits = np.all([True if len(retrive_label(l)) == 1 else False for l in edges ])
        labels = sorted(list(set(retrive_labels(edges))),key=lambda x: (len(x),x))
        if is_digits and len(labels) == 1:
            result = labels[0]
        elif is_digits:
            result = "[" + "".join(labels) + "]"
        else:
            result = "(" + "|".join(labels) + ")"
    return result


def substitute_node(g, nodeid):
    graph = g.copy()
    print("# edges = ", len(graph.edges.data()))
    loops = [e for e in graph.edges.data() if e[0] == e[1] == nodeid]
    edge_in = [e for e in graph.edges.data() if e[1] == nodeid and e[0] != e[1]]
    edge_out = [e for e in graph.edges.data() if e[0] == nodeid and e[0] != e[1]]
    # print(f"loops: {loops}\nin: {edge_in}\nout: {edge_out}")
    graph.remove_node(nodeid)

    rins = edges2regex(edge_in)
    rloops = edges2regex(loops)
    routs = edges2regex(edge_out)

    for ein in edge_in:
        for eout in edge_out:
            edge, data = create_edge_regex(ein,eout, compose_default_regex(rins,rloops,routs))
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
    # draw_labeled_multigraph(graph, "label")
    # graph = substitute_node(graph, 3)
    # draw_labeled_multigraph(graph, "label")
    # graph = substitute_node(graph, 4)
    # draw_labeled_multigraph(graph, "label")
    for n in reversed(range(1,args.div)):
        graph = substitute_node(graph, n)

    inner_regex = edges2regex(graph.edges.data())

    print("Regex length = ", len(f"({inner_regex})*"))
    print("Regex = ", f"({inner_regex})*")

if __name__ == "__main__":
    main()