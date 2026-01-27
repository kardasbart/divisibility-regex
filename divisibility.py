import networkx as nx
import argparse
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import re


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
    result = ""
    if begin == inner:
        result = f"{begin}+{end}"
    elif inner == end:
        result = f"{begin}{end}+"
    # elif begin == end == "":
    #     result = inner + "*"
    else:
        result = f"({begin}{inner}*{end})"
    return result


def create_edge_regex(ein, eout, label):
    return (ein[0], eout[1]), {"label" : label}

def retrive_label(edge):
    return edge[2]["label"]

def retrive_labels(edges):
    return [retrive_label(e) for e in edges]

def merge_labels(labels):
    if len(labels) >= 2:
        patern = labels[0]
        target = labels[1]
        if patern in target:
            rx = re.compile(r"(\[.*\])\+(\[.*\])")
            match = rx.match(target)
            if match is not None:
                labels = [f"{match.group(1)}*{match.group(2)}"] + (labels[2:] if len(labels) > 2 else [])
                # print(match.group(1), match.group(2), patern)
    return labels


def edges2regex(edges):
    result = ""
    if len(edges):
        digits_edges = [e for e in edges if len(retrive_label(e)) == 1]
        other_edges = [e for e in edges if len(retrive_label(e)) != 1]
        labels = retrive_labels(other_edges)
        
        if len(digits_edges) > 1:
            digits_labels = "[" + "".join(retrive_labels(digits_edges)) + "]"
            labels.append(digits_labels)
        elif len(digits_edges) == 1:
            labels.append(retrive_label(digits_edges[0]))
        labels = sorted(list(set(labels)),key=lambda x: (len(x),x))
        # is_digits = np.all([True if len(retrive_label(l)) == 1 else False for l in edges ])
        # labels = sorted(list(set(retrive_labels(edges))),key=lambda x: (len(x),x))
        # labels = merge_labels(labels)
        # labels = merge_labels(labels)
        # labels = merge_labels(labels)
        # labels = merge_labels(labels)
        if len(labels) == 1:
            result = labels[0]
        else:
            result = "(" + "|".join(labels) + ")"
    return result

def groupby(edges, key):
    values = set(map(lambda x:x[key], edges))
    groups = [[y for y in edges if y[key]==x] for x in values]
    return groups

def substitute_node(g, nodeid):
    graph = g.copy()
    
    loops = [e for e in graph.edges.data() if e[0] == e[1] == nodeid]
    edge_in = [e for e in graph.edges.data() if e[1] == nodeid and e[0] != e[1]]
    edge_out = [e for e in graph.edges.data() if e[0] == nodeid and e[0] != e[1]]
    
    graph.remove_node(nodeid)
    
    rloops = edges2regex(loops)

    for ein in groupby(edge_in,0):
        for eout in groupby(edge_out,1):
            id_in = ein[0][0]
            id_out = eout[0][1]
            rins = edges2regex(ein)
            routs = edges2regex(eout)
            graph.add_edge(id_in, id_out,**{"label" : compose_default_regex(rins,rloops,routs)})
    return graph

def check_div(base, recompiled, num):
    snum = np.base_repr(num, base = base)
    # print("REPR: ", snum)
    m = recompiled.match(snum)
    # print("M:", m)
    return m is not None

def main():
    args = parse_args()
    graph = nx.MultiDiGraph()
    for n in range(args.div):
        graph.add_node(n)

    for n in range(args.div):
        for e in range(args.base):
            dst = (n * args.base + e) % args.div
            graph.add_edge(n,dst,label=str(e))

    for n in reversed(range(1,args.div)):
        # draw_labeled_multigraph(graph, "label")
        graph = substitute_node(graph, n)

    inner_regex = edges2regex(graph.edges.data())

    print("inner = ", inner_regex)
    rlen = len(f"({inner_regex})*")
    print("Regex length = ", rlen)
    if rlen < 5000:
        rgx = re.compile(f"^({inner_regex})*$")
        fails = 0
        for num in range(0, 10000):
            m = check_div(args.base, rgx, num)
            m2 = (num % args.div) == 0
            if m != m2:
                fails+=1
                print(f"Something went wrong for num {num}: regex return = {m}, while % return {m2}")
        print(f"Performed 10k tests, failed: {fails}")



if __name__ == "__main__":
    main()