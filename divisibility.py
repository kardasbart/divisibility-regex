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
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

    pos = nx.shell_layout(G)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    nx.draw_networkx_edges(
        G, pos, connectionstyle=connectionstyle, ax=ax, arrowsize=20
    )

    edge_labels = {
        (u, v, k): d[attr_name]
        for u, v, k, d in G.edges(keys=True, data=True)
    }
    
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        connectionstyle=connectionstyle,
        font_color="red",
        bbox={"alpha": 0.7, "color": "white", "boxstyle": "round,pad=0.2"},
        ax=ax,
    )

    ax.set_title(f"Graph State: {len(G.nodes)} nodes remaining")
    
    plt.show()
    
def compose_default_regex(begin, inner, end):
    middle = f"({inner})*" if inner else ""
    return f"{begin}{middle}{end}"


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
    m = recompiled.match(snum)
    return m is not None

def get_node_weight(graph, nodeid):
    """Calculates the weight of a node based on connectivity change."""
    in_neighbors = {e[0] for e in graph.in_edges(nodeid) if e[0] != nodeid}
    out_neighbors = {e[1] for e in graph.out_edges(nodeid) if e[1] != nodeid}
    
    num_in = len(in_neighbors)
    num_out = len(out_neighbors)
    return (num_in * num_out) - (num_in + num_out)

def main():
    args = parse_args()
    graph = nx.MultiDiGraph()
    for n in range(args.div):
        graph.add_node(n)

    for n in range(args.div):
        for e in range(args.base):
            dst = (n * args.base + e) % args.div
            graph.add_edge(n,dst,label=str(e))

    nodes_to_eliminate = list(range(1,args.div))
    while nodes_to_eliminate:
        best_node = min(nodes_to_eliminate, key=lambda n: get_node_weight(graph, n))
        draw_labeled_multigraph(graph, "label")
        graph = substitute_node(graph, best_node)
        nodes_to_eliminate.remove(best_node)

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