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
    parser.add_argument('--draw', action='store_true', help='draw graph at each step')
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
    
def is_atomic(regex):
    """Checks if the regex is atomic (doesn't need parens for *)"""
    if not regex: return True
    if len(regex) == 1: return True
    if regex.startswith('[') and regex.endswith(']') and regex.find(']', 1, -1) == -1:
        return True
    if regex.startswith('(') and regex.endswith(')'):
        depth = 0
        for i, char in enumerate(regex):
            if char == '(': depth += 1
            elif char == ')': depth -= 1
            if depth == 0 and i < len(regex) - 1:
                return False
        return True
    if len(regex) == 2 and regex[0] == '\\': return True
    return False

def format_star(inner):
    if not inner: return ""
    if is_atomic(inner):
        return f"{inner}*"
    return f"({inner})*"

def compose_default_regex(begin, inner, end):
    middle = format_star(inner)
    return f"{begin}{middle}{end}"

def create_edge_regex(ein, eout, label):
    return (ein[0], eout[1]), {"label" : label}

def retrive_label(edge):
    return edge[2]["label"]

def retrive_labels(edges):
    return [retrive_label(e) for e in edges]

def common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]

def get_balanced_prefix_len(s):
    """Returns valid lengths of prefixes that are balanced"""
    valid_indices = {0}
    stack = []
    in_bracket = False
    escaped = False
    
    for i, char in enumerate(s):
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif in_bracket:
            if char == ']':
                in_bracket = False
        else:
            if char == '[':
                in_bracket = True
            elif char == '(':
                stack.append('(')
            elif char == ')':
                if not stack: 
                     return valid_indices
                stack.pop()
        
        if not in_bracket and not stack and not escaped:
            valid_indices.add(i + 1)
            
    return valid_indices

def factor_labels(labels):
    if len(labels) < 2: return labels
    labels = sorted(labels)
    new_labels = []
    
    i = 0
    while i < len(labels):
        group = [labels[i]]
        best_prefix = ""
        
        # Look ahead for matches
        for j in range(i + 1, len(labels)):
            cp = common_prefix(group[0], labels[j])
            valid_lens = get_balanced_prefix_len(cp)
            # Pick longest valid length
            # Filter valid_lens to be <= len(cp)
            safe_len = max([l for l in valid_lens if l <= len(cp)] + [0])
            p = cp[:safe_len]
            
            if len(p) > 2:
                # If we have a current best_prefix, check if this p is compatible (should be, since sorted)
                # But as we add more items, the common prefix shrinks.
                # We want to greedily grab the largest group with a "good enough" prefix?
                # Or just grab pairs?
                # Strategy: Grab all that share at least len 3 prefix.
                # If adding label[j] reduces prefix length below 3, stop group.
                
                # Current common prefix of the WHOLE group + new item
                # Actually, we should check common prefix of new item with current group's common prefix.
                
                if len(group) == 1:
                    current_common = p
                else:
                    # intersect current_common and p
                    # Since sorted, p is common between first and j-th. 
                    # The CP of the group is determined by first and last?
                    # Yes, for sorted strings, CP(list) = CP(first, last).
                    current_common = common_prefix(group[-1], labels[j]) # wait.
                    # CP(group + new) = CP(group_first, new) AND CP(group_last, new) ?
                    # Actually CP(S) = CP(min(S), max(S)).
                    # Since 'labels' is sorted, we just check CP(labels[i], labels[j]).
                    
                    # Update current_common with new constraint
                    cp_new = common_prefix(labels[i], labels[j])
                    v_new = get_balanced_prefix_len(cp_new)
                    safe_len_new = max([l for l in v_new if l <= len(cp_new)] + [0])
                    p = cp_new[:safe_len_new]
                
                if len(p) > 2:
                    best_prefix = p
                    group.append(labels[j])
                else:
                    break
            else:
                 break
        
        if len(group) > 1:
            # Create factored label
            remainders = []
            for g in group:
                rem = g[len(best_prefix):]
                remainders.append(rem)
            
            # Recursively factor remainders?
            # Yes, might be useful.
            remainders = factor_labels(remainders)
            
            # Join remainders. Handle empty.
            # If empty string in remainders, it means one label was exactly the prefix.
            # "Prefix" | "PrefixA" -> Prefix( |A).
            rems_str = []
            has_empty = False
            for r in remainders:
                if r == "": has_empty = True
                else: rems_str.append(r)
            
            suffix = ""
            if not rems_str:
                suffix = "" # All were empty? impossible if set unique
            elif len(rems_str) == 1:
                suffix = rems_str[0]
            else:
                suffix = "(" + "|".join(rems_str) + ")"
            
            if has_empty:
                # P(|S) -> P(S)? 
                # P(S|) -> P(S?)?
                # If suffix is (A|B), we need (A|B|) -> (A|B)?
                if suffix == "":
                    # Meaning we had "" and empty list?
                    label = best_prefix
                else:
                    # Check if suffix is (...)
                    if suffix.startswith('(') and suffix.endswith(')'):
                         # (A|B)| -> (A|B)?
                         # We can't easily express ? in our limited set without adding ? support
                         # But (A|B|) is valid regex.
                         label = f"{best_prefix}(|{suffix})"
                         # Wait, (|A) is valid? Yes.
                    else:
                         label = f"{best_prefix}(|{suffix})"
            else:
                label = f"{best_prefix}{suffix}"
            
            new_labels.append(label)
            i += len(group)
        else:
            new_labels.append(labels[i])
            i += 1
            
    return new_labels

def edges2regex(edges):
    if not edges: return ""
    
    raw_labels = retrive_labels(edges)
    
    complex_labels = []
    chars = set()
    
    for l in raw_labels:
        if len(l) == 1:
            chars.add(l)
        elif l.startswith('[') and l.endswith(']') and l.find(']', 1, -1) == -1:
             chars.update(l[1:-1])
        else:
            complex_labels.append(l)
            
    if chars:
        sorted_chars = sorted(list(chars))
        if len(sorted_chars) == 1:
            complex_labels.append(sorted_chars[0])
        else:
            complex_labels.append("[" + "".join(sorted_chars) + "]")
    
    # Factor labels
    # Use set to dedup before factoring
    labels = sorted(list(set(complex_labels)), key=lambda x: x) # Lexicographical sort for prefixing
    labels = factor_labels(labels)
    # Sort by length for final aesthetic?
    labels = sorted(labels, key=lambda x: (len(x), x))
    
    if len(labels) == 1:
        return labels[0]
    else:
        return "(" + "|".join(labels) + ")"

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

def get_node_weight_regex_len(graph, nodeid):
    loops = [e for e in graph.edges.data() if e[0] == e[1] == nodeid]
    edge_in = [e for e in graph.edges.data() if e[1] == nodeid and e[0] != e[1]]
    edge_out = [e for e in graph.edges.data() if e[0] == nodeid and e[0] != e[1]]
    
    rloops = edges2regex(loops)
    
    ins = groupby(edge_in, 0)
    outs = groupby(edge_out, 1)
    
    new_len = 0
    for ein in ins:
        rins = edges2regex(ein)
        for eout in outs:
            routs = edges2regex(eout)
            label = compose_default_regex(rins, rloops, routs)
            new_len += len(label)
            
    return new_len

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
        best_node = min(nodes_to_eliminate, key=lambda n: get_node_weight_regex_len(graph, n))
        if args.draw:
            draw_labeled_multigraph(graph, "label")
        graph = substitute_node(graph, best_node)
        nodes_to_eliminate.remove(best_node)

    inner_regex = edges2regex(graph.edges.data())
    result_regex = "^" + compose_default_regex ("", inner_regex, "") + "$"
    result_length = len(result_regex)
    print(f"Final regex (length {result_length}):")
    if result_length > 1000:
        print("Regex too long to display completely.")
    else:
        print(result_regex)
    rlen = len(result_regex)
    if rlen < 5000:
        rgx = re.compile(f"^{result_regex}$")
        fails = 0
        for num in range(0, 10000):
            m = check_div(args.base, rgx, num)
            m2 = (num % args.div) == 0
            if m != m2:
                fails+=1
        print(f"Performed 10k tests, failed: {fails}")



if __name__ == "__main__":
    main()