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
    parser.add_argument('--no-reps', action='store_true', help='disable repetition optimization')
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
    if len(regex) > 3 and regex[-1] == '}' and '{' in regex:
        # Check if it's atomic{m,n}
        # Find last {
        start_brace = regex.rfind('{')
        if start_brace > 0:
            core = regex[:start_brace]
            return is_atomic(core)
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
            elif char == '{':
                # Start of repetition, treated like a bracket effectively for splitting
                # We should NOT split inside {}
                in_bracket = True
            elif char == '}':
                # Should not happen if strictly correct regex unless inside bracket
                # But here we treat { ... } as atomic block for splitting purposes
                # Actually, our in_bracket logic handles it if we treat { as [
                pass 
        
        # We handle { by entering in_bracket mode. But { isn't exactly [.
        # It's fine for now, assuming valid regex.
        if char == '}' and in_bracket:
             # This is weak if we have [ ... { ... ] but simplified for now
             in_bracket = False

        if not in_bracket and not stack and not escaped:
            valid_indices.add(i + 1)
            
    return valid_indices
    return valid_indices

def get_balanced_suffix_len(s):
    """Returns valid lengths of suffixes that are balanced (from the end)"""
    valid_lens = {0}
    stack = []
    
    n = len(s)
    in_bracket = False
    
    for i in range(n - 1, -1, -1):
        char = s[i]
        
        # Check escape
        bs_count = 0
        k = i - 1
        while k >= 0 and s[k] == '\\':
            bs_count += 1
            k -= 1
        
        is_escaped = (bs_count % 2 == 1)
        
        if is_escaped:
            pass 
        elif in_bracket:
            if char == '[': # Closing the bracket (since we are going backwards)
                 in_bracket = False
        else:
            if char == ']':
                in_bracket = True
            elif char == ')':
                stack.append(')')
            elif char == '(':
                if not stack:
                    return valid_lens
                stack.pop()
            elif char == '}':
                in_bracket = True 
            elif char == '{':
                if in_bracket: in_bracket = False
        
        if not in_bracket and not stack:
             # Check if the split point (n - i) is safe.
             # Suffix starts at s[i].
             # If s[i] is a quantifier, then this suffix relies on preceding atom.
             # So we cannot act as if this suffix is independent.
             if s[i] in {'*', '+', '?', '{'}:
                 # Safe only if escaped?
                 # We checked escape above.
                 # If we are here, s[i] is the char.
                 # We need to know if it was escaped.
                 # We calculated is_escaped for s[i].
                 if not is_escaped:
                      pass # Reject
                 else:
                      valid_lens.add(n - i)
             else:
                 valid_lens.add(n - i)
             
    return valid_lens

def common_suffix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(1, min_len + 1):
        if s1[-i] != s2[-i]:
            if i == 1: return ""
            return s1[-(i-1):]
    return s1[-min_len:]

def factor_suffixes(labels):
    if len(labels) < 2: return labels
    labels = sorted(labels, key=lambda x: x[::-1])
    new_labels = []
    
    i = 0
    while i < len(labels):
        group = [labels[i]]
        best_suffix = ""
        
        for j in range(i + 1, len(labels)):
            cs = common_suffix(group[0], labels[j]) 
            valid_lens = get_balanced_suffix_len(cs)
            safe_len = max([l for l in valid_lens if l <= len(cs)] + [0])
            s = cs[-safe_len:] if safe_len > 0 else ""
            
            if len(s) >= 1: 
                if len(group) == 1:
                    current_common = s
                else:
                    cs_new = common_suffix(labels[i], labels[j]) 
                    v_new = get_balanced_suffix_len(cs_new)
                    safe_len_new = max([l for l in v_new if l <= len(cs_new)] + [0])
                    s = cs_new[-safe_len_new:] if safe_len_new > 0 else ""
                
                if len(s) >= 1:
                    best_suffix = s
                    group.append(labels[j])
                else:
                    break
            else:
                break
        
        if len(group) > 1:
            prefixes = []
            for g in group:
                if len(best_suffix) == len(g):
                    prefixes.append("")
                else:
                    prefixes.append(g[:-len(best_suffix)])
            
            prefixes = factor_labels(prefixes)
            
            p_strs = [p for p in prefixes if p != ""]
            has_empty = any(p == "" for p in prefixes)
            
            if not p_strs:
                prefix_str = ""
            elif len(p_strs) == 1:
                prefix_str = p_strs[0]
            else:
                prefix_str = "(" + "|".join(p_strs) + ")"
            
            if has_empty:
                if prefix_str == "":
                    label = best_suffix
                else:
                    label = f"(|{prefix_str}){best_suffix}"
            else:
                label = f"{prefix_str}{best_suffix}"
                
            new_labels.append(label)
            i += len(group)
        else:
            new_labels.append(labels[i])
            i += 1
            
    return new_labels
def get_repetition_info(s):
    """
    Checks if s consists of repeating a substring p, n times.
    Returns (p, n) where n is maximal.
    """
    if not s: return s, 1
    n_len = len(s)
    # Try meaningful small units first? Or smallest unit?
    # Smallest unit maximizes count.
    for unit_len in range(1, n_len // 2 + 1):
        if n_len % unit_len == 0:
            unit = s[:unit_len]
            if unit * (n_len // unit_len) == s:
                return unit, n_len // unit_len
    return s, 1

def simplify_repetitions(labels):
    """
    Scans labels for patterns like X, XX, XXX and merges them into X{1,3}.
    Also handles X{m}, X{n} logic if labels came pre-processed (unlikely).
    """
    # Parse all labels into (root, count)
    parsed = []
    for l in labels:
        root, count = get_repetition_info(l)
        parsed.append({'original': l, 'root': root, 'count': count})
        
    # Group by root
    parsed.sort(key=lambda x: x['root'])
    
    new_labels = []
    
    for root, group in it.groupby(parsed, key=lambda x: x['root']):
        group_list = list(group)
        # Sort by count
        group_list.sort(key=lambda x: x['count'])
        
        # Find contiguous ranges or beneficial merges
        # We have counts c1, c2, c3...
        # If we have 1, 2, 3 -> {1,3}
        # If we have 1, 2, 4 -> {1,2}|{4} (probably)
        
        counts = [g['count'] for g in group_list]
        
        # Simple greedy range finding
        i = 0
        while i < len(counts):
            start = counts[i]
            j = i + 1
            while j < len(counts) and counts[j] == counts[j-1] + 1:
                j += 1
            
            end = counts[j-1]
            
            # Candidate: root{start, end}
            # Or root{start} if start==end
            
            # Calculate cost
            # Original cost: sum(len(original)) for items in range
            original_cost = sum(len(group_list[k]['original']) for k in range(i, j))
            
            # New cost
            root_regex = root
            if not is_atomic(root):
                root_regex = f"({root})"
            
            if start == end:
                if start == 1:
                    new_label = root_regex
                else:
                    new_label = f"{root_regex}{{{start}}}"
            else:
                if start == 1:
                    if end == 1: 
                        new_label = root_regex # Should be covered above
                    else:
                        new_label = f"{root_regex}{{1,{end}}}"
                else:
                    new_label = f"{root_regex}{{{start},{end}}}"
            
            # Check length + overhead of '|'
            # overhead is (count - 1) pipes
            saved_pipes = (j - i) - 1
            
            # If we merge, we have 1 item instead of (j-i).
            # So meaningful comparison is:
            # Cost OLD = original_cost + saved_pipes
            # Cost NEW = len(new_label)
            
            cost_old = original_cost + saved_pipes
            cost_new = len(new_label)
            
            if cost_new < cost_old:
                new_labels.append(new_label)
            else:
                # Keep original
                for k in range(i, j):
                    new_labels.append(group_list[k]['original'])
            
            i = j
            
    return new_labels

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
            
            if len(p) >= 1:
                # Check if p is safe for group[0] and labels[j]
                # A split is unsafe if the character immediately following p is a quantifier.
                
                unsafe = False
                for candidate in [group[0], labels[j]]:
                    if len(candidate) > len(p):
                        next_char = candidate[len(p)]
                        if next_char in {'*', '+', '?', '{'}:
                            unsafe = True
                            break
                            
                if not unsafe:
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
                
                if len(p) >= 1:
                    # Validate p again for the new candidate (labels[j]) and existing group
                    # Actually we just updated p. We must check safety on labels[j] and group[0]
                    # We only need to check safety if we reduced p?
                    # Or if labels[j] has a quantifier at the new cut point.
                    
                    unsafe = False
                    # Check labels[j]
                    if len(labels[j]) > len(p):
                        if labels[j][len(p)] in {'*', '+', '?', '{'}:
                            unsafe = True
                    
                    # Check group representative (since p applies to whole group)
                    if not unsafe and len(group[0]) > len(p):
                         if group[0][len(p)] in {'*', '+', '?', '{'}:
                             unsafe = True
                             
                    if not unsafe:
                        best_prefix = p
                        group.append(labels[j])
                    else:
                        break
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

def labels2regex(raw_labels):
    if not raw_labels: return ""
    
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
    
    labels = sorted(list(set(complex_labels)), key=lambda x: x)
    labels = simplify_repetitions(labels)
    labels = factor_labels(labels)
    labels = factor_suffixes(labels)
    labels = sorted(labels, key=lambda x: (len(x), x))
    
    if len(labels) == 1:
        return labels[0]
    else:
        return "(" + "|".join(labels) + ")"

def edges2regex(edges):
    if not edges: return ""
    return labels2regex(retrive_labels(edges))

def check_suffix_based(base, div):
    try:
        if div <= 1: return 1
        rem = 1
        limit_k = 6 
        for k in range(1, limit_k + 1):
            rem = (rem * base) % div
            if rem == 0:
                count = (base ** k) // div 
                if count <= 200:
                    return k
        return None
    except:
        return None

def generate_suffix_regex(base, div, k):
    # Short
    short_labels = []
    for l in range(1, k):
        for chars in it.product([np.base_repr(i, base=base) for i in range(base)], repeat=l):
            s = "".join(chars)
            val = int(s, base)
            if val % div == 0:
                short_labels.append(s)
                
    # Long
    long_labels = []
    for chars in it.product([np.base_repr(i, base=base) for i in range(base)], repeat=k):
        s = "".join(chars)
        val = int(s, base)
        if val % div == 0:
            long_labels.append(s)

    # SPECIAL OPTIMIZATION for Zeros
    # Check if short_labels are all zeros.
    # Actually, we can just check if any label is pure zeros and replace with 0+ coverage?
    # Safe logic: if short_labels contains '0', and we are doing divisibility,
    # '0' is divisible. '00' is divisible. '000' is divisible.
    # Effectively, 0+ is valid.
    # But wait, we must be sure we aren't masking other things?
    # No, short_labels are OR'd.
    # If we add '0+', it just adds more valid strings (all zeros).
    # Since 0 is always divisible, 0+ is always valid.
    # So if short_labels has '0' (it should), we can just replace all pure-zero labels with '0+'.
    
    has_zero = any(l == '0' for l in short_labels)
    if has_zero:
        # Remove all pure zero labels
        short_labels = [l for l in short_labels if set(l) != {'0'}]
        # Add '0+'
        short_labels.append('0+')
        
    short_regex = labels2regex(short_labels)
    long_suffix_regex = labels2regex(long_labels)
    
    all_digits = [np.base_repr(i, base=base) for i in range(base)]
    prefix_regex = labels2regex(all_digits)
    prefix = format_star(prefix_regex)
    
    if short_regex:
        # Strip outer parens from short_regex if it is a group, to avoid ((A|B)|C) -> (A|B|C)
        if short_regex.startswith('(') and short_regex.endswith(')') and is_atomic(short_regex):
             short_regex = short_regex[1:-1]
        return f"^({short_regex}|{prefix}{long_suffix_regex})$"
    else:
        return f"^{prefix}{long_suffix_regex}$"

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
    
    # Check for suffix optimization
    k = check_suffix_based(args.base, args.div)
    if k:
        print(f"Suffix optimization detected (last {k} digits determine divisibility).")
        result_regex = generate_suffix_regex(args.base, args.div, k)
    else:
        graph = nx.MultiDiGraph()
        for n in range(args.div):
            graph.add_node(n)
    
        for n in range(args.div):
            for e in range(args.base):
                dst = (n * args.base + e) % args.div
                # FIX: use np.base_repr for labels > 9
                label_val = np.base_repr(e, base=args.base)
                graph.add_edge(n,dst,label=label_val)
    
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
        rgx = re.compile(f"^{result_regex}$") # regex in main logic might be full string already
        # Wait, suffix regex includes anchors ^ and $.
        # graph regex does too.
        # But verification adds wrapping anchors `^...$`.
        # My graph regex construct line 352 adds anchors.
        # My suffix regex construct adds anchors.
        # So result_regex HAS anchors.
        # But verification `rgx = re.compile(f"^{result_regex}$")` adds MORE anchors?
        # `^^...$$`? Python accepts it?
        # Safe to remove anchors from result_regex before compiling verification or fix verification.
        # I'll fix verification to NOT add anchors if they exist.
        
        # Actually simplest: Strip anchors.
        clean_regex = result_regex.strip("^$")
        rgx = re.compile(f"^{clean_regex}$")
        
        fails = 0
        for num in range(0, 10000):
            m = check_div(args.base, rgx, num)
            m2 = (num % args.div) == 0
            if m != m2:
                fails+=1
        print(f"Performed 10k tests, failed: {fails}")

if __name__ == "__main__":
    main()