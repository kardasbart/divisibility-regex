import networkx as nx
import argparse
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import re
import sympy.ntheory as ntheory
from typing import List, Set, Dict, Tuple, Optional, Any

class RegexUtils:
    """Utilities for handling and formatting regular expressions."""

    @staticmethod
    def is_atomic(regex: str) -> bool:
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
            start_brace = regex.rfind('{')
            if start_brace > 0:
                core = regex[:start_brace]
                return RegexUtils.is_atomic(core)
        return False

    @staticmethod
    def format_star(inner: str) -> str:
        """Wraps a regex in a star (repetition) operator, adding parens if needed."""
        if not inner: return ""
        if RegexUtils.is_atomic(inner):
            return f"{inner}*"
        return f"({inner})*"

    @staticmethod
    def compose_default_regex(begin: str, inner: str, end: str) -> str:
        """Composes a regex logic: begin + (inner)* + end."""
        middle = RegexUtils.format_star(inner)
        return f"{begin}{middle}{end}"

    @staticmethod
    def common_prefix(s1: str, s2: str) -> str:
        """Finds the common prefix of two strings."""
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] != s2[i]:
                return s1[:i]
        return s1[:min_len]

    @staticmethod
    def common_suffix(s1: str, s2: str) -> str:
        """Finds the common suffix of two strings."""
        min_len = min(len(s1), len(s2))
        for i in range(1, min_len + 1):
            if s1[-i] != s2[-i]:
                if i == 1: return ""
                return s1[-(i-1):]
        return s1[-min_len:]

    @staticmethod
    def get_balanced_prefix_len(s: str) -> Set[int]:
        """Returns valid lengths of prefixes that are balanced in terms of brackets."""
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
                    in_bracket = True
                elif char == '}':
                    pass 
            
            if char == '}' and in_bracket:
                 in_bracket = False

            if not in_bracket and not stack and not escaped:
                valid_indices.add(i + 1)
                
        return valid_indices
    @staticmethod
    def create_edge_regex(ein, eout, label):
        return (ein[0], eout[1]), {"label" : label}

    @staticmethod
    def retrieve_label(edge):
        return edge[2]["label"]

    @staticmethod
    def retrieve_labels(edges):
        return [RegexUtils.retrieve_label(e) for e in edges]


class RegexOptimizer:
    """Handles simplification and optimization of regex lists."""

    @staticmethod
    def get_balanced_suffix_len(s: str) -> Set[int]:
        """Returns valid lengths of suffixes that are balanced (from the end)."""
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
                 if s[i] in {'*', '+', '?', '{'}:
                     if not is_escaped:
                          pass # Reject
                     else:
                          valid_lens.add(n - i)
                 else:
                     valid_lens.add(n - i)
                 
        return valid_lens

    @staticmethod
    def factor_suffixes(labels: List[str]) -> List[str]:
        if len(labels) < 2: return labels
        labels = sorted(labels, key=lambda x: x[::-1])
        new_labels = []
        
        i = 0
        while i < len(labels):
            group = [labels[i]]
            best_suffix = ""
            
            for j in range(i + 1, len(labels)):
                cs = RegexUtils.common_suffix(group[0], labels[j]) 
                valid_lens = RegexOptimizer.get_balanced_suffix_len(cs)
                safe_len = max([l for l in valid_lens if l <= len(cs)] + [0])
                s = cs[-safe_len:] if safe_len > 0 else ""
                
                if len(s) >= 1: 
                    if len(group) == 1:
                        pass
                    else:
                        cs_new = RegexUtils.common_suffix(labels[i], labels[j]) 
                        v_new = RegexOptimizer.get_balanced_suffix_len(cs_new)
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
                
                prefixes = RegexOptimizer.factor_labels(prefixes)
                
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

    @staticmethod
    def get_repetition_info(s: str) -> Tuple[str, int]:
        """Checks if s consists of repeating a substring p, n times."""
        if not s: return s, 1
        n_len = len(s)
        for unit_len in range(1, n_len // 2 + 1):
            if n_len % unit_len == 0:
                unit = s[:unit_len]
                if unit * (n_len // unit_len) == s:
                    return unit, n_len // unit_len
        return s, 1

    @staticmethod
    def simplify_repetitions(labels: List[str]) -> List[str]:
        """Scans labels for patterns like X, XX, XXX and merges them into X{m,n}."""
        parsed = []
        for l in labels:
            root, count = RegexOptimizer.get_repetition_info(l)
            parsed.append({'original': l, 'root': root, 'count': count})
            
        parsed.sort(key=lambda x: x['root'])
        new_labels = []
        
        for root, group in it.groupby(parsed, key=lambda x: x['root']):
            group_list = list(group)
            group_list.sort(key=lambda x: x['count'])
            counts = [g['count'] for g in group_list]
            
            i = 0
            while i < len(counts):
                start = counts[i]
                j = i + 1
                while j < len(counts) and counts[j] == counts[j-1] + 1:
                    j += 1
                
                end = counts[j-1]
                
                original_cost = sum(len(group_list[k]['original']) for k in range(i, j))
                
                root_regex = root
                if not RegexUtils.is_atomic(root):
                    root_regex = f"({root})"
                
                if start == end:
                    if start == 1:
                        new_label = root_regex
                    else:
                        new_label = f"{root_regex}{{{start}}}"
                else:
                    if start == 1:
                        if end == 1: 
                            new_label = root_regex
                        else:
                            new_label = f"{root_regex}{{1,{end}}}"
                    else:
                        new_label = f"{root_regex}{{{start},{end}}}"
                
                saved_pipes = (j - i) - 1
                cost_old = original_cost + saved_pipes
                cost_new = len(new_label)
                
                if cost_new < cost_old:
                    new_labels.append(new_label)
                else:
                    for k in range(i, j):
                        new_labels.append(group_list[k]['original'])
                
                i = j
                
        return new_labels

    @staticmethod
    def factor_labels(labels: List[str]) -> List[str]:
        if len(labels) < 2: return labels
        labels = sorted(labels)
        new_labels = []
        
        i = 0
        while i < len(labels):
            group = [labels[i]]
            best_prefix = ""
            
            for j in range(i + 1, len(labels)):
                cp = RegexUtils.common_prefix(group[0], labels[j])
                valid_lens = RegexUtils.get_balanced_prefix_len(cp)
                safe_len = max([l for l in valid_lens if l <= len(cp)] + [0])
                p = cp[:safe_len]
                
                if len(p) >= 1:
                    unsafe = False
                    for candidate in [group[0], labels[j]]:
                        if len(candidate) > len(p):
                            next_char = candidate[len(p)]
                            if next_char in {'*', '+', '?', '{'}:
                                unsafe = True
                                break
                                
                    if not unsafe:
                        if len(group) > 1:
                            cp_new = RegexUtils.common_prefix(labels[i], labels[j])
                            v_new = RegexUtils.get_balanced_prefix_len(cp_new)
                            safe_len_new = max([l for l in v_new if l <= len(cp_new)] + [0])
                            p = cp_new[:safe_len_new]
                    
                    if len(p) >= 1:
                        unsafe = False
                        if len(labels[j]) > len(p):
                            if labels[j][len(p)] in {'*', '+', '?', '{'}:
                                unsafe = True
                        
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
                remainders = []
                for g in group:
                    rem = g[len(best_prefix):]
                    remainders.append(rem)
                
                remainders = RegexOptimizer.factor_labels(remainders)
                
                rems_str = []
                has_empty = False
                for r in remainders:
                    if r == "": has_empty = True
                    else: rems_str.append(r)
                
                suffix = ""
                if not rems_str:
                    suffix = ""
                elif len(rems_str) == 1:
                    suffix = rems_str[0]
                else:
                    suffix = "(" + "|".join(rems_str) + ")"
                
                if has_empty:
                    if suffix == "":
                        label = best_prefix
                    else:
                        if suffix.startswith('(') and suffix.endswith(')'):
                             label = f"{best_prefix}(|{suffix})"
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

    @staticmethod
    def labels2regex(raw_labels: List[str]) -> str:
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
        labels = RegexOptimizer.simplify_repetitions(labels)
        labels = RegexOptimizer.factor_labels(labels)
        labels = RegexOptimizer.factor_suffixes(labels)
        labels = sorted(labels, key=lambda x: (len(x), x))
        
        if len(labels) == 1:
            return labels[0]
        else:
            return "(" + "|".join(labels) + ")"

    @staticmethod
    def edges2regex(edges: List[Tuple]) -> str:
        if not edges: return ""
        return RegexOptimizer.labels2regex(RegexUtils.retrieve_labels(edges))


class SuffixSolver:
    """Handles regex generation for cases where divisibility depends on the last k digits."""

    @staticmethod
    def check_suffix_based(base: int, div: int) -> Optional[int]:
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

    @staticmethod
    def generate_regex(base: int, div: int, k: int) -> str:
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

        # Optimization for Zeros
        has_zero = any(l == '0' for l in short_labels)
        if has_zero:
            short_labels = [l for l in short_labels if set(l) != {'0'}]
            short_labels.append('0+')
            
        short_regex = RegexOptimizer.labels2regex(short_labels)
        long_suffix_regex = RegexOptimizer.labels2regex(long_labels)
        
        all_digits = [np.base_repr(i, base=base) for i in range(base)]
        prefix_regex = RegexOptimizer.labels2regex(all_digits)
        prefix = RegexUtils.format_star(prefix_regex)
        
        if short_regex:
            if short_regex.startswith('(') and short_regex.endswith(')') and RegexUtils.is_atomic(short_regex):
                 short_regex = short_regex[1:-1]
            return f"^({short_regex}|{prefix}{long_suffix_regex})$"
        else:
            return f"^{prefix}{long_suffix_regex}$"


class DivisibilityGraph:
    """Manages the graph conversion to regex."""
    
    def __init__(self, base: int, div: int, draw: bool = False):
        self.base = base
        self.div = div
        self.draw_graphs = draw
        self.graph = nx.MultiDiGraph()
        self._build()

    def _build(self):
        for n in range(self.div):
            self.graph.add_node(n)

        for n in range(self.div):
            for e in range(self.base):
                dst = (n * self.base + e) % self.div
                label_val = np.base_repr(e, base=self.base)
                self.graph.add_edge(n, dst, label=label_val)

    @staticmethod
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

    @staticmethod
    def groupby(edges, key):
        values = set(map(lambda x:x[key], edges))
        groups = [[y for y in edges if y[key]==x] for x in values]
        return groups

    def get_node_weight_regex_len(self, nodeid: int) -> int:
        loops = [e for e in self.graph.edges.data() if e[0] == e[1] == nodeid]
        edge_in = [e for e in self.graph.edges.data() if e[1] == nodeid and e[0] != e[1]]
        edge_out = [e for e in self.graph.edges.data() if e[0] == nodeid and e[0] != e[1]]
        
        rloops = RegexOptimizer.edges2regex(loops)
        
        ins = self.groupby(edge_in, 0)
        outs = self.groupby(edge_out, 1)
        
        new_len = 0
        for ein in ins:
            rins = RegexOptimizer.edges2regex(ein)
            for eout in outs:
                routs = RegexOptimizer.edges2regex(eout)
                label = RegexUtils.compose_default_regex(rins, rloops, routs)
                new_len += len(label)
                
        return new_len

    def eliminate_node(self, nodeid: int):
        loops = [e for e in self.graph.edges.data() if e[0] == e[1] == nodeid]
        edge_in = [e for e in self.graph.edges.data() if e[1] == nodeid and e[0] != e[1]]
        edge_out = [e for e in self.graph.edges.data() if e[0] == nodeid and e[0] != e[1]]
        
        self.graph.remove_node(nodeid)
        
        rloops = RegexOptimizer.edges2regex(loops)

        for ein in self.groupby(edge_in, 0):
            for eout in self.groupby(edge_out, 1):
                id_in = ein[0][0]
                id_out = eout[0][1]
                rins = RegexOptimizer.edges2regex(ein)
                routs = RegexOptimizer.edges2regex(eout)
                label = RegexUtils.compose_default_regex(rins, rloops, routs)
                self.graph.add_edge(id_in, id_out, label=label)

    def solve(self) -> str:
        nodes_to_eliminate = list(range(1, self.div))
        while nodes_to_eliminate:
            best_node = min(nodes_to_eliminate, key=lambda n: self.get_node_weight_regex_len(n))
            if self.draw_graphs:
                DivisibilityGraph.draw_labeled_multigraph(self.graph, "label")
            self.eliminate_node(best_node)
            nodes_to_eliminate.remove(best_node)

        inner_regex = RegexOptimizer.edges2regex(self.graph.edges.data())
        return "^" + RegexUtils.compose_default_regex("", inner_regex, "") + "$"


class DivisibilityApp:
    """Main application orchestrator."""
    
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('base', type=int, help='numeric system base')
        parser.add_argument('div', type=int, help='generate regex for divisibility by \'div\'')
        parser.add_argument('--draw', action='store_true', help='draw graph at each step')
        parser.add_argument('--no-reps', action='store_true', help='disable repetition optimization')
        parser.add_argument('--no-lookahead', action='store_true', help='disable lookahead-based factorization')
        return parser.parse_args()

    @staticmethod
    def check_div(base: int, recompiled: Any, num: int) -> bool:
        snum = np.base_repr(num, base=base)
        m = recompiled.match(snum)
        return m is not None

    @staticmethod
    def run():
        args = DivisibilityApp.parse_args()
        
        k = SuffixSolver.check_suffix_based(args.base, args.div)
        if k:
            print(f"Suffix optimization detected (last {k} digits determine divisibility).")
            result_regex = SuffixSolver.generate_regex(args.base, args.div, k)
        else:
            solver = DivisibilityGraph(args.base, args.div, args.draw)
            result_regex = solver.solve()

    @staticmethod
    def get_factors(div: int) -> Dict[int, int]:
        """Returns prime factorization as a dict {prime: power}."""
        return ntheory.factorint(div)

    @staticmethod
    def solve_composite(base: int, div: int, draw: bool, no_lookahead: bool = False) -> str:
        k_global = SuffixSolver.check_suffix_based(base, div)
        if k_global:
            print(f"Global suffix optimization detected for divisor {div} (last {k_global} digits).")
            return SuffixSolver.generate_regex(base, div, k_global)

        factors = DivisibilityApp.get_factors(div)
        
        # If passed a prime power that wasn't suffix based (or just one factor), use graph
        if len(factors) <= 1 or no_lookahead:
            solver = DivisibilityGraph(base, div, draw)
            return solver.solve()
        
        print(f"Composite divisor {div} detected. Factors: {factors}")
        
        suffix_group = 1
        graph_factors = []
        
        for p, power in factors.items():
            f = p ** power
            # Check if this factor is suffix-based
            k = SuffixSolver.check_suffix_based(base, f)
            if k:
                suffix_group *= f
            else:
                graph_factors.append(f)
                
        regexes = []
        
        # Process the suffix group
        if suffix_group > 1:
            # We re-verify the suffix property for the group. 
            # It should maintain the property with k = max(k_factors).
            k_group = SuffixSolver.check_suffix_based(base, suffix_group)
            if k_group:
                print(f"Merged suffix factors into {suffix_group} (last {k_group} digits).")
                regexes.append(SuffixSolver.generate_regex(base, suffix_group, k_group))
            else:
                # Fallback if merger fails (unlikely given logic): process individually
                # But since we already established they are suffix based, we can just split them back?
                # Or just treat the group as graph?
                # Let's treat as graph for safety, or individual lookaheads.
                # Given our reasoning, k_group should exist. 
                # If it fails, it might be due to 200 limit (but picking more specific suffix should reduce count).
                # Fallback: Treat as graph.
                print(f"Warning: Merged suffix group {suffix_group} failed check. Fallback to graph.")
                solver = DivisibilityGraph(base, suffix_group, draw)
                regexes.append(solver.solve())
        
        # Process graph factors
        for f in graph_factors:
            # Recursively solve? Or just Graph?
            # Since f is a prime power and NOT suffix based (we checked), 
            # and solve_composite for prime power -> checks suffix (False) -> Graph.
            # So we can just instantiate Graph directly.
            print(f"Using graph for factor {f}.")
            solver = DivisibilityGraph(base, f, draw)
            regexes.append(solver.solve())
            
        # Combine regexes
        if not regexes: return "" # Should not happen
        if len(regexes) == 1: return regexes[0]
        
        regexes.sort(key=len)
        main_regex = regexes.pop()
        
        final_regex = ""
        for r in regexes:
             final_regex += f"(?={r})"
        final_regex += main_regex
        
        return final_regex

    @staticmethod
    def run():
        args = DivisibilityApp.parse_args()
        
        result_regex = DivisibilityApp.solve_composite(args.base, args.div, args.draw, args.no_lookahead)
            
        result_length = len(result_regex)
        print(f"Final regex (length {result_length}):")
        if result_length > 1000:
            print("Regex too long to display completely.")
        else:
            print(result_regex)
            
        rlen = len(result_regex)
        if rlen < 5000:
            clean_regex = result_regex.strip("^$")
            rgx = re.compile(f"^{clean_regex}$")
            
            fails = 0
            for num in range(0, 10000):
                m = DivisibilityApp.check_div(args.base, rgx, num)
                m2 = (num % args.div) == 0
                if m != m2:
                    fails += 1
            print(f"Performed 10k tests, failed: {fails}")


if __name__ == "__main__":
    DivisibilityApp.run()