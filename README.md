# Divisibility Regex Generator

This tool generates a **Regular Expression** that matches the string representation of any number divisible by a given divisor in a specified numeric base.

It employs multiple strategies including **DFA State Elimination**, **Suffix Analysis**, and **Lookahead Factorization** to generate the most concise regex possible.

## 🚀 How It Works

The tool automatically selects the best strategy based on the divisor and base:

1.  **Suffix Optimization**:
    *   For divisors where divisibility depends only on the last $k$ digits (e.g., divisible by 2, 4, 5, 10 in decimal), the tool generates a simple pattern matching those suffixes.
    *   This produces extremely short regexes compared to the DFA approach.

2.  **Composite Factorization with Lookaheads**:
    *   If the divisor is composite (e.g., 6 = 2 * 3), the tool splits it into coprime factors.
    *   It generates a regex for each factor and combines them using **Positive Lookaheads** `(?=...)`.
    *   *Example*: Divisibility by 6 becomes `(?=.*divisible_by_2)divisible_by_3`. This intersection often results in much shorter patterns than a monolithic DFA for large divisors.

3.  **DFA Construction & State Elimination**:
    *   For prime factors or cases where other optimizations don't apply, it builds a graph where nodes represent remainders.
    *   **State Elimination**: It iteratively removes nodes from the graph.
    *   **Heuristic Optimization**: A "weight" heuristic picks the cheapest node to eliminate next to minimize regex explosion.
    *   **Repetition Optimization**: Merges repeated patterns like `AA` into `A{2}` or `A{2,4}`.

## 🛠 Installation

Ensure you have Python installed along with the required dependencies:

```bash
pip install networkx matplotlib numpy sympy
```

## 📖 Usage

Run the script from the command line:

```bash
python divisibility.py <base> <divisor> [options]
```

### Arguments:

*   `base`: The number system (e.g., `2` for binary, `10` for decimal).
*   `div`: The number you want to check divisibility for.
*   `--draw`: (Optional) Open a window visualizing the graph reduction process (for DFA method).
*   `--no-reps`: (Optional) Disable `{m,n}` repetition optimizations.
*   `--no-lookahead`: (Optional) Disable splitting composite numbers; forces a single giant DFA.

### Example: Binary Divisibility by 6

```bash
python divisibility.py 2 6
```

**Output:**
```text
Composite divisor 6 detected. Factors: {2: 1, 3: 1}
Final regex (length 18):
^(?=(0|1)*0)(1(01*0)*1|0)*$
Performed 10k tests, failed: 0
```

## 🔍 Code Structure

*   `DivisibilityGraph`: Manages the graph nodes and edges for the DFA method.
*   `RegexOptimizer`: Handles simplification (`factor_labels`, `simplify_repetitions`) and merging of edges.
*   `SuffixSolver`: specialized solver for divisors defined by trailing digits.
*   `DivisibilityApp`: Orchestrates the strategy selection (Suffix vs. Composite vs. Graph).
