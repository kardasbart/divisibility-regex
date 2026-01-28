# Divisibility Regex Generator

This tool generates a **Regular Expression** that matches the string representation of any number divisible by a given divisor  in a specified numeric base .

It works by constructing a Deterministic Finite Automaton (DFA) where states represent the remainder of the number modulo , and then converting that DFA into a regular expression using the **State Elimination Method**.

## 🚀 How It Works

1. **DFA Construction**: The script builds a directed graph where:
* **Nodes**: Represent remainders .
* **Edges**: For every state  and digit , an edge exists to state .


2. **State Elimination**: To convert the graph into a regex, the script iteratively removes nodes. When a node is removed, it updates the labels of the remaining edges to include the patterns of the paths that used to pass through the deleted node.
3. **Heuristic Optimization**: The script uses a "weight" heuristic to decide which node to eliminate next, aiming to keep the resulting regex as short as possible by minimizing the product of incoming and outgoing edges.
4. **Verification**: After generating the regex, the script runs 10,000 tests to ensure the regex correctly identifies divisible numbers compared to the modulo operator.

## 🛠 Installation

Ensure you have Python installed along with the required dependencies:

```bash
pip install networkx matplotlib numpy PyQt5

```

## 📖 Usage

Run the script from the command line by providing the **base** and the **divisor**.

```bash
python main.py <base> <divisor> [--draw]

```

### Arguments:

* `base`: The number system (e.g., `2` for binary, `10` for decimal).
* `div`: The number you want to check divisibility for.
* `--draw`: (Optional) Use this flag to open a window visualizing the graph reduction process at every step.

### Example: Binary Divisibility by 4

To find the regex for binary numbers (Base 2) divisible by 4:

![div by 4](https://github.com/kardasbart/divisibility-regex/raw/master/images/bin_div_4_init.png "div by 4")

```bash
python main.py 2 4

```

**Output:**

```text
Final regex (length 35):
^((0|1(0|1(1)*0)(1(0|1(1)*0))*0))*$
Performed 10k tests, failed: 0
```

## 🔍 Code Structure

* `substitute_node()`: The core logic for removing a state and re-routing edges using Kleene star operations.
* `edges2regex()`: Handles the formatting of labels, including combining multiple digits into character sets (e.g., `[02468]`).
* `get_node_weight()`: A greedy heuristic to pick the "cheapest" node to eliminate to prevent exponential growth of the regex string.
