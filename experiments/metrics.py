"""Directed graph scores and prior average precision; rows are causes and columns effects."""

import networkx as nx
import numpy as np


def to_adjacency_matrix(graph, node_order=None, label_to_internal=None):
    """
    Convert a graph to a binary adjacency matrix.

    adj[i, j] = 1 means node_order[i] -> node_order[j]

    Supports:
    - numpy.ndarray
    - networkx.DiGraph
    - causal-learn GeneralGraph
    """

    # Case 1: numpy array
    if isinstance(graph, np.ndarray):
        return (graph != 0).astype(int), node_order

    # Case 2: networkx DiGraph
    if isinstance(graph, nx.DiGraph):
        if node_order is None:
            node_order = list(graph.nodes())

        adj = nx.to_numpy_array(graph, nodelist=node_order, dtype=int)
        return (adj != 0).astype(int), node_order

    # Case 3: causal-learn GeneralGraph
    graph_type = str(type(graph))
    if "causallearn.graph.GeneralGraph.GeneralGraph" in graph_type:
        nodes = graph.get_nodes()
        internal_names = [node.get_name() for node in nodes]
        internal_name_to_node = {node.get_name(): node for node in nodes}

        if node_order is None:
            node_order = internal_names

        if label_to_internal is None:
            label_to_internal = {name: name for name in node_order}

        n = len(node_order)
        adj = np.zeros((n, n), dtype=int)

        for row, src_label in enumerate(node_order):
            for col, dst_label in enumerate(node_order):
                if src_label == dst_label:
                    continue

                src_internal = str(label_to_internal[src_label]).strip()
                dst_internal = str(label_to_internal[dst_label]).strip()

                src_node = internal_name_to_node.get(src_internal)
                dst_node = internal_name_to_node.get(dst_internal)

                if src_node is None or dst_node is None:
                    raise ValueError(
                        f"Could not find nodes '{src_label}' -> '{src_internal}' "
                        f"or '{dst_label}' -> '{dst_internal}' in causal-learn graph."
                    )

                edge = graph.get_edge(src_node, dst_node)
                if edge is None:
                    continue

                node1 = edge.get_node1()
                node2 = edge.get_node2()

                endpoint1 = edge.get_endpoint1().name
                endpoint2 = edge.get_endpoint2().name

                # Put endpoints in src -> dst order
                if node1 == src_node and node2 == dst_node:
                    src_endpoint, dst_endpoint = endpoint1, endpoint2
                elif node1 == dst_node and node2 == src_node:
                    src_endpoint, dst_endpoint = endpoint2, endpoint1
                else:
                    raise RuntimeError("Unexpected edge-node ordering.")

                # Strict directed edge: src -> dst
                if src_endpoint == "TAIL" and dst_endpoint == "ARROW":
                    adj[row, col] = 1

        return adj, node_order

    raise TypeError(
        "Unsupported graph type. Expected numpy.ndarray, networkx.DiGraph, "
        "or causal-learn GeneralGraph."
    )


def precision_recall_f1(true_graph, pred_graph, node_order=None, label_to_internal=None):
    true_adj, node_order = to_adjacency_matrix(true_graph, node_order, label_to_internal)
    pred_adj, _ = to_adjacency_matrix(pred_graph, node_order, label_to_internal)

    tp = np.sum((true_adj == 1) & (pred_adj == 1))
    fp = np.sum((true_adj == 0) & (pred_adj == 1))
    fn = np.sum((true_adj == 1) & (pred_adj == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def to_adjacency_matrix_all_edges(graph, node_order=None, label_to_internal=None):
    """
    Paper-style adjacency matrix.

    adj[i, j] = 1 means an edge compatible with i -> j exists.

    Edge encoding:
    - i -> j      : adj[i, j] = 1
    - i <- j      : adj[j, i] = 1
    - i --- j     : adj[i, j] = adj[j, i] = 1
    - i <-> j     : adj[i, j] = adj[j, i] = 1
    - i o-> j     : adj[i, j] = 1
    - i o-o j     : adj[i, j] = adj[j, i] = 1
    """

    if isinstance(graph, np.ndarray):
        return (graph != 0).astype(int), node_order

    if isinstance(graph, nx.DiGraph):
        if node_order is None:
            node_order = list(graph.nodes())
        adj = nx.to_numpy_array(graph, nodelist=node_order, dtype=int)
        return (adj != 0).astype(int), node_order

    if hasattr(graph, "get_nodes") and hasattr(graph, "get_edge"):
        nodes = graph.get_nodes()
        internal_name_to_node = {node.get_name(): node for node in nodes}

        if node_order is None:
            node_order = [node.get_name() for node in nodes]

        if label_to_internal is None:
            label_to_internal = {name: name for name in node_order}

        n = len(node_order)
        adj = np.zeros((n, n), dtype=int)

        for row, src_label in enumerate(node_order):
            for col, dst_label in enumerate(node_order):
                if src_label == dst_label:
                    continue

                src_internal = str(label_to_internal[src_label]).strip()
                dst_internal = str(label_to_internal[dst_label]).strip()

                src_node = internal_name_to_node.get(src_internal)
                dst_node = internal_name_to_node.get(dst_internal)

                if src_node is None or dst_node is None:
                    raise ValueError(
                        f"Could not find nodes '{src_label}' -> '{src_internal}' "
                        f"or '{dst_label}' -> '{dst_internal}' in causal-learn graph."
                    )

                edge = graph.get_edge(src_node, dst_node)
                if edge is None:
                    continue

                node1 = edge.get_node1()
                node2 = edge.get_node2()

                endpoint1 = edge.get_endpoint1().name
                endpoint2 = edge.get_endpoint2().name

                # Put endpoints in src,dst order
                if node1 == src_node and node2 == dst_node:
                    src_endpoint, dst_endpoint = endpoint1, endpoint2
                elif node1 == dst_node and node2 == src_node:
                    src_endpoint, dst_endpoint = endpoint2, endpoint1
                else:
                    raise RuntimeError("Unexpected edge-node ordering.")

                # src -> dst
                if src_endpoint == "TAIL" and dst_endpoint == "ARROW":
                    adj[row, col] = 1

                # src <- dst
                elif src_endpoint == "ARROW" and dst_endpoint == "TAIL":
                    adj[col, row] = 1

                # src --- dst, src <-> dst, src o-o dst
                elif (
                    (src_endpoint == "TAIL" and dst_endpoint == "TAIL")
                    or (src_endpoint == "ARROW" and dst_endpoint == "ARROW")
                    or (src_endpoint == "CIRCLE" and dst_endpoint == "CIRCLE")
                ):
                    adj[row, col] = 1
                    adj[col, row] = 1

                # src o-> dst
                elif src_endpoint == "CIRCLE" and dst_endpoint == "ARROW":
                    adj[row, col] = 1

                # src <-o dst
                elif src_endpoint == "ARROW" and dst_endpoint == "CIRCLE":
                    adj[col, row] = 1

        return adj, node_order

    raise TypeError(
        "Unsupported graph type. Expected numpy.ndarray, networkx.DiGraph, "
        "or causal-learn GeneralGraph."
    )


def structural_hamming_distance(true_graph, pred_graph, node_order=None, label_to_internal=None):
    """
    SHD = NE + NM + NR

    NE: extra/redundant edges
    NM: missing edges
    NR: reversed edges

    A reversed edge counts as 1 error, not 2.
    """

    # Represent unresolved adjacencies by both directions for SHD.
    true_adj, node_order = to_adjacency_matrix_all_edges(
        true_graph, node_order=node_order, label_to_internal=label_to_internal
    )

    pred_adj, _ = to_adjacency_matrix_all_edges(
        pred_graph, node_order=node_order, label_to_internal=label_to_internal
    )

    true_adj = (true_adj != 0).astype(int)
    pred_adj = (pred_adj != 0).astype(int)

    np.fill_diagonal(true_adj, 0)
    np.fill_diagonal(pred_adj, 0)

    n = true_adj.shape[0]

    NE = 0  # extra edges
    NM = 0  # missing edges
    NR = 0  # reversed edges

    for i in range(n):
        for j in range(i + 1, n):
            true_ij = true_adj[i, j]
            true_ji = true_adj[j, i]
            pred_ij = pred_adj[i, j]
            pred_ji = pred_adj[j, i]

            if true_ij == pred_ij and true_ji == pred_ji:
                continue

            # reversed edge
            if true_ij == 1 and pred_ji == 1 and true_ji == 0 and pred_ij == 0:
                NR += 1
            elif true_ji == 1 and pred_ij == 1 and true_ij == 0 and pred_ji == 0:
                NR += 1

            else:
                # count extra edges
                NE += max(0, pred_ij - true_ij)
                NE += max(0, pred_ji - true_ji)

                # count missing edges
                NM += max(0, true_ij - pred_ij)
                NM += max(0, true_ji - pred_ji)

    SHD = NE + NM + NR

    return int(SHD)


def _make_label_to_internal(col_names: list[str]) -> dict:
    """
    causal-learn names columns as X1, X2, … (1-indexed).
    Our datasets use X0, X1, … (0-indexed).
    Returns a mapping  {"X0": "X1", "X1": "X2", …}.
    """
    return {col: f"X{i + 1}" for i, col in enumerate(col_names)}


def compute_metrics(true_adj: np.ndarray, pred_graph, col_names: list[str]) -> dict:
    """Return the retained graph metrics for one reference/prediction pair."""
    mapping = _make_label_to_internal(col_names)
    precision, recall, f1 = precision_recall_f1(
        true_adj, pred_graph, node_order=col_names, label_to_internal=mapping
    )
    shd = structural_hamming_distance(
        true_adj, pred_graph, node_order=col_names, label_to_internal=mapping
    )
    return dict(f1=f1, fbeta=fbeta_score(precision, recall), precision=precision,
                recall=recall, SHD=shd)


def fbeta_score(precision, recall, beta=0.5):
    """Precision/recall F-beta; beta=0.5 is used in model selection."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    denominator = beta**2 * precision + recall
    return float((1 + beta**2) * precision * recall / denominator) if denominator else 0.0


def order_fnr(true_adj, order):
    """Fraction of true edges that contradict a cause-to-effect node ordering.

    ``order`` lists node indices from earliest cause to latest effect. For each
    true edge i -> j, an error occurs when i appears after j. An edgeless
    reference has score 0.
    """
    truth = np.asarray(true_adj)
    ordering = np.asarray(order)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise ValueError("The reference adjacency must be square")
    n = len(truth)
    if (ordering.shape != (n,) or not np.issubdtype(ordering.dtype, np.integer)
            or not np.array_equal(np.sort(ordering), np.arange(n))):
        raise ValueError("Order must be a permutation of the reference node indices")
    if not np.isfinite(truth).all() or np.any(np.diag(truth) != 0):
        raise ValueError("The reference must be finite and have no self-loops")
    position = np.empty(n, dtype=int)
    position[ordering] = np.arange(n)
    sources, targets = np.nonzero(truth)
    return float(np.mean(position[sources] > position[targets])) if len(sources) else 0.0


def directed_auprc(true_adj, scores):
    """Average precision over ordered, off-diagonal candidate edges."""
    from sklearn.metrics import average_precision_score

    truth, values = np.asarray(true_adj), np.asarray(scores, dtype=float)
    if truth.shape != values.shape or truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise ValueError("Reference and score matrices must be square and label-aligned")
    mask = ~np.eye(len(truth), dtype=bool)
    if not np.isfinite(values[mask]).all():
        raise ValueError("Edge scores must be finite")
    return float(average_precision_score((truth[mask] != 0).astype(int), values[mask]))


def existence_auprc(true_adj, scores):
    """Average precision over unordered pairs against the true skeleton."""
    from sklearn.metrics import average_precision_score

    truth, values = np.asarray(true_adj), np.asarray(scores, dtype=float)
    if truth.shape != values.shape or truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise ValueError("Reference and score matrices must be square and label-aligned")
    upper = np.triu_indices(len(truth), 1)
    if not np.isfinite(values[upper]).all():
        raise ValueError("Pair scores must be finite")
    return float(average_precision_score(((truth + truth.T) != 0)[upper], values[upper]))
