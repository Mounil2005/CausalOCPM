"""
Phase 1: OCEL Object Interaction Graph

Constructs a typed heterogeneous graph from the object-centric event log.
Each event involves multiple objects of different types (Case, Machine, Worker,
Material, Shipment/Discharge). We connect all objects co-occurring in the same
event, weighted by co-occurrence count.

The graph is domain-agnostic: it accepts manufacturing, healthcare, or BPI 2019
data through the same interface by mapping domain-specific column names to
the 5 canonical OCEL object roles.
"""

import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict


ROLE_COLORS = {
    'Case':             '#00C9A7',   # Teal — the primary case object
    'Resource_Machine': '#F0B429',   # Amber — machine / ward
    'Resource_Worker':  '#FF9A3C',   # Orange — worker / clinician
    'Artifact':         '#A78BFA',   # Purple — material / medication / vendor
    'Outcome':          '#FF6B6B',   # Red — shipment / discharge / invoice
}

# Column mapping per domain to the 5 OCEL object roles
_DOMAIN_COLUMNS = {
    'manufacturing': {
        'Case':             'order_id',
        'Resource_Machine': 'machine_id',
        'Resource_Worker':  'worker_id',
        'Artifact':         'material_id',
        'Outcome':          'shipment_id',
    },
    'healthcare': {
        'Case':             'patient_id',
        'Resource_Machine': 'ward_id',
        'Resource_Worker':  'clinician_id',
        'Artifact':         'medication_id',
        'Outcome':          'discharge_id',
    },
    'bpi2019': {
        'Case':             'order_id',
        'Resource_Machine': 'machine_id',
        'Resource_Worker':  'worker_id',
        'Artifact':         'material_id',
        'Outcome':          'shipment_id',
    },
}


def build_object_graph(df: pd.DataFrame, domain: str = 'manufacturing') -> nx.Graph:
    """
    Build typed heterogeneous object interaction graph from OCEL event data.

    Each row represents an event. Objects that co-occur in the same event are
    connected by an edge. Edge weight counts how often the pair co-occurs.
    Node attributes store role, object type, and occurrence count.

    Parameters
    ----------
    df : pd.DataFrame
        Event log in CausalOCPM OCEL format.
    domain : str
        One of 'manufacturing', 'healthcare', 'bpi2019'.

    Returns
    -------
    nx.Graph
        Heterogeneous object interaction graph.
    """
    col_map = _DOMAIN_COLUMNS.get(domain, _DOMAIN_COLUMNS['manufacturing'])

    # Build role → column name mapping for columns that exist in df
    available = {role: col for role, col in col_map.items() if col in df.columns}

    G = nx.Graph()
    edge_counts = defaultdict(int)
    node_counts = defaultdict(int)
    node_roles = {}

    for _, row in df.iterrows():
        # Collect all object IDs present in this event row
        event_objects = []
        for role, col in available.items():
            obj_id = str(row[col]) if pd.notna(row[col]) else None
            if obj_id and obj_id != 'nan':
                event_objects.append((obj_id, role, col))
                node_counts[obj_id] += 1
                node_roles[obj_id] = (role, col.replace('_id', '').replace('_', ' ').title())

        # Connect all pairs of objects that share this event
        for i in range(len(event_objects)):
            for j in range(i + 1, len(event_objects)):
                obj_a = event_objects[i][0]
                obj_b = event_objects[j][0]
                key = tuple(sorted([obj_a, obj_b]))
                edge_counts[key] += 1

    # Add nodes with attributes
    for obj_id, (role, obj_type) in node_roles.items():
        G.add_node(
            obj_id,
            role=role,
            object_type=obj_type,
            instance_count=node_counts[obj_id],
        )

    # Add edges with co-occurrence weights
    for (obj_a, obj_b), weight in edge_counts.items():
        if G.has_node(obj_a) and G.has_node(obj_b):
            G.add_edge(obj_a, obj_b, weight=weight)

    return G


def graph_summary(G: nx.Graph) -> dict:
    """
    Compute summary statistics for the object interaction graph.

    Returns
    -------
    dict with keys:
        total_nodes, total_edges, count per role, avg_degree,
        most_connected_node
    """
    roles = nx.get_node_attributes(G, 'role')
    role_counts = defaultdict(int)
    for node, role in roles.items():
        role_counts[role] += 1

    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values())) if degrees else 0
    most_connected = max(degrees, key=degrees.get) if degrees else None

    summary = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'avg_degree':  round(avg_degree, 2),
        'most_connected_node': most_connected,
    }
    for role in ROLE_COLORS:
        summary[f'count_{role}'] = role_counts.get(role, 0)

    return summary


def get_node_colors(G: nx.Graph) -> dict:
    """Map each node to its role hex colour."""
    roles = nx.get_node_attributes(G, 'role')
    return {node: ROLE_COLORS.get(role, '#888888')
            for node, role in roles.items()}


def get_sample_subgraph(G: nx.Graph, n_cases: int = 50, seed: int = 42) -> nx.Graph:
    """
    Sample n_cases Case nodes and their immediate neighbours.

    IMPORTANT: Always use this for Plotly visualisation. The full graph has
    thousands of nodes and spring_layout will hang or produce unreadable output.
    """
    rng = np.random.default_rng(seed)

    case_nodes = [n for n, d in G.nodes(data=True) if d.get('role') == 'Case']
    if len(case_nodes) > n_cases:
        case_nodes = rng.choice(case_nodes, size=n_cases, replace=False).tolist()

    # Include neighbours of sampled cases
    neighbour_nodes = set(case_nodes)
    for case in case_nodes:
        neighbour_nodes.update(G.neighbors(case))

    return G.subgraph(neighbour_nodes).copy()


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.generate_data import load_or_generate

    df = load_or_generate()
    G = build_object_graph(df, domain='manufacturing')
    summary = graph_summary(G)

    print("[Phase 1 — Object Interaction Graph]")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Phase 1 complete.")
