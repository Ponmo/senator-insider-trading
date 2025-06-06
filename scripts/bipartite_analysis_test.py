#!/usr/bin/env python3
import os
import csv
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from networkx.algorithms import community
from networkx.algorithms.bipartite.projection import generic_weighted_projected_graph, weighted_projected_graph
import numpy as np
import pandas as pd


def parse_amount(amount_str):
    """
    Convert an amount range string like "$15,001 - $50,000" to its midpoint as a float.
    """
    # Strip quotes, commas, dollar signs
    s = amount_str.strip().replace('"', '').replace(',', '').replace('$', '')
    try:
        low, high = map(int, s.split(' - '))
    except ValueError:
        # Fallback: if single value or malformed, try to parse integer
        try:
            return float(s)
        except ValueError:
            return 0.0
    return (low + high) / 2.0


def load_senator_data(filepath):
    """
    Load senator stock transaction data from a CSV file.
    Returns the senator name and a list of (ticker, transaction_type, amount) tuples.
    """
    senator_name = Path(filepath).stem
    transactions = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('Ticker', '').strip()
            txn_raw = row.get('Transaction Type', '').strip().lower()
            amt_raw = row.get('Amount', '')
            # skip missing or placeholder tickers or missing transactions
            if not ticker or ticker == '--' or not txn_raw:
                continue

            # Classify transaction
            if txn_raw.startswith('sale'):
                txn_type = 'sale'
            elif txn_raw.startswith('purchase') or txn_raw.startswith('buy'):
                txn_type = 'purchase'
            else:
                continue

            # Parse the amount range into a numeric weight
            amount = parse_amount(amt_raw)
            transactions.append((ticker, txn_type, amount))

    return senator_name, transactions


def create_bipartite_graph(data_dir):
    """
    Create a bipartite graph between senators and stock tickers,
    tagging edges by transaction type and weight.
    """
    G = nx.Graph()
    senator_nodes = set()
    ticker_nodes = set()

    i = 0
    for filename in os.listdir(data_dir):
        i += 1
        if not filename.endswith('.csv'):
            print("Skipping")
            continue
        filepath = os.path.join(data_dir, filename)
        senator, trans_list = load_senator_data(filepath)

        if not trans_list:
            continue
        G.add_node(senator, bipartite=0)
        senator_nodes.add(senator)

        for ticker, txn_type, amount in trans_list:
            G.add_node(ticker, bipartite=1)
            ticker_nodes.add(ticker)
            G.add_edge(senator, ticker,
                       transaction=txn_type,
                       weight=amount)
            
    print("NUMBER OF SENATORS:", len(senator_nodes))

    return G, senator_nodes, ticker_nodes


def filter_by_degree(G, senator_nodes, ticker_nodes, min_degree):
    """
    Filter the bipartite graph to include only tickers with degree >= min_degree
    and the senators connected to them.
    """
    ticker_deg = {t: G.degree(t) for t in ticker_nodes}
    filtered_tickers = {t for t, deg in ticker_deg.items() if deg >= min_degree}

    filtered_senators = {
        nbr for t in filtered_tickers for nbr in G.neighbors(t)
        if nbr in senator_nodes
    }

    sub_nodes = filtered_senators.union(filtered_tickers)
    H = G.subgraph(sub_nodes).copy()
    return H, filtered_senators, filtered_tickers

def get_communities(G, senator_nodes):
    """
    1) Projects G onto senator_nodes, summing each shared ticker's 'weight'
       (we add the two senators' amounts for every ticker they both trade).
    2) Runs Louvain on that projection.
    """
    def weight_fn(B, u, v):
        # find all tickers both u and v are connected to
        shared = set(B[u]) & set(B[v])
        total = 0.0
        for t in shared:
            w_u = B[u][t].get('weight', 0)
            w_v = B[v][t].get('weight', 0)
            total += (w_u + w_v)
        return total

    # build the projection
    P = generic_weighted_projected_graph(
        G,
        senator_nodes,
        weight_function=weight_fn
    )

    # detect Louvain communities (it will look at P[u][v]['weight'])
    communities = community.louvain_communities(P, weight='weight', seed=42)

    # flatten into a node→community index map
    membership = {
        node: idx
        for idx, comm in enumerate(communities)
        for node in comm
    }

    # Calculate modularity score
    modularity = community.modularity(P, communities, weight='weight')
    print(f"Bipartite projection modularity: {modularity:.4f}")

    return P, communities, membership


def analyze_centrality_safely(G):
    """
    Calculate centrality metrics for the graph, handling disconnected components.
    """
    # Get connected components
    components = list(nx.connected_components(G))
    print(f"Graph has {len(components)} connected components")
    
    # Initialize centrality dict
    centrality = {}
    
    # For each component with more than 1 node, calculate eigenvector centrality
    for i, component in enumerate(components):
        if len(component) > 1:  # Skip isolated nodes
            subgraph = G.subgraph(component).copy()
            try:
                # Calculate eigenvector centrality for this component
                comp_centrality = nx.eigenvector_centrality_numpy(subgraph, weight='weight')
                # Add to overall centrality dict
                centrality.update(comp_centrality)
            except nx.PowerIterationFailedConvergence:
                print(f"Warning: Eigenvector centrality failed to converge for component {i}")
        else:
            # For isolated nodes, assign zero centrality
            node = list(component)[0]
            centrality[node] = 0.0
    
    return centrality


def visualize_bipartite_graph(G, senator_nodes, ticker_nodes, title):
    """
    Plot the bipartite graph, coloring purchase edges green and sale edges red,
    and scaling edge widths by transaction amount.
    """
    plt.figure(figsize=(12, 10))
    pos = nx.bipartite_layout(G, senator_nodes)

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(senator_nodes),
        node_color='lightblue',
        node_size=300,
        alpha=0.8,
        label='Senators'
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(ticker_nodes),
        node_color='lightgreen',
        node_size=200,
        alpha=0.8,
        label='Tickers'
    )

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    min_w, max_w = min(weights), max(weights)
    min_width, max_width = 0.4, 6.0

    edge_width = {}
    for u, v, d in G.edges(data=True):
        if max_w > min_w:
            norm = (d['weight'] - min_w) / (max_w - min_w)
        else:
            norm = 1.0
        edge_width[(u, v)] = min_width + norm * (max_width - min_width)

    purchase_edges = [(u, v) for u, v, d in G.edges(data=True) if d['transaction'] == 'purchase']
    sale_edges     = [(u, v) for u, v, d in G.edges(data=True) if d['transaction'] == 'sale']

    nx.draw_networkx_edges(
        G, pos,
        edgelist=purchase_edges,
        edge_color='green',
        width=[edge_width[(u, v)] for u, v in purchase_edges],
        alpha=0.6
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=sale_edges,
        edge_color='red',
        width=[edge_width[(u, v)] for u, v in sale_edges],
        alpha=0.6
    )

    nx.draw_networkx_labels(G, pos, font_size=8)
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Purchase'),
        Line2D([0], [0], color='red',   lw=2, label='Sale'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def save_graph_analysis(G, communities, P, senators, tickers, filename="graph_analysis.txt"):
    """
    Save the analysis results to a text file.
    """
    with open(filename, 'w') as f:
        f.write("Bipartite Graph Analysis Results\n")
        f.write("===============================\n\n")
        
        f.write(f"Graph Statistics:\n")
        f.write(f"- Senators: {len(senators)}\n")
        f.write(f"- Tickers: {len(tickers)}\n")
        f.write(f"- Edges in bipartite graph: {G.number_of_edges()}\n")
        f.write(f"- Edges in senator projection: {P.number_of_edges()}\n\n")
        
        f.write(f"Communities:\n")
        for i, comm in enumerate(communities):
            f.write(f"Community {i} (size={len(comm)}):\n")
            f.write(f"{', '.join(sorted(comm))}\n\n")
        
        # Calculate and report centrality metrics
        eig_cent = analyze_centrality_safely(P)
        
        # Create DataFrame for easy sorting and display
        df = pd.DataFrame({
            'senator': list(eig_cent.keys()),
            'eigenvector': list(eig_cent.values()),
        })
        
        f.write(f"Top Senators by Eigenvector Centrality:\n")
        f.write(
            df
            .sort_values('eigenvector', ascending=False)
            .head(10)
            .to_string(index=False)
        )


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir   = script_dir.parent / "data/senator-transactions"

    G, senators, tickers = create_bipartite_graph(data_dir)
    print(f"Original graph: {len(senators)} senators, {len(tickers)} tickers, {G.number_of_edges()} edges")

    H, filtered_senators, filtered_tickers = filter_by_degree(G, senators, tickers, min_degree=10)
    print(f"Filtered graph: {len(filtered_senators)} senators, {len(filtered_tickers)} tickers, {H.number_of_edges()} edges")

    visualize_bipartite_graph(H, filtered_senators, filtered_tickers, "Filtered Bipartite")

    # For community detection, use the ORIGINAL graph with ALL senators
    # P, communities, membership = get_communities(G, filtered_senators) OLD CODE
    P, communities, membership = get_communities(G, senators)
    
    # Save analysis to file
    save_graph_analysis(G, communities, P, senators, tickers, "graph_analysis.txt")

    
    # print summary
    for i, comm in enumerate(communities):
        print(f"Community {i} (size={len(comm)}): {comm}")

    # visualize
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(P, seed=42)
    colors = [membership[node] for node in P.nodes()]
    nx.draw_networkx_nodes(P, pos, node_color=colors, cmap=plt.cm.tab20, node_size=200, alpha=0.9)
    nx.draw_networkx_edges(P, pos, alpha=0.3)
    nx.draw_networkx_labels(P, pos, font_size=7)
    plt.title("Senator-Senator Projection, Colored by Louvain Community")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("senator_connections.png", dpi=300)
    plt.close()
    
    # Ticker projection analysis
    P_tickers = weighted_projected_graph(H, filtered_tickers)

    eig_cent_t = nx.eigenvector_centrality_numpy(P_tickers, weight='weight')

    df_t = pd.DataFrame({
        'ticker':      list(P_tickers.nodes()),
        'eigenvector': [eig_cent_t[n]   for n in P_tickers.nodes()],
    })

    # Save ticker analysis to file
    with open("graph_analysis.txt", "a") as f:
        f.write("\n\nTop 10 tickers by eigenvector centrality:\n")
        f.write(
            df_t
            .sort_values('eigenvector', ascending=False)
            .head(10)
            .to_string(index=False)
        )
    
    # Visualize ticker connections
    plt.figure(figsize=(10,8))
    pos_t = nx.spring_layout(P_tickers, seed=42)
    nx.draw_networkx_nodes(P_tickers, pos_t, node_color='lightgreen', node_size=200, alpha=0.9)
    nx.draw_networkx_edges(P_tickers, pos_t, alpha=0.3)
    nx.draw_networkx_labels(P_tickers, pos_t, font_size=7)
    plt.title("Ticker-Ticker Projection")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("ticker_connections.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
