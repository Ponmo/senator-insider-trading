#!/usr/bin/env python3
import os
import csv
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

def load_senator_data(filepath):
    """
    Load senator stock transaction data from a CSV file.
    Returns the senator name and a list of (ticker, transaction_type) tuples.
    """
    senator_name = Path(filepath).stem
    transactions = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('Ticker', '').strip()
            txn_raw = row.get('Transaction Type', '').strip().lower()
            # skip missing or placeholder tickers
            if not ticker or ticker == '--':
                continue
            if not txn_raw:
                continue

            # Classify transaction
            if txn_raw.startswith('sale'):
                txn_type = 'sale'
            elif txn_raw.startswith('purchase') or txn_raw.startswith('buy'):
                txn_type = 'purchase'
            else:
                # skip other transaction types
                continue

            transactions.append((ticker, txn_type))

    return senator_name, transactions


def create_bipartite_graph(data_dir):
    """
    Create a bipartite graph between senators and stock tickers, tagging edges by transaction type.
    """
    G = nx.Graph()
    senator_nodes = set()
    ticker_nodes = set()

    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(data_dir, filename)
        senator, trans_list = load_senator_data(filepath)

        # only add senators that have valid transactions
        if not trans_list:
            continue
        G.add_node(senator, bipartite=0)
        senator_nodes.add(senator)

        for ticker, txn_type in trans_list:
            G.add_node(ticker, bipartite=1)
            ticker_nodes.add(ticker)
            G.add_edge(senator, ticker, transaction=txn_type)

    return G, senator_nodes, ticker_nodes


def filter_by_degree(G, senator_nodes, ticker_nodes, min_degree):
    """
    Filter the bipartite graph to include only tickers with degree >= min_degree
    and the senators connected to them.
    """
    # compute degree for each ticker
    ticker_deg = {t: G.degree(t) for t in ticker_nodes}
    # select tickers above threshold
    filtered_tickers = {t for t, deg in ticker_deg.items() if deg >= min_degree}

    # find senators connected to those tickers
    filtered_senators = {
        nbr for t in filtered_tickers for nbr in G.neighbors(t)
        if nbr in senator_nodes
    }

    # induce subgraph
    sub_nodes = filtered_senators.union(filtered_tickers)
    H = G.subgraph(sub_nodes).copy()
    return H, filtered_senators, filtered_tickers


def visualize_bipartite_graph(G, senator_nodes, ticker_nodes, title):
    """
    Plot the bipartite graph, coloring purchase edges green and sale edges red.
    """
    plt.figure(figsize=(12, 10))
    pos = nx.bipartite_layout(G, senator_nodes)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(senator_nodes), node_color='lightblue',
                           node_size=300, alpha=0.8, label='Senators')
    nx.draw_networkx_nodes(G, pos, nodelist=list(ticker_nodes), node_color='lightgreen',
                           node_size=200, alpha=0.8, label='Tickers')

    # Separate edges by transaction type
    purchase_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('transaction') == 'purchase']
    sale_edges     = [(u, v) for u, v, d in G.edges(data=True) if d.get('transaction') == 'sale']

    nx.draw_networkx_edges(G, pos, edgelist=purchase_edges, edge_color='green', alpha=0.6, width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=sale_edges,     edge_color='red',   alpha=0.6, width=1.5)

    # Labels and legend
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
    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir   = script_dir.parent / "data"

    G, senators, tickers = create_bipartite_graph(data_dir)
    print(f"Original graph: {len(senators)} senators, {len(tickers)} tickers, {G.number_of_edges()} edges")

    H, filtered_senators, filtered_tickers = filter_by_degree(G, senators, tickers, min_degree=3)
    print(f"Filtered graph: {len(filtered_senators)} senators, {len(filtered_tickers)} tickers, {H.number_of_edges()} edges")

    visualize_bipartite_graph(H, filtered_senators, filtered_tickers, "Filtered Bipartite")

if __name__ == '__main__':
    main()
