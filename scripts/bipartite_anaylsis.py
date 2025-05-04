#!/usr/bin/env python3
import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def load_senator_data(filepath):
    """
    Load senator stock transaction data from CSV files.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        tuple: (senator_name, set of tickers)
    """
    senator_name = Path(filepath).stem
    tickers = set()
    

    with open("../" + filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('Ticker', '').strip()
            if ticker and ticker != '--':  # Skip empty or placeholder tickers
                tickers.add(ticker)
    
    return senator_name, tickers

def create_bipartite_graph(data_dir):
    """
    Create a bipartite graph between senators and stock tickers.
    
    Args:
        data_dir (str): Directory containing senator CSV files
    
    Returns:
        nx.Graph: Bipartite graph with senators and tickers
    """
    G = nx.Graph()
    
    # Process each senator's file
    senator_nodes = set()
    ticker_nodes = set()
    
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), "../data")):
        if not filename.endswith('.csv'):
            continue
        
        filepath = os.path.join(data_dir, filename)
        senator, tickers = load_senator_data(filepath)
        
        # Add senator node
        G.add_node(senator, bipartite=0)
        senator_nodes.add(senator)
        
        # Add ticker nodes and edges
        for ticker in tickers:
            if ticker not in G:
                G.add_node(ticker, bipartite=1)
                ticker_nodes.add(ticker)
            
            # Add edge between senator and ticker
            G.add_edge(senator, ticker)
    
    return G, senator_nodes, ticker_nodes

def compute_senator_connections(G, senator_nodes, ticker_nodes):
    """
    Compute connections between senators using matrix multiplication.
    
    Args:
        G (nx.Graph): Bipartite graph
        senator_nodes (set): Set of senator nodes
        ticker_nodes (set): Set of ticker nodes
    
    Returns:
        nx.Graph: Graph of senator connections
    """
    # Create the adjacency matrix
    biadjacency_matrix = nx.bipartite.biadjacency_matrix(G, row_order=list(senator_nodes), 
                                                       column_order=list(ticker_nodes))
    
    # Multiply A * A^T to get senator-to-senator connections
    senator_matrix = biadjacency_matrix @ biadjacency_matrix.T
    
    # Create senator graph
    senator_graph = nx.Graph()
    senator_list = list(senator_nodes)
    
    for i in range(len(senator_list)):
        for j in range(i+1, len(senator_list)):
            weight = senator_matrix[i, j]
            if weight > 0:
                senator_graph.add_edge(senator_list[i], senator_list[j], weight=int(weight))
    
    return senator_graph

def compute_ticker_connections(G, senator_nodes, ticker_nodes):
    """
    Compute connections between tickers using matrix multiplication.
    
    Args:
        G (nx.Graph): Bipartite graph
        senator_nodes (set): Set of senator nodes
        ticker_nodes (set): Set of ticker nodes
    
    Returns:
        nx.Graph: Graph of ticker connections
    """
    # Create the adjacency matrix
    biadjacency_matrix = nx.bipartite.biadjacency_matrix(G, row_order=list(senator_nodes), 
                                                       column_order=list(ticker_nodes))
    
    # Multiply A^T * A to get ticker-to-ticker connections
    ticker_matrix = biadjacency_matrix.T @ biadjacency_matrix
    
    # Create ticker graph
    ticker_graph = nx.Graph()
    ticker_list = list(ticker_nodes)
    
    for i in range(len(ticker_list)):
        for j in range(i+1, len(ticker_list)):
            weight = ticker_matrix[i, j]
            if weight > 0:
                ticker_graph.add_edge(ticker_list[i], ticker_list[j], weight=int(weight))
    
    return ticker_graph

def visualize_bipartite_graph(G, senator_nodes, ticker_nodes, title):
    """
    Visualize the bipartite graph.
    
    Args:
        G (nx.Graph): Bipartite graph
        senator_nodes (set): Set of senator nodes
        ticker_nodes (set): Set of ticker nodes
        title (str): Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Create position layout
    pos = nx.bipartite_layout(G, senator_nodes)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=senator_nodes, node_color='lightblue', 
                          node_size=500, alpha=0.8, label='Senators')
    nx.draw_networkx_nodes(G, pos, nodelist=ticker_nodes, node_color='lightgreen', 
                          node_size=300, alpha=0.8, label='Tickers')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

def visualize_connection_graph(G, title, node_color='skyblue', with_labels=True):
    """
    Visualize a connection graph.
    
    Args:
        G (nx.Graph): Connection graph
        title (str): Plot title
        node_color (str): Color for nodes
        with_labels (bool): Whether to show labels
    """
    plt.figure(figsize=(12, 10))
    
    # Get weights for edge thickness
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [2.0 * w / max_weight for w in edge_weights]
    
    # Generate positions with spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=300, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.5)
    
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

def main():
    data_dir = "data"
    
    # Create bipartite graph
    bipartite_graph, senator_nodes, ticker_nodes = create_bipartite_graph(data_dir)
    print(f"Bipartite graph created with {len(senator_nodes)} senators and {len(ticker_nodes)} tickers")
    
    # Visualize bipartite graph
    visualize_bipartite_graph(bipartite_graph, senator_nodes, ticker_nodes, "Senator-Stock Bipartite Graph")
    
    # Compute and visualize senator connections
    senator_graph = compute_senator_connections(bipartite_graph, senator_nodes, ticker_nodes)
    print(f"Senator graph created with {senator_graph.number_of_edges()} connections")
    visualize_connection_graph(senator_graph, "Senator Connections", node_color='lightblue')
    
    # Compute and visualize ticker connections
    ticker_graph = compute_ticker_connections(bipartite_graph, senator_nodes, ticker_nodes)
    print(f"Ticker graph created with {ticker_graph.number_of_edges()} connections")
    visualize_connection_graph(ticker_graph, "Ticker Connections", node_color='lightgreen')
    
    # Find most connected senators
    senator_degrees = sorted([(node, senator_graph.degree(node)) for node in senator_graph.nodes()], 
                            key=lambda x: x[1], reverse=True)
    print("\nTop connected senators:")
    for senator, degree in senator_degrees[:5]:
        print(f"{senator}: {degree} connections")
    
    # Find most connected tickers
    ticker_degrees = sorted([(node, ticker_graph.degree(node)) for node in ticker_graph.nodes()], 
                           key=lambda x: x[1], reverse=True)
    print("\nTop connected tickers:")
    for ticker, degree in ticker_degrees[:5]:
        print(f"{ticker}: {degree} connections")

if __name__ == "__main__":
    main()