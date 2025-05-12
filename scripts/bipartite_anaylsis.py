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
    

    with open(filepath, 'r', encoding='utf-8') as f:
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
    
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), "../data/senator-transactions")):
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
    
    print(ticker_nodes)
    return ticker_graph

def analyze_unusual_trading_patterns(bipartite_graph, senator_nodes, ticker_nodes):
    """
    Analyze unusual trading patterns among senators.
    
    Args:
        bipartite_graph (nx.Graph): Bipartite graph
        senator_nodes (set): Set of senator nodes
        ticker_nodes (set): Set of ticker nodes
    
    Returns:
        dict: Dictionary with various unusual pattern metrics
    """
    results = {}
    
    # 1. Exclusivity Score - senators who trade stocks that few others trade
    exclusivity_scores = {}
    for senator in senator_nodes:
        senator_tickers = set(bipartite_graph.neighbors(senator))
        exclusivity_score = 0
        for ticker in senator_tickers:
            # How many senators trade this ticker?
            ticker_traders = len(list(bipartite_graph.neighbors(ticker)))
            # Tickers traded by fewer senators contribute more to exclusivity
            exclusivity_score += 1.0 / ticker_traders
        
        # Normalize by number of tickers
        if len(senator_tickers) > 0:
            exclusivity_scores[senator] = exclusivity_score / len(senator_tickers)
        else:
            exclusivity_scores[senator] = 0
    
    results['exclusivity_scores'] = exclusivity_scores
    
    # 2. Diversity Score - senators who trade a wide variety of stocks
    diversity_scores = {}
    for senator in senator_nodes:
        diversity_scores[senator] = len(list(bipartite_graph.neighbors(senator)))
    
    results['diversity_scores'] = diversity_scores
    
    # 3. Clustering Coefficient - how interconnected a senator's stocks are
    senator_graph = compute_senator_connections(bipartite_graph, senator_nodes, ticker_nodes)
    clustering_coefficients = nx.clustering(senator_graph, weight='weight')
    results['clustering_coefficients'] = clustering_coefficients
    
    # 4. Eigenvector Centrality - influence in the network
    eigenvector_centrality = nx.eigenvector_centrality_numpy(senator_graph, weight='weight')
    results['eigenvector_centrality'] = eigenvector_centrality
    
    # 5. Betweenness Centrality - senators who bridge different trading communities
    betweenness_centrality = nx.betweenness_centrality(senator_graph, weight='weight')
    results['betweenness_centrality'] = betweenness_centrality
    
    # 6. Unusual Trading Score - composite score
    unusual_trading_scores = {}
    for senator in senator_nodes:
        # Combine metrics (with weights that can be adjusted)
        unusual_score = (
            0.3 * exclusivity_scores.get(senator, 0) +
            0.2 * (diversity_scores.get(senator, 0) / max(max(diversity_scores.values(), default=1), 1)) +
            0.2 * clustering_coefficients.get(senator, 0) +
            0.15 * eigenvector_centrality.get(senator, 0) +
            0.15 * betweenness_centrality.get(senator, 0)
        )
        unusual_trading_scores[senator] = unusual_score
    
    results['unusual_trading_scores'] = unusual_trading_scores
    
    return results

def analyze_rare_ticker_connections(bipartite_graph, senator_nodes, ticker_nodes):
    """
    Identify senators who trade rare combinations of stocks.
    
    Args:
        bipartite_graph (nx.Graph): Bipartite graph
        senator_nodes (set): Set of senator nodes
        ticker_nodes (set): Set of ticker nodes
    
    Returns:
        dict: Dictionary with rare ticker connection metrics
    """
    # Create ticker graph
    ticker_graph = compute_ticker_connections(bipartite_graph, senator_nodes, ticker_nodes)
    
    # For each senator, calculate how unusual their ticker combinations are
    unusual_combination_scores = {}
    
    for senator in senator_nodes:
        senator_tickers = set(bipartite_graph.neighbors(senator))
        if len(senator_tickers) < 2:
            unusual_combination_scores[senator] = 0
            continue
        
        # Calculate average connection weight between this senator's tickers
        total_weight = 0
        pair_count = 0
        
        for ticker1 in senator_tickers:
            for ticker2 in senator_tickers:
                if ticker1 < ticker2 and ticker_graph.has_edge(ticker1, ticker2):
                    total_weight += ticker_graph[ticker1][ticker2]['weight']
                    pair_count += 1
        
        # Lower average weight means more unusual combinations
        if pair_count > 0:
            avg_weight = total_weight / pair_count
            # Invert so higher score means more unusual
            unusual_combination_scores[senator] = 1.0 / (avg_weight + 1)
        else:
            # If no connections, this is extremely unusual
            unusual_combination_scores[senator] = 1.0
    
    return unusual_combination_scores

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
    # plt.show()

# Replace your visualize_connection_graph function with these parameters
def visualize_connection_graph(G, title, node_color='skyblue', with_labels=True, 
                              min_weight=0, k=0.3, figsize=(20, 20), node_size=100, 
                              label_size=10, edge_alpha=0.3, filter_nodes=None):
    """
    Visualize a connection graph with better control over display parameters.
    
    Args:
        G (nx.Graph): Connection graph
        title (str): Plot title
        node_color (str): Color for nodes
        with_labels (bool): Whether to show labels
        min_weight (int): Minimum edge weight to display
        k (float): Spring layout strength parameter
        figsize (tuple): Figure size
        node_size (int): Size of nodes
        label_size (int): Size of labels
        edge_alpha (float): Transparency of edges
        filter_nodes (int): Only show top N nodes by degree
    """
    # Filter graph to only include edges with weight >= min_weight
    filtered_G = nx.Graph()
    for u, v, data in G.edges(data=True):
        if data['weight'] >= min_weight:
            filtered_G.add_edge(u, v, **data)
    
    # Add any isolated nodes that should be displayed
    for node in G.nodes():
        filtered_G.add_node(node)
    
    # Optionally filter to only top N nodes by degree
    if filter_nodes:
        degree_dict = dict(filtered_G.degree())
        top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:filter_nodes]
        top_nodes = [node for node, _ in top_nodes]
        filtered_G = filtered_G.subgraph(top_nodes)
    
    plt.figure(figsize=figsize)
    
    # Get weights for edge thickness
    edge_weights = [filtered_G[u][v]['weight'] for u, v in filtered_G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [1.0 + (3.0 * w / max_weight) for w in edge_weights]
    
    # Generate positions with spring layout (k controls node spacing)
    pos = nx.spring_layout(filtered_G, seed=42, k=k)
    
    # Draw the graph
    nx.draw_networkx_nodes(filtered_G, pos, node_color=node_color, node_size=node_size, alpha=0.8)
    nx.draw_networkx_edges(filtered_G, pos, width=normalized_weights, alpha=edge_alpha)
    
    if with_labels:
        nx.draw_networkx_labels(filtered_G, pos, font_size=label_size)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    # plt.show()

def analyze_community_overlap(senator_graph, ticker_graph):
    """
    Analyze how senators overlap in communities and identify potential collusion.
    
    Args:
        senator_graph (nx.Graph): Graph of senator connections
        ticker_graph (nx.Graph): Graph of ticker connections
    
    Returns:
        dict: Dictionary with community overlap metrics
    """
    from networkx.algorithms.community import louvain_communities
    
    # Get communities
    senator_communities = list(louvain_communities(senator_graph))
    
    # Calculate overlap scores
    overlap_scores = {}
    
    for senator in senator_graph.nodes():
        # Find which community this senator belongs to
        for i, community in enumerate(senator_communities):
            if senator in community:
                community_idx = i
                break
        
        # Calculate how strongly connected this senator is to their community
        community = senator_communities[community_idx]
        total_weight = 0
        possible_connections = len(community) - 1  # Exclude self
        
        for other_senator in community:
            if other_senator != senator and senator_graph.has_edge(senator, other_senator):
                total_weight += senator_graph[senator][other_senator]['weight']
        
        # Calculate community cohesion score
        if possible_connections > 0:
            overlap_scores[senator] = total_weight / possible_connections
        else:
            overlap_scores[senator] = 0
    
    return overlap_scores

def main():
    data_dir = "data/senator-transactions"
    output_file = "graph_analysis.txt"
    
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
    
    # Run advanced analysis
    unusual_patterns = analyze_unusual_trading_patterns(bipartite_graph, senator_nodes, ticker_nodes)
    rare_connections = analyze_rare_ticker_connections(bipartite_graph, senator_nodes, ticker_nodes)
    community_overlap = analyze_community_overlap(senator_graph, ticker_graph)
    
    # Analyze and write results to a text file
    with open(output_file, 'w') as f:
        # Most connected senators
        senator_degrees = sorted([(node, senator_graph.degree(node)) for node in senator_graph.nodes()], 
                                 key=lambda x: x[1], reverse=True)
        f.write("Top connected senators:\n")
        for senator, degree in senator_degrees[:5]:
            f.write(f"{senator}: {degree} connections\n")
        f.write("\n")
        
        # Most connected tickers
        ticker_degrees = sorted([(node, ticker_graph.degree(node)) for node in ticker_graph.nodes()], 
                                key=lambda x: x[1], reverse=True)
        f.write("Top connected tickers:\n")
        for ticker, degree in ticker_degrees[:5]:
            f.write(f"{ticker}: {degree} connections\n")
        f.write("\n")
        
        # Centrality measures for senators
        f.write("Senator centrality measures:\n")
        senator_centrality = nx.degree_centrality(senator_graph)
        for senator, centrality in sorted(senator_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"{senator}: {centrality:.4f}\n")
        f.write("\n")
        
        # Centrality measures for tickers
        f.write("Ticker centrality measures:\n")
        ticker_centrality = nx.degree_centrality(ticker_graph)
        for ticker, centrality in sorted(ticker_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"{ticker}: {centrality:.4f}\n")
        f.write("\n")
        
        # Communities in senator graph
        f.write("Senator communities:\n")
        from networkx.algorithms.community import louvain_communities
        from networkx.algorithms.community import greedy_modularity_communities

        senator_communities = list(louvain_communities(senator_graph))
        for i, community in enumerate(senator_communities):
            f.write(f"Community {i+1}: {', '.join(community)}\n")
        f.write("\n")
        
        # Communities in ticker graph
        f.write("Ticker communities:\n")
        ticker_communities = list(louvain_communities(ticker_graph))
        for i, community in enumerate(ticker_communities):
            f.write(f"Community {i+1}: {', '.join(community)}\n")
        f.write("\n")
        
        # NEW ANALYSIS SECTIONS
        
        # Unusual Trading Patterns
        f.write("=== POTENTIAL CONCERNING TRADING PATTERNS ===\n\n")
        
        # Exclusivity Score - senators trading stocks few others trade
        f.write("Senators trading unusual stocks (Exclusivity Score):\n")
        for senator, score in sorted(unusual_patterns['exclusivity_scores'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{senator}: {score:.4f}\n")
        f.write("\n")
        
        # Diversity Score - senators trading many different stocks
        f.write("Senators trading the most diverse portfolios:\n")
        for senator, score in sorted(unusual_patterns['diversity_scores'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{senator}: {score} different stocks\n")
        f.write("\n")
        
        # Eigenvector Centrality - most influential senators in the network
        f.write("Most influential senators in the trading network (Eigenvector Centrality):\n")
        for senator, score in sorted(unusual_patterns['eigenvector_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{senator}: {score:.4f}\n")
        f.write("\n")
        
        # Betweenness Centrality - senators bridging different trading communities
        f.write("Senators bridging different trading communities (Betweenness Centrality):\n")
        for senator, score in sorted(unusual_patterns['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{senator}: {score:.4f}\n")
        f.write("\n")
        
        # Unusual Trading Score - composite score
        f.write("Senators with most unusual trading patterns (Composite Score):\n")
        for senator, score in sorted(unusual_patterns['unusual_trading_scores'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{senator}: {score:.4f}\n")
        f.write("\n")
        
        # Rare Ticker Connections
        f.write("Senators trading unusual combinations of stocks:\n")
        for senator, score in sorted(rare_connections.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{senator}: {score:.4f}\n")
        f.write("\n")
        
        # Community Overlap
        f.write("Senators with strongest community cohesion (potential coordination):\n")
        for senator, score in sorted(community_overlap.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{senator}: {score:.4f}\n")
        f.write("\n")
        
        # Calculate a "Potential Concern Index"
        f.write("=== POTENTIAL CONCERN INDEX ===\n")
        f.write("(Higher scores may indicate more unusual trading patterns)\n\n")
        
        concern_index = {}
        for senator in senator_nodes:
            # Combine multiple metrics with weights
            index = (
                0.25 * unusual_patterns['unusual_trading_scores'].get(senator, 0) +
                0.25 * rare_connections.get(senator, 0) +
                0.20 * unusual_patterns['betweenness_centrality'].get(senator, 0) +
                0.15 * unusual_patterns['exclusivity_scores'].get(senator, 0) +
                0.15 * community_overlap.get(senator, 0)
            )
            concern_index[senator] = index
        
        for senator, score in sorted(concern_index.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{senator}: {score:.4f}\n")
    

    print(f"Enhanced graph analysis results written to {output_file}")

if __name__ == "__main__":
    main()