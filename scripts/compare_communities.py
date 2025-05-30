#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx import community
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from collections import defaultdict

# Import the necessary modules from our existing scripts
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from embedding_graph import load_senator_embeddings, create_similarity_graph, get_communities
from bipartite_analysis_test import create_bipartite_graph, filter_by_degree, get_communities as get_bipartite_communities
senator_name_map = {
    "A. Mitchell McConnell Jr.": "Mitch McConnell",
    "Jacklyn S Rosen": "Jacklyn Rosen",
    "John W Hickenlooper": "John Hickenlooper",
    "Mike Rounds": "Mike Rounds",
    "Sheldon Whitehouse": "Sheldon Whitehouse",
    "Angus S King Jr.": "Angus King",
    "James Banks": "Jim Banks",
    "Joseph Manchin III": "Joseph Manchin",
    "Pat Roberts": "Pat Roberts",
    "Shelley M Capito": "Shelley Capito",
    "Ashley Moody": "Ashley Moody",
    "James M Inhofe": "Jim Inhofe",
    "Katie Britt": "Katie Britt",
    "Patrick J Toomey": "Patrick Toomey",
    "Steve Daines": "Steve Daines",
    "Benjamin L Cardin": "Ben Cardin",
    "Jeanne Shaheen": "Jeanne Shaheen",
    "Kelly Loeffler": "Kelly  Loeffler",
    "Patty Murray": "Patty Murray",
    "Susan M Collins": "Susan Collins",
    "Bernie Moreno": "Bernie Moreno",
    "Jefferson B Sessions III": "Jefferson Sessions",
    "Ladda Tammy Duckworth": "Tammy Duckworth",
    "Rafael E Cruz": "Ted Cruz",
    "Tammy Duckworth": "Tammy Duckworth",
    "Chris Van Hollen": "Chris Van Hollen",
    "Jerry Moran": "Jerry Moran",
    "Lamar Alexander": "Lamar Alexander",
    "Rand Paul": "Rand Paul",
    "Thomas H Tuberville": "Tommy Tuberville",
    "Christopher A Coons": "Christopher Coons",
    "John A Barrasso": "John Barrasso",
    "Lindsey Graham": "Lindsey Graham",
    "Rick Scott": "Rick Scott",
    "Thomas R Carper": "Thomas Carper",
    "Cory A Booker": "Cory Booker",
    "John Boozman": "John Boozman",
    "Marco Rubio": "Marco Rubio",
    "Robert J Portman": "Rob Portman",
    "Thomas R Tillis": "Thomas Tillis",
    "Cynthia M Lummis": "Cynthia Lummis",
    "John Cornyn": "John Cornyn",
    "Maria Cantwell": "Maria Cantwell",
    "Robert P Casey Jr.": "Bob Casey",
    "Thomas Udall": "Tom Udall",
    "Daniel S Sullivan": "Dan Sullivan",
    "John F Reed": "John Reed",
    "Mark E Kelly": "Mark Kelly",
    "Roger F Wicker": "Roger Wicker",
    "Timothy M Kaine": "Timothy Kaine",
    "David A Perdue  Jr": "David Perdue",
    "John Fetterman": "John Fetterman",
    "Mark R Warner": "Mark Warner",
    "Roger W Marshall": "Roger Marshall",
    "Tina Smith": "Tina Smith",
    "David H McCormick": "David McCormick",
    "John Hoeven": "John Hoeven",
    "Markwayne Mullin": "Markwayne Mullin",
    "Ron Johnson": "Ron Johnson",
    "William Cassidy": "Bill Cassidy",
    "Debra S Fischer": "Deb Fischer",
    "John N Kennedy": "John Kennedy",
    "Michael  B Enzi": "Michael Enzi",
    "Ron L Wyden": "Ron Wyden",
    "William F Hagerty IV": "Bill Hagerty",
    "Gary C Peters": "Gary Peters",
    "John P Ricketts": "Pete Ricketts",
    "Michael D Crapo": "Michael Crapo",
    "Ron Wyden": "Ron Wyden",
    "JD Vance": "J. Vance",
    "John R Thune": "John Thune",
    "Michael F Bennet": "Michael Bennet",
    "Roy Blunt": "Roy Blunt"
}


def load_bipartite_communities():
    """
    Run the bipartite analysis to get the communities from stock trading patterns.
    """
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data/senator-transactions"
    
    # Create bipartite graph with ALL senators
    G, senators, tickers = create_bipartite_graph(data_dir)
    
    # Get communities using ALL senators
    P, communities, membership = get_bipartite_communities(G, senators)
    
    return P, communities, membership, senators


def load_embedding_communities():
    """
    Run the embedding analysis to get the communities from voting patterns.
    """
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data/senator-bill-embeddings"
    
    # Load senator embeddings
    embeddings = load_senator_embeddings(data_dir)
    
    # Create similarity graph
    G = create_similarity_graph(embeddings, threshold=0.7)
    
    # Get communities
    communities, membership, modularity = get_communities(G)
    
    return G, communities, membership, set(G.nodes())


def map_senator_names(name, reverse=False):
    """
    Map senator names between the two datasets.
    If reverse=True, map from embedding format to transaction format.
    """
    if reverse:
        # Find key by value
        for k, v in senator_name_map.items():
            if v == name:
                return k
        return name
    else:
        # Map from transaction format to embedding format
        return senator_name_map.get(name, name)


def calculate_mutual_information(bipartite_membership, embedding_membership):
    """
    Calculate mutual information between two community partitions.
    """
    # Find common senators between the two partitions
    common_senators = set(bipartite_membership.keys()) & set(embedding_membership.keys())
    
    if not common_senators:
        return 0.0, 0.0, []
    
    # Create lists of community assignments for common senators
    bipartite_labels = [bipartite_membership[senator] for senator in common_senators]
    embedding_labels = [embedding_membership[senator] for senator in common_senators]
    
    # Calculate normalized mutual information
    nmi = normalized_mutual_info_score(bipartite_labels, embedding_labels)
    
    # Calculate adjusted mutual information (adjusts for chance)
    ami = adjusted_mutual_info_score(bipartite_labels, embedding_labels)
    
    return nmi, ami, list(common_senators)


def create_comparison_table(bipartite_membership, embedding_membership, common_senators):
    """
    Create a table comparing community assignments for common senators.
    """
    data = []
    for senator in common_senators:
        data.append({
            'Senator': senator,
            'Stock Trading Community': bipartite_membership.get(senator, -1),
            'Voting Pattern Community': embedding_membership.get(senator, -1)
        })
    
    return pd.DataFrame(data)


def calculate_modularity_scores(bipartite_graph, bipartite_communities, embedding_graph, embedding_communities):
    """
    Calculate and return modularity scores for both community partitions.
    """
    bipartite_modularity = community.modularity(bipartite_graph, bipartite_communities, weight='weight')
    embedding_modularity = community.modularity(embedding_graph, embedding_communities, weight='weight')
    
    return bipartite_modularity, embedding_modularity


def main():
    # Load communities from bipartite analysis (stock trading)
    bipartite_graph, bipartite_communities, bipartite_membership, bipartite_senators = load_bipartite_communities()
    
    # Load communities from embedding analysis (voting patterns)
    embedding_graph, embedding_communities, embedding_membership, embedding_senators = load_embedding_communities()
    
    # Map senator names to be consistent between the two datasets
    mapped_bipartite_membership = {}
    for senator, community in bipartite_membership.items():
        mapped_name = map_senator_names(senator)
        mapped_bipartite_membership[mapped_name] = community
    
    # Calculate mutual information
    nmi, ami, common_senators = calculate_mutual_information(
        mapped_bipartite_membership, 
        embedding_membership
    )
    
    # Calculate modularity scores
    bipartite_modularity, embedding_modularity = calculate_modularity_scores(
        bipartite_graph, bipartite_communities, 
        embedding_graph, embedding_communities
    )
    
    # Create comparison table
    comparison_df = create_comparison_table(
        mapped_bipartite_membership,
        embedding_membership,
        common_senators
    )
    
    # Save results to file
    with open("community_comparison.txt", "w") as f:
        f.write("Community Structure Comparison: Stock Trading vs. Voting Patterns\n")
        f.write("==================================================================\n\n")
        
        f.write(f"Number of senators in stock trading analysis: {len(bipartite_senators)}\n")
        f.write(f"Number of senators in voting pattern analysis: {len(embedding_senators)}\n")
        f.write(f"Number of senators in both analyses: {len(common_senators)}\n\n")
        
        f.write(f"Normalized Mutual Information (NMI): {nmi:.4f}\n")
        
        f.write(f"Modularity Scores:\n")
        f.write(f"- Stock Trading Communities: {bipartite_modularity:.4f}\n")
        f.write(f"- Voting Pattern Communities: {embedding_modularity:.4f}\n\n")
        
        f.write("Interpretation:\n")
        f.write("- NMI ranges from 0 (no mutual information) to 1 (perfect correlation)\n")
        f.write("- Modularity measures the quality of the community structure (higher is better)\n\n")
        
        f.write("Community Assignments for Common Senators:\n")
        f.write(comparison_df.to_string(index=False))
        
        f.write("\n\nStock Trading Communities:\n")
        for i, comm in enumerate(bipartite_communities):
            mapped_comm = [map_senator_names(senator) for senator in comm]
            f.write(f"Community {i} (size={len(comm)}): {sorted(mapped_comm)}\n")
        
        f.write("\n\nVoting Pattern Communities:\n")
        for i, comm in enumerate(embedding_communities):
            f.write(f"Community {i} (size={len(comm)}): {sorted(comm)}\n")
    
    print(f"Results saved to community_comparison.txt")
    
    # Create a visualization of the comparison
    plt.figure(figsize=(12, 10))
    
    # Create a mapping of senators to positions
    common_senators_list = sorted(common_senators)
    positions = {senator: i for i, senator in enumerate(common_senators_list)}
    
    # Create a matrix for visualization
    matrix = np.zeros((len(embedding_communities), len(bipartite_communities)))
    
    # Fill the matrix with counts of senators in each community pair
    for senator in common_senators:
        stock_comm = mapped_bipartite_membership.get(senator, -1)
        vote_comm = embedding_membership.get(senator, -1)
        if stock_comm >= 0 and vote_comm >= 0:
            matrix[vote_comm, stock_comm] += 1
    
    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        if row_sums[i] > 0:
            normalized_matrix[i] = matrix[i] / row_sums[i]
    
    # Plot heatmap
    plt.imshow(normalized_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Proportion of Senators')
    plt.xlabel('Stock Trading Communities')
    plt.ylabel('Voting Pattern Communities')
    plt.title('Community Overlap Between Stock Trading and Voting Patterns')
    
    # Add x and y ticks
    plt.xticks(range(len(bipartite_communities)), range(len(bipartite_communities)))
    plt.yticks(range(len(embedding_communities)), range(len(embedding_communities)))
    
    plt.tight_layout()
    plt.savefig('community_comparison.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()