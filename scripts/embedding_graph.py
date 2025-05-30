#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms import community
from matplotlib.lines import Line2D

# Import the senator name mapping
# Manually created map from senator names in ../data/senator-transactions to senator names in ../data/senator-bill-embeddings
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

def load_senator_embeddings(data_dir):
    """
    Load senator embeddings from JSON files in the data directory.
    Returns a dictionary mapping senator names to their embeddings.
    """
    embeddings = {}
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        senator_name = Path(filename).stem
        
        # Check if this senator is in our mapping or already in the correct format
        mapped_name = None
        
        # Check if the name is a value in the mapping (already in the correct format)
        if senator_name in senator_name_map.values():
            mapped_name = senator_name
        # Check if the name is a key in the mapping (needs to be converted)
        elif senator_name in senator_name_map:
            mapped_name = senator_name_map[senator_name]
        
        # Skip senators that aren't in our mapping
        if mapped_name is None:
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract the embedding vector for "yea" votes
            if "yea" in data and "embedding" in data["yea"]:
                embeddings[mapped_name] = np.array(data["yea"]["embedding"])
    
    print(f"Loaded embeddings for {len(embeddings)} senators")
    return embeddings


def create_similarity_graph(embeddings, threshold=0.0):
    """
    Create a weighted graph where:
    - Nodes are senators
    - Edges connect senators with cosine similarity above threshold
    - Edge weights are the cosine similarity values
    """
    G = nx.Graph()
    
    # Add all senators as nodes
    for senator in embeddings.keys():
        G.add_node(senator)
    
    # Calculate pairwise similarities and add edges
    senators = list(embeddings.keys())
    for i, senator1 in enumerate(senators):
        for senator2 in senators[i+1:]:  # Avoid duplicate pairs and self-loops
            emb1 = embeddings[senator1].reshape(1, -1)
            emb2 = embeddings[senator2].reshape(1, -1)
            
            # Calculate cosine similarity
            sim = cosine_similarity(emb1, emb2)[0][0]
            
            # Add edge if similarity is above threshold
            if sim > threshold:
                G.add_edge(senator1, senator2, weight=sim)
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def get_communities(G):
    """
    Detect communities in the graph using the Louvain algorithm.
    Returns the communities and a mapping of nodes to community IDs.
    """
    # Detect Louvain communities with higher resolution for more fine-grained communities
    communities = community.louvain_communities(G, weight='weight', resolution=1.04, seed=42)
    
    # Create a mapping from node to community ID
    membership = {
        node: idx
        for idx, comm in enumerate(communities)
        for node in comm
    }
    
    # Calculate modularity of the partition
    modularity = community.modularity(G, communities, weight='weight')

    print("Number of communities embeddings:" , len(communities))
    
    return communities, membership, modularity


def visualize_similarity_graph(G, membership, title="Senator Similarity Network"):
    """
    Visualize the senator similarity graph with nodes colored by community.
    """
    plt.figure(figsize=(12, 10))
    
    # Use spring layout with seed for reproducibility
    pos = nx.spring_layout(G, seed=42)
    
    # Get edge weights for width scaling
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_w, max_w = min(weights), max(weights)
    
    # Scale edge widths based on weight
    edge_width = [(w - min_w) / (max_w - min_w) * 5 + 0.5 for w in weights]
    
    # Draw nodes colored by community
    colors = [membership[node] for node in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        cmap=plt.cm.tab20,
        node_size=300,
        alpha=0.8
    )
    
    # Draw edges with width proportional to similarity
    nx.draw_networkx_edges(
        G, pos,
        width=edge_width,
        alpha=0.1,
        edge_color='gray'
    )
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Create legend for communities
    unique_communities = sorted(set(membership.values()))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=plt.cm.tab20(i), 
               markersize=10, label=f'Community {i}')
        for i in unique_communities
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def visualize_similarity_matrix(embeddings, filename="senator_similarity_matrix.png"):
    """
    Create a 2D visualization of the similarity matrix between senators.
    """
    # Get list of senators
    senators = list(embeddings.keys())
    n_senators = len(senators)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n_senators, n_senators))
    
    # Fill similarity matrix
    for i, senator1 in enumerate(senators):
        for j, senator2 in enumerate(senators):
            emb1 = embeddings[senator1].reshape(1, -1)
            emb2 = embeddings[senator2].reshape(1, -1)
            similarity_matrix[i, j] = cosine_similarity(emb1, emb2)[0][0]
    
    # Find the actual min and max values in the matrix
    min_sim = np.min(similarity_matrix)
    max_sim = np.max(similarity_matrix)
    
    # Normalize similarity matrix to [0, 1] range based on actual min and max
    # This ensures we use the full color range
    if min_sim != max_sim:  # Avoid division by zero
        normalized_matrix = (similarity_matrix - min_sim) / (max_sim - min_sim)
    else:
        normalized_matrix = np.ones_like(similarity_matrix)  # All values are the same
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    im = plt.imshow(normalized_matrix, cmap='viridis', vmin=0, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, label=f'Normalized Similarity (Original range: [{min_sim:.3f}, {max_sim:.3f}])')
    
    # Add labels
    plt.xticks(range(n_senators), senators, rotation=90, fontsize=8)
    plt.yticks(range(n_senators), senators, fontsize=8)
    
    plt.title("Senator Similarity Matrix (Min-Max Normalized)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_analysis_to_file(G, communities, centrality_df, modularity, filename="senator_embedding_analysis.txt"):
    """
    Save the analysis results to a text file instead of printing them.
    """
    with open(filename, 'w') as f:
        f.write(f"Graph Analysis Results\n")
        f.write(f"====================\n\n")
        
        f.write(f"Graph Statistics:\n")
        f.write(f"- Nodes: {G.number_of_nodes()}\n")
        f.write(f"- Edges: {G.number_of_edges()}\n")
        f.write(f"- Modularity: {modularity:.4f}\n\n")
        
        f.write(f"Communities:\n")
        for i, comm in enumerate(communities):
            f.write(f"Community {i} (size={len(comm)}):\n")
            f.write(f"{', '.join(sorted(comm))}\n\n")
        
        f.write(f"Top Senators by Eigenvector Centrality:\n")
        f.write(
            centrality_df
            .sort_values('eigenvector', ascending=False)
            .to_string(index=False)
        )


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data/senator-bill-embeddings"
    
    # Load senator embeddings
    embeddings = load_senator_embeddings(data_dir)
    
    # Create similarity graph (adjust threshold if needed)
    G = create_similarity_graph(embeddings, threshold=0.7)
    
    # Detect communities
    communities, membership, modularity = get_communities(G)
    
    # Analyze centrality
    centrality_df = analyze_centrality(G)
    
    # Save analysis to file instead of printing
    save_analysis_to_file(G, communities, centrality_df, modularity)
    
    # Visualize the similarity matrix
    visualize_similarity_matrix(embeddings)
    
    # Visualize the graph
    visualize_similarity_graph(G, membership, "Senator Voting Similarity Network")


def analyze_centrality(G):
    """
    Calculate and print centrality metrics for the graph.
    """
    # Calculate eigenvector centrality
    eig_cent = nx.eigenvector_centrality_numpy(G, weight='weight')
    
    # Create DataFrame for easy sorting and display
    df = pd.DataFrame({
        'senator': list(G.nodes()),
        'eigenvector': [eig_cent[n] for n in G.nodes()],
    })
    
    return df


if __name__ == '__main__':
    main()
