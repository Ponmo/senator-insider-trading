# Senator Stock Network Analysis

This project analyzes the relationships between US Senators based on their stock trading patterns and voting behaviors.

## Overview

The analysis consists of three main components:

1. **Bipartite Analysis**: Examines senator-stock relationships through a bipartite graph
2. **Embedding Analysis**: Analyzes senator voting patterns using bill embeddings
3. **Community Comparison**: Compares the community structures from both analyses

## Setup

1. Clone this repository
2. Install the required dependencies:

## Running the Analysis

You can run the entire workflow with a single command:

This will:

1. Run the bipartite analysis of stock trading patterns
2. Run the embedding analysis of voting patterns
3. Compare the community structures from both analyses
4. Generate visualizations and text reports

## Key Output Files

- `graph_analysis.txt` - Results from bipartite stock trading analysis
- `senator_embedding_analysis.txt` - Results from voting pattern analysis
- `community_comparison.txt` - Comparison between the two community structures
- `community_comparison.png` - Visualization of community overlap

## Individual Scripts

If you want to run the analyses separately:

- `scripts/bipartite_analysis_test.py` - Stock trading network analysis
- `scripts/embedding_graph.py` - Voting pattern network analysis
- `scripts/compare_communities.py` - Community structure comparison

## Data Sources

- `data/senator-transactions/` - Senator stock transaction data
- `data/senator-bill-embeddings/` - Senator voting pattern embeddings

```

```
