#!/usr/bin/env python3
"""
Workflow Coordinator for Senator Network Analysis

This script coordinates the execution of the bipartite analysis, embedding analysis,
and community comparison to ensure consistent data flow and zero manual intervention.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_script(script_name):
    """Run a Python script and return its exit code."""
    print(f"\n{'='*80}\nRunning {script_name}...\n{'='*80}")
    script_path = Path(__file__).parent / script_name
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    return result.returncode

def check_output_files(files):
    """Check if all output files exist."""
    missing = []
    for file in files:
        if not Path(file).exists():
            missing.append(file)
    return missing

def main():
    # Define the scripts to run in order
    scripts = [
        "bipartite_analysis_test.py",
        "embedding_graph.py",
        "compare_communities.py"
    ]
    
    # Define expected output files
    expected_outputs = {
        "bipartite_analysis_test.py": [
            "filtered_bipartite.png",
            "senator_connections.png",
            "ticker_connections.png",
            "graph_analysis.txt"
        ],
        "embedding_graph.py": [
            "senator_embedding_analysis.txt",
            "senator_similarity_matrix.png",
            "senator_voting_similarity_network.png"
        ],
        "compare_communities.py": [
            "community_comparison.txt",
            "community_comparison.png"
        ]
    }
    
    # Run each script and validate outputs
    for script in scripts:
        # Run the script
        exit_code = run_script(script)
        
        if exit_code != 0:
            print(f"Error: {script} failed with exit code {exit_code}")
            sys.exit(exit_code)
        
        # Check for expected output files
        missing_files = check_output_files(expected_outputs[script])
        if missing_files:
            print(f"Error: The following expected output files are missing after running {script}:")
            for file in missing_files:
                print(f"  - {file}")
            sys.exit(1)
        
        print(f"Successfully completed {script} with all expected outputs")
    
    print("\n" + "="*80)
    print("Workflow completed successfully!")
    print("All analyses have been run and outputs generated.")
    print("="*80)
    
    # Summarize the key output files
    print("\nKey output files:")
    print("1. graph_analysis.txt - Results from bipartite stock trading analysis")
    print("2. senator_embedding_analysis.txt - Results from voting pattern analysis")
    print("3. community_comparison.txt - Comparison between the two community structures")
    print("4. community_comparison.png - Visualization of community overlap")

if __name__ == "__main__":
    main()