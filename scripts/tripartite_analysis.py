#!/usr/bin/env python3
import csv
import requests
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def load_senator_data(filepath):
    """
    Load one senator's stock tickers from CSV.
    Returns: (senator_name, set of tickers)
    """
    senator = Path(filepath).stem
    tickers = set()
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get('Ticker', '').strip()
            if t and t != '--':
                tickers.add(t)
    return senator, tickers

def build_bipartite_stock_graph(data_dir):
    """
    Create G with nodes(layer='senator' or 'stock') and edges senator–ticker.
    """
    G = nx.Graph()
    senators, stocks = set(), set()
    for csvfile in Path(data_dir).glob("*.csv"):
        sen, ticks = load_senator_data(csvfile)
        G.add_node(sen, layer='senator')
        senators.add(sen)
        for t in ticks:
            G.add_node(t, layer='stock')
            stocks.add(t)
            G.add_edge(sen, t)
    return G, senators, stocks

def fetch_all_votes(congress=118):
    """
    Get all Senate votes that passed in the given Congress.
    Returns a list of vote objects (each has 'bill' and 'voters').
    """
    base = "https://www.govtrack.us/api/v2/vote"
    params = {
        "chamber": "senate",
        "congress": congress,
        "limit":   100
    }
    votes = []
    while True:
        r = requests.get(base, params=params)
        r.raise_for_status()
        data = r.json()
        votes.extend(data.get("objects", []))
        nxt = data.get("meta", {}).get("next")
        if not nxt:
            break
        # GovTrack returns 'next' as "/api/v2/vote?offset=100…"
        base = "https://www.govtrack.us" + nxt
        params.clear()
    return votes

def fetch_senator_name_map(congress=118):
    """
    Map each current senator's bioguide_id → full name, via GovTrack /role.
    """
    url = "https://www.govtrack.us/api/v2/role"
    params = {
        "current":   "true",
        "role_type": "senator",
        "limit":     750
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    mapping = {}
    for role in r.json().get("objects", []):
        person = role.get("person", {})
        bgid   = person.get("bioguideid")
        name   = person.get("name")
        if bgid and name:
            mapping[bgid] = name
    return mapping

def visualize_tripartite(G, senators, stocks, bills, out_png):
    """
    Position: x = 0 senators, x = 1 stocks, x = -1 bills; y spaced by index.
    """
    layers = {'senator': 0, 'stock': 1, 'bill': -1}
    pos = {}
    for layer, nodes in (('senator', senators),
                         ('stock', stocks),
                         ('bill', bills)):
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (layers[layer], -i)

    plt.figure(figsize=(12,12))
    nx.draw_networkx_nodes(G, pos, nodelist=senators, node_color='lightblue',
                           node_size=300, label='Senators')
    nx.draw_networkx_nodes(G, pos, nodelist=stocks,   node_color='lightgreen',
                           node_size=200, label='Stocks')
    nx.draw_networkx_nodes(G, pos, nodelist=bills,    node_color='lightcoral',
                           node_size=100, label='Bills')
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.0)
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    # locate your data/
    script_dir = Path(__file__).resolve().parent
    data_dir   = script_dir.parent / "data"

    # 1) build senator–stock layer
    G, senators, stocks = build_bipartite_stock_graph(data_dir)

    # 2) fetch passed votes and name map
    all_votes = fetch_all_votes(congress=118)
    name_map   = fetch_senator_name_map(congress=118)
    
    passed_votes = [v for v in all_votes if v.get("passed") is True]

    # 3) add each bill and link every Yea/Nay vote
    bills = set()
    for v in passed_votes:
        bill = v.get("bill", {}).get("bill_id")
        if not bill:
            continue
        bills.add(bill)
        G.add_node(bill, layer='bill')
        voters = v.get("voters", {})
        for bgid, info in voters.items():
            vote_value = info.get("vote")
            if vote_value in ("Yea", "Nay"):
                sen = name_map.get(bgid)
                if sen in senators:
                    G.add_edge(sen, bill, vote=vote_value)

    # 4) save the tripartite layout
    visualize_tripartite(G, senators, stocks, bills, "tripartite_graph.png")
    print("Saved tripartite_graph.png")

if __name__ == "__main__":
    main()
