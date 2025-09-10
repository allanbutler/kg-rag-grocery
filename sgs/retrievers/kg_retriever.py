import pickle, re
from typing import List, Dict
import networkx as nx
from sgs.config import settings

def _load_graph() -> nx.Graph:
    with open(settings.graph_path, "rb") as f:
        return pickle.load(f)

def kg_search(query: str, k: int = 8) -> List[Dict]:
    G = _load_graph()
    tokens = re.findall(r"[a-zA-Z0-9%]+", query.lower())
    hits, contexts, products = set(), [], set()
    for n, data in G.nodes(data=True):
        if any(t in str(data.get("name","")).lower() for t in tokens):
            hits.add(n)
    for h in list(hits)[:20]:
        for nbr in G.neighbors(h):
            ndata = G.nodes[nbr]; edata = G.get_edge_data(h, nbr)
            if ndata.get("label") == "Product": products.add(nbr)
            contexts.append(f"{G.nodes[h].get('label')}({G.nodes[h].get('name')}) -[{edata.get('type')}]-> {ndata.get('label')}({ndata.get('name')})")
    cards = []
    for p in list(products)[:k]:
        d = G.nodes[p]
        cards.append({"product_id": p.split(":",1)[1], "name": d["name"], "brand": d["brand"], "category": d["category"], "sub_category": d["sub_category"], "price": d["price"]})
    return [{"type":"graph_fact","text": c} for c in contexts[:50]] + [{"type":"product","payload": pc} for pc in cards]
