import pickle
import re
from functools import lru_cache
from typing import List, Dict, Tuple

import networkx as nx

from sgs.config import settings


# -------- Helpers --------

@lru_cache(maxsize=1)
def _load_graph() -> nx.Graph:
    """Load and cache the KG once per process."""
    with open(settings.graph_path, "rb") as f:
        return pickle.load(f)

def _norm(s: str | None) -> str:
    return str(s or "").strip().lower()

def _parse_budget(query: str) -> float | None:
    """
    Extract a budget like 'under $5', '<= 4.50', 'under 3 bucks'.
    Returns a float if we see 'under/</less/budget' context; else None.
    """
    q = query.lower()
    m = re.search(r"\$?\s*(\d+(?:\.\d{1,2})?)", q)
    if not m:
        return None
    amt = float(m.group(1))
    if any(tok in q for tok in ("under", "<", "less", "budget", "max", "≤", "<=")):
        return amt
    return None

ATTR_SYNONYMS: dict[str, list[str]] = {
    "nut_free": ["nut-free", "nut free", "no nuts", "peanut-free", "peanut free"],
    "gluten_free": ["gluten-free", "gluten free"],
    "low_sodium": ["low sodium", "reduced sodium", "less sodium"],
    "low_sugar": ["low sugar", "no sugar", "reduced sugar", "less sugar"],
    "zero_sugar": ["zero sugar", "no sugar", "unsweetened"],
    "high_protein": ["high protein", "protein"],
    "vegan": ["vegan", "plant-based", "plant based"],
    "vegetarian": ["vegetarian"],
    "caffeinated": ["caffeinated", "with caffeine"],
    "kids": ["kids", "for kids", "kid"],
}

def _wanted_attrs(query: str) -> set[str]:
    q = query.lower()
    wants = set()
    for attr, phrases in ATTR_SYNONYMS.items():
        if any(p in q for p in phrases):
            wants.add(attr)
    return wants

def _tokenize(query: str) -> list[str]:
    toks = re.findall(r"[a-z0-9%]+", query.lower())
    # Light stoplist to reduce noisy matches
    stop = {"the", "a", "an", "and", "or", "with", "for", "to", "of", "in", "on", "under", "less"}
    return [t for t in toks if t not in stop]


# -------- Main search --------

def kg_search(query: str, k: int = 8) -> List[Dict]:
    """
    Lightweight KG retrieval:
      1) Find node hits by token overlap on node 'name'
      2) Collect neighboring Product nodes
      3) Score products by attribute match, budget, and token matches
      4) Return compact graph facts + top-k product cards
    """
    G = _load_graph()

    tokens = _tokenize(query)
    wants = _wanted_attrs(query)
    budget = _parse_budget(query)

    hits: set[str] = set()
    for n, data in G.nodes(data=True):
        name = _norm(data.get("name"))
        if not name:
            continue
        if any(t in name for t in tokens):
            hits.add(n)

    # If query has explicit attribute words, seed hits with those attribute nodes too
    for attr in wants:
        node_id = f"attr:{attr}"
        if node_id in G:
            hits.add(node_id)

    # Collect candidate product nodes and context triples
    products: set[str] = set()
    contexts: list[str] = []

    # Include direct product hits as well as neighbors-of-hits
    for h in list(hits)[:50]:
        h_label = G.nodes[h].get("label")
        h_name = G.nodes[h].get("name")
        if h_label == "Product":
            products.add(h)
        # One-hop neighborhood
        for nbr in G.neighbors(h):
            ndata = G.nodes[nbr]
            edata = G.get_edge_data(h, nbr) or {}
            if ndata.get("label") == "Product":
                products.add(nbr)
            # Compact fact line
            contexts.append(
                f"{h_label}({_norm(h_name)}) -[{edata.get('type', '')}]-> "
                f"{ndata.get('label')}({_norm(ndata.get('name'))})"
            )

    # Build product cards with scoring
    candidates: list[Tuple[float, Dict]] = []

    def score_product(pnode: str) -> Tuple[float, Dict]:
        d = G.nodes[pnode]
        name = d.get("name")
        brand = d.get("brand")
        category = d.get("category")
        subcat = d.get("sub_category")
        price = float(d.get("price") or 0.0)

        # Attributes string reconstruction from neighbors (for filtering/explain)
        attrs = []
        for nbr in G.neighbors(pnode):
            if G.nodes[nbr].get("label") == "Attribute":
                attrs.append(_norm(G.nodes[nbr].get("name")))
        attr_str = ";".join(sorted(set(attrs)))

        # Base score
        score = 0.0

        # Attribute matches: +1 each wanted attribute present, small penalty if wanted but missing
        for a in wants:
            if a in attr_str:
                score += 1.0
            else:
                score -= 0.25

        # Budget preference
        if budget is not None:
            if price <= budget:
                score += 0.75
            else:
                score -= 0.75

        # Token matches across facets
        facets = " ".join([_norm(name), _norm(brand), _norm(category), _norm(subcat)])
        if tokens:
            hit_count = sum(1 for t in tokens if t in facets)
            score += min(hit_count * 0.2, 0.8)

        # Prefer lower price slightly (but don’t dominate)
        score += max(0.0, 5.0 - price) * 0.05  # ~+0.25 if price ~0.0

        card = {
            "product_id": int(str(pnode).split(":", 1)[1]) if ":" in str(pnode) else None,
            "name": name,
            "brand": brand,
            "category": category,
            "sub_category": subcat,
            "price": price,
            "attributes": attr_str,
        }
        return score, card

    for p in products:
        try:
            sc, card = score_product(p)
            candidates.append((sc, card))
        except Exception:
            continue

    # Sort by score desc, dedupe by product name, filter by explicit wants/budget
    candidates.sort(key=lambda x: x[0], reverse=True)

    seen_names = set()
    cards: list[Dict] = []
    for sc, c in candidates:
        pname = _norm(c.get("name"))
        if not pname or pname in seen_names:
            continue
        # Hard filters
        if budget is not None and c.get("price", 1e9) > budget:
            continue
        ok = True
        for a in wants:
            if a not in (c.get("attributes") or ""):
                ok = False
                break
        if not ok:
            continue
        seen_names.add(pname)
        cards.append(c)
        if len(cards) >= k:
            break

    # Compact context (cap to 50 lines)
    facts = [{"type": "graph_fact", "text": ctx} for ctx in contexts[:50]]
    product_payloads = [{"type": "product", "payload": pc} for pc in cards]

    return facts + product_payloads
