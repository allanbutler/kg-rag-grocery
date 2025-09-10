import pandas as pd, networkx as nx, os, pickle
from sgs.config import settings

def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in df.iterrows():
        pid = f"product:{r.product_id}"
        G.add_node(pid, label="Product", name=r.name, brand=r.brand, category=r.category, sub_category=r.sub_category, price=float(r.price))
        b = f"brand:{r.brand}"; c = f"category:{r.category}"; sc = f"subcat:{r.sub_category}"
        G.add_node(b, label="Brand", name=r.brand); G.add_node(c, label="Category", name=r.category); G.add_node(sc, label="SubCategory", name=r.sub_category)
        G.add_edge(pid, b, type="MADE_BY"); G.add_edge(pid, sc, type="IN_SUBCATEGORY"); G.add_edge(sc, c, type="IN_CATEGORY")
        for ing in str(r.ingredients).split(","):
            ing = ing.strip().lower();  n = f"ing:{ing}"
            if ing: G.add_node(n, label="Ingredient", name=ing); G.add_edge(pid, n, type="HAS_INGREDIENT")
        for att in str(r.attributes).split(";"):
            att = att.strip().lower();  n = f"attr:{att}"
            if att: G.add_node(n, label="Attribute", name=att); G.add_edge(pid, n, type="HAS_ATTRIBUTE")
    return G

def main():
    df = pd.read_csv(settings.data_csv)
    G = build_graph(df)
    os.makedirs(os.path.dirname(settings.graph_path), exist_ok=True)
    with open(settings.graph_path, "wb") as f: pickle.dump(G, f)
    print(f"Saved graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {settings.graph_path}")

if __name__ == "__main__":
    main()
