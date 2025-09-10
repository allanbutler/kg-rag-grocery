import os, json
from typing import List, Dict
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from sgs.config import settings

def _load_index():
    ip = os.path.join(settings.index_dir, "products.index")
    mp = os.path.join(settings.index_dir, "meta.json")
    if not (os.path.exists(ip) and os.path.exists(mp)):
        return None, None
    return faiss.read_index(ip), json.load(open(mp))

def _load_products():
    return pd.read_csv(settings.data_csv)

def vector_search(query: str, k: int = 5) -> List[Dict]:
    index, meta = _load_index()
    df = _load_products()
    if index is None:
        mask = df["name"].str.contains(query, case=False, na=False) | df["attributes"].str.contains(query, case=False, na=False)
        hits = df[mask].head(k)
        return [{"type":"text","text": f"{r.name} ({r.brand}) - ${r.price:.2f} | {r.category}/{r.sub_category} | attrs: {r.attributes}"} for _, r in hits.iterrows()]
    model = SentenceTransformer(settings.embedding_model)
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, I = index.search(q, k)
    rows = []
    for idx, sc in zip(I[0], scores[0]):
        meta_row = meta["mapping"][idx]
        r = df[df["product_id"] == meta_row["product_id"]].iloc[0]
        rows.append({"type":"text","text": f"{r.name} ({r.brand}) - ${r.price:.2f} | {r.category}/{r.sub_category} | attrs: {r.attributes}", "score": float(sc)})
    return rows
