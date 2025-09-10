import json
import re
from typing import List, Dict

import dspy
import pandas as pd

from sgs.pipeline.signatures import ProductSearchSignature, ProductAnswerSignature
from sgs.retrievers.vector_retriever import vector_search
from sgs.retrievers.kg_retriever import kg_search
from sgs.config import settings


def _parse_budget(query: str) -> float | None:
    """Extract a budget like 'under $5', '<= $4.50', 'under 3 bucks'."""
    q = query.lower()
    m = re.search(r"\$?\s*(\d+(?:\.\d{1,2})?)", q)
    if not m:
        return None
    amt = float(m.group(1))
    if "under" in q or "<" in q or "less" in q or "budget" in q:
        return amt
    return None

def _want_attr(query: str, attr_key: str) -> bool:
    q = query.lower()
    synonyms = {
        "nut_free": ["nut-free", "nut free", "no nuts", "peanut-free", "peanut free"],
        "gluten_free": ["gluten-free", "gluten free"],
        "low_sodium": ["low sodium", "less sodium"],
        "low_sugar": ["low sugar", "no sugar", "zero sugar", "unsweetened"],
        "high_protein": ["high protein", "protein"],
        "vegan": ["vegan", "plant-based", "plant based"],
        "vegetarian": ["vegetarian"],
        "caffeinated": ["caffeinated", "with caffeine"],
        "zero_sugar": ["zero sugar", "no sugar", "unsweetened"],
    }
    for phrase in synonyms.get(attr_key, []):
        if phrase in q:
            return True
    return False


class HybridSearchProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search_llm = dspy.Predict(ProductSearchSignature)
        # Load product table once (for nice names/attrs/price)
        self.df = pd.read_csv(settings.data_csv)

    def _enrich_from_df(self, product_id: int | str | None = None, name: str | None = None) -> Dict:
        if product_id is not None:
            try:
                pid = int(str(product_id))
                row = self.df[self.df["product_id"] == pid].iloc[0]
                return {
                    "product_id": int(row["product_id"]),
                    "product": str(row["name"]),          # <-- FIX: use column, not row.name
                    "brand": str(row["brand"]),
                    "price": float(row["price"]),
                    "category": str(row["category"]),
                    "sub_category": str(row["sub_category"]),
                    "attributes": str(row.get("attributes", "") or ""),
                }
            except Exception:
                pass
        if name is not None:
            hits = self.df[self.df["name"].str.lower() == str(name).lower()]
            if len(hits):
                row = hits.iloc[0]
                return {
                    "product_id": int(row["product_id"]),
                    "product": str(row["name"]),          # <-- FIX here too
                    "brand": str(row["brand"]),
                    "price": float(row["price"]),
                    "category": str(row["category"]),
                    "sub_category": str(row["sub_category"]),
                    "attributes": str(row.get("attributes", "") or ""),
                }
        return {}

    def forward(self, query: str):
        vec = vector_search(query, k=8)
        kg = kg_search(query, k=12)

        # Build unified context and candidate list
        ctx: List[str] = []
        candidates: Dict[str, Dict] = {}  # key by normalized product name

        # From vector hits (free text lines like "Name (Brand) - $4.79 | cat/sub | attrs: ...")
        for item in vec:
            if item.get("type") != "text":
                continue
            text = item["text"]
            ctx.append(text)
            m = re.match(r"(.+?) \((.+?)\) - \$([\d.]+)", text)
            if m:
                name, brand, price = m.group(1).strip(), m.group(2).strip(), float(m.group(3))
                key = name.lower()
                base = {"product": name, "brand": brand, "price": price, "source": "vector"}
                base |= self._enrich_from_df(name=name)
                candidates.setdefault(key, base)

        # From KG hits (structured product cards and graph facts)
        for item in kg:
            t = item.get("type")
            if t == "graph_fact":
                ctx.append(item["text"])
            elif t == "product":
                p = item["payload"]
                # p has product_id, name, brand, price, etc.
                info = self._enrich_from_df(product_id=p.get("product_id")) or {
                    "product_id": int(str(p.get("product_id"))),
                    "product": p.get("name"),
                    "brand": p.get("brand"),
                    "price": float(p.get("price", 0.0)),
                    "category": p.get("category"),
                    "sub_category": p.get("sub_category"),
                    "attributes": "",
                }
                ctx.append(
                    f"PRODUCT: {info['product']} | brand={info['brand']} | "
                    f"cat={info['category']}/{info['sub_category']} | price=${info['price']:.2f}"
                )
                prod_name = str(info.get("product", "")).strip()
                if prod_name:
                    candidates.setdefault(prod_name.lower(), {**info, "source": "kg"})

        # Lightweight constraint handling
        budget = _parse_budget(query)
        want_attrs = [a for a in ["nut_free", "gluten_free", "low_sodium", "low_sugar",
                                  "high_protein", "vegan", "vegetarian", "zero_sugar"]
                      if _want_attr(query, a)]

        def _score(c: Dict) -> tuple:
            # Prefer KG matches, then lower price, then vector score if present
            source_rank = 0 if c.get("source") == "kg" else 1
            price = c.get("price", 9999.0)
            return (source_rank, price)

        filtered = list(candidates.values())
        if budget is not None:
            filtered = [c for c in filtered if c.get("price", 9999) <= budget]
        for attr in want_attrs:
            filtered = [c for c in filtered if attr in (c.get("attributes", "") or "").lower()]

        filtered.sort(key=_score)
        top = [
            {
                "product": c.get("product"),
                "brand": c.get("brand"),
                "price": c.get("price"),
                "category": c.get("category"),
                "sub_category": c.get("sub_category"),
                "source": c.get("source"),
                "attributes": c.get("attributes"),
            }
            for c in filtered[:5]
        ]

        # Try DSPy; if LM configured, we’ll let it rewrite suggestions.
        try:
            pred = dspy.Predict(ProductSearchSignature)(
                query=query, hybrid_context=ctx[:20]
            )
            if getattr(pred, "suggestions", None):
                # If it returns valid JSON, use it; otherwise keep our top list.
                try:
                    llm_suggestions = json.loads(pred.suggestions)
                    if isinstance(llm_suggestions, list) and llm_suggestions:
                        top = llm_suggestions
                except Exception:
                    pass
        except Exception:
            pass

        return top, ctx

class QAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer_llm = dspy.ChainOfThought(ProductAnswerSignature)

    def forward(self, query: str, contexts: List[str]):
        try:
            return self.answer_llm(query=query, contexts=contexts[:20]).answer
        except Exception:
            bullets = "\n- ".join(contexts[:5]) if contexts else "No context."
            return f"Suggested options based on retrieved context:\n- {bullets}"

class GroceryRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.hybrid_prog = HybridSearchProgram()
        self.qa = QAProgram()

    def search(self, query: str):
        return self.hybrid_prog(query)  # use module call (no forward warning)

    def forward(self, query: str):
        suggestions, ctx = self.hybrid_prog(query)
        answer = self.qa(query, ctx)
        return {"suggestions": suggestions, "answer": answer, "contexts": ctx[:10]}
