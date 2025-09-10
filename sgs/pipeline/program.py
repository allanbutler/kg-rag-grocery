# sgs/pipeline/program.py
import json
import dspy
from typing import List

from sgs.pipeline.signatures import ProductSearchSignature, ProductAnswerSignature
from sgs.retrievers.vector_retriever import vector_search
from sgs.retrievers.kg_retriever import kg_search

class HybridSearchProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search_llm = dspy.Predict(ProductSearchSignature)

    def forward(self, query: str):
        vec = vector_search(query, k=6)
        kg  = kg_search(query, k=6)

        # Build unified textual context (for LLM and for fallback)
        ctx = []
        product_cards = []
        for item in vec + kg:
            t = item.get("type")
            if t in ("text", "graph_fact"):
                ctx.append(item["text"])
            elif t == "product":
                p = item["payload"]
                product_cards.append(p)
                ctx.append(
                    f"PRODUCT: {p['name']} | brand={p['brand']} | "
                    f"cat={p['category']}/{p['sub_category']} | price=${p['price']:.2f}"
                )

        # Try DSPy → if LM is missing/misconfigured, fall back to a heuristic
        suggestions = None
        try:
            pred = self.search_llm(query=query, hybrid_context=ctx[:20])
            raw = getattr(pred, "suggestions", None)
            if raw:
                suggestions = json.loads(raw)
        except Exception:
            suggestions = None

        if suggestions is None:
            # Heuristic fallback: prefer product cards, then top text hits
            if product_cards:
                suggestions = [
                    {
                        "product": pc["name"],
                        "brand": pc["brand"],
                        "price": pc["price"],
                        "why": "matches query via KG/vector",
                    }
                    for pc in product_cards[:5]
                ]
            else:
                suggestions = [
                    {"product": c, "why": "relevant snippet"} for c in ctx[:5]
                ]

        return suggestions, ctx

class QAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer_llm = dspy.ChainOfThought(ProductAnswerSignature)

    def forward(self, query: str, contexts: List[str]):
        # Try to answer with DSPy; if LM missing, return a simple stitched answer.
        try:
            return self.answer_llm(query=query, contexts=contexts[:20]).answer
        except Exception:
            bullet_points = "\n- ".join(contexts[:5]) if contexts else "No context."
            return f"Suggested options based on retrieved context:\n- {bullet_points}"

class GroceryRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.hybrid_prog = HybridSearchProgram()
        self.qa = QAProgram()

    def search(self, query: str):
        # use module call (avoids the 'direct forward' warning)
        return self.hybrid_prog(query)

    def forward(self, query: str):
        suggestions, ctx = self.hybrid_prog(query)
        answer = self.qa(query, ctx)
        return {"suggestions": suggestions, "answer": answer, "contexts": ctx[:10]}
