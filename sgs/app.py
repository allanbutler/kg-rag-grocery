# sgs/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import dspy

from sgs.pipeline.program import GroceryRAG
from sgs.config import settings

try:
    dspy.settings.configure(lm=dspy.OpenAI(model=settings.model))
except Exception:
    class EchoLM:
        def __call__(self, *_, **__):
            class R:
                def __getattr__(self, _): return "Configure an LM provider (e.g., OpenAI) for answers."
            return R()
    dspy.settings.configure(lm=EchoLM())

app = FastAPI(title="Smart Grocery Search (KG-RAG + DSPy)")
rag = GroceryRAG()

class AskBody(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(q: str):
    suggestions, ctx = rag.search(q)   # <-- was rag.hybrid(q)
    return {"suggestions": suggestions, "contexts": ctx[:10]}

@app.post("/ask")
def ask(body: AskBody):
    return rag(body.query)
