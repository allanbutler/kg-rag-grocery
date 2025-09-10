from sgs.pipeline.program import GroceryRAG

def test_pipeline_smoke():
    rag = GroceryRAG()
    res = rag("nut-free granola under $5")
    assert "suggestions" in res and "contexts" in res
