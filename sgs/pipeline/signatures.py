import dspy

class ProductSearchSignature(dspy.Signature):
    """Return product suggestions and key facts based on a grocery search query."""
    query = dspy.InputField(desc="User question or need (e.g., 'nut-free granola under $5')")
    hybrid_context = dspy.InputField(desc="Snippets from vector and KG retrievers")
    suggestions = dspy.OutputField(desc="JSON list of product suggestions with short reasons")

class ProductAnswerSignature(dspy.Signature):
    """Answer a grocery question grounded in provided contexts (include brief citations like [1])."""
    query = dspy.InputField()
    contexts = dspy.InputField()
    answer = dspy.OutputField()
