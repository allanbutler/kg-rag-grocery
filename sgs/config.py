from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "sgs" / "data"

@dataclass
class Settings:
    model: str = os.getenv("DSPY_MODEL", "gpt-4o-mini")  # swap to your provider if needed
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    use_neo4j: bool = os.getenv("USE_NEO4J", "false").lower() == "true"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    data_csv: str = os.getenv("DATA_CSV", str(DATA_DIR / "sample_products.csv"))
    index_dir: str = os.getenv("INDEX_DIR", str(ARTIFACTS / "index"))
    graph_path: str = os.getenv("GRAPH_PATH", str(ARTIFACTS / "graph.pkl"))

settings = Settings()
