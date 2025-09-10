import os, json, pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from sgs.config import settings

def build_vector_index(df: pd.DataFrame, out_dir: str):
    model = SentenceTransformer(settings.embedding_model)
    texts = (df["name"] + " | " + df["brand"] + " | " + df["category"] + " | " + df["sub_category"] + " | " + df["ingredients"].fillna("") + " | " + df["attributes"].fillna("") + " | " + df["nutrition_text"].fillna("")).tolist()
    embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs.astype("float32"))
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(idx, os.path.join(out_dir, "products.index"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"mapping": df[["product_id","name"]].to_dict(orient="records")}, f)
    print(f"Built index: {len(texts)} vectors → {out_dir}")

def main():
    df = pd.read_csv(settings.data_csv)
    build_vector_index(df, settings.index_dir)

if __name__ == "__main__":
    main()
