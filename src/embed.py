# Embedding module for converting text to numerical vectors
# Supports both TF-IDF + SVD and SBERT approaches for semantic representation

import yaml, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

# Generate TF-IDF vectors and reduce dimensionality with SVD
def tfidf_svd(corpus, dims, min_df, max_df):
    tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = tfidf.fit_transform(corpus)
    svd = TruncatedSVD(n_components=dims, random_state=0)
    Z = svd.fit_transform(X)
    return Z, {"tfidf": tfidf, "svd": svd}

# Generate semantic embeddings using sentence transformers
def sbert_embed(corpus, model_name):
    model = SentenceTransformer(model_name)
    Z = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(Z), {"model": model}

if __name__ == "__main__":
    print("Starting embedding generation...")
    cfg = yaml.safe_load(open("src/config.yaml"))
    df = pd.read_parquet("data/interim/clean.parquet")
    print(f"Processing {len(df)} papers...")
    if cfg["use_sbert"]:
        Z, artifacts = sbert_embed(df["text"].tolist(), cfg["sbert_model"])
    else:
        Z, artifacts = tfidf_svd(df["text"].tolist(), cfg["svd_dims"], cfg["min_df"], cfg["max_df"])
    np.save("data/interim/embeddings.npy", Z)
    print("Embedding generation complete!")
