# Topic labeling module for generating human-readable cluster names
# Uses enhanced TF-IDF with stopword filtering and background contrast for better labels

import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# domain n English stopwords to filter out generic terms
CUSTOM_STOP = {
    "with","on","are","by","this","as","an","from","at","be","our","which",
    "using","based","approach","paper","method","model","results","data",
    "propose","show","study","analysis","work","problem","new","task","system"
}

# Extract meaningful terms using background contrast and enhanced filtering
def top_terms_per_cluster(texts, labels, k=12):
    print("Building TF-IDF matrix with enhanced filtering...")
    # remove short tokens, keep words ≥3 letters; include bigrams
    vec = TfidfVectorizer(
        min_df=10, max_df=0.6,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        ngram_range=(1,2)
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    print(f"Initial vocabulary: {len(terms)} terms")

    # drop any terms in CUSTOM_STOP if present
    if CUSTOM_STOP:
        keep = np.array([t not in CUSTOM_STOP for t in terms])
        X = X[:, keep]
        terms = terms[keep]
        print(f"After custom stopword filtering: {len(terms)} terms")

    labels = np.asarray(labels)
    names = {}

    # background mean to penalize generic terms
    print("Computing background term frequencies...")
    bg = X.mean(axis=0).A1

    print("Extracting distinctive terms for each cluster...")
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 200:  # skip tiny clusters
            continue
        idx = idx[idx < X.shape[0]]
        mu = X[idx].mean(axis=0).A1

        # contrast against background, add small eps for stability
        score = mu - bg
        top = terms[np.argsort(-score)[:k]]
        names[int(c)] = ", ".join(top)

    return names

if __name__ == "__main__":
    print("Generating cluster labels...")
    df = pd.read_parquet("data/interim/clean.parquet")
    y  = pd.read_csv("data/interim/labels.csv", header=None)[0]

    n = min(len(df), len(y))
    print(f"Aligning lengths: texts={len(df)}, labels={len(y)}, using={n}")
    df = df.iloc[:n].copy()
    y  = y.iloc[:n].to_numpy()

    names = top_terms_per_cluster(df["text"], y, k=12)
    pd.Series(names).to_csv("outputs/tables/cluster_labels.csv")
    print(f"Generated labels for {len(names)} clusters")
